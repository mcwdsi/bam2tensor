"""Tests for the opt-in non-converted and EM over-conversion read filters."""

import warnings

import numpy as np
import pysam
import pytest
import scipy.sparse

from bam2tensor import __version__, embedding, functions
from bam2tensor.functions import (
    count_non_cpg_retained_xm,
    count_non_cpg_retained_reference,
    is_em_overconversion_read,
)
from bam2tensor.metadata import (
    compute_cpg_index_crc32,
    read_npz_metadata,
    read_npz_tlen,
    write_npz_metadata,
    write_npz_tlen,
)
from tests.paths import TEST_BAM, TEST_FASTA

# ---------------------------------------------------------------------------
# Helpers: build a tmp FASTA + BAM with one or more reads
# ---------------------------------------------------------------------------


def _write_fasta(path, seq: str) -> int:
    """Write a single-contig FASTA and return its length."""
    with open(path, "w") as f:
        f.write(">chr1\n")
        f.write(seq + "\n")
    return len(seq)


def _make_embedding(tmp_path, seq: str, name: str):
    fasta_path = tmp_path / "ref.fa"
    _write_fasta(fasta_path, seq)
    return embedding.GenomeMethylationEmbedding(
        genome_name=name,
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
        verbose=False,
    )


def _write_bam(tmp_path, seq_len: int, reads: list[dict]) -> str:
    """Build an indexed BAM in tmp_path from a list of read spec dicts.

    Each spec may set: query_name, query_sequence, reference_start,
    flag, cigar (list of (op, len) tuples), md, xm, yd.
    """
    bam_path = tmp_path / "test.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": seq_len, "SN": "chr1"}]}
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as out_bam:
        for spec in reads:
            a = pysam.AlignedSegment()
            a.query_name = spec["query_name"]
            a.query_sequence = spec["query_sequence"]
            a.flag = spec.get("flag", 0)
            a.reference_id = 0
            a.reference_start = spec.get("reference_start", 0)
            a.mapping_quality = spec.get("mapping_quality", 60)
            a.cigartuples = spec.get("cigar", [(0, len(spec["query_sequence"]))])
            if "md" in spec:
                a.set_tag("MD", spec["md"])
            if "xm" in spec:
                a.set_tag("XM", spec["xm"])
            if "yd" in spec:
                a.set_tag("YD", spec["yd"])
            out_bam.write(a)
    pysam.index(str(bam_path))
    return str(bam_path)


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


def test_count_non_cpg_retained_xm_counts_H_X_U() -> None:
    assert count_non_cpg_retained_xm("..Z..hhh..HHH..z..") == 3
    assert count_non_cpg_retained_xm("..Z..HHXXUU..z..") == 6
    assert count_non_cpg_retained_xm("........") == 0


def test_is_em_overconversion_read_requires_min_cpgs() -> None:
    assert is_em_overconversion_read([0, 0, 0], min_cpgs=3) is True
    assert is_em_overconversion_read([0, 0, 0, 0], min_cpgs=3) is True
    assert is_em_overconversion_read([0, 0, 1], min_cpgs=3) is False
    # Below threshold: don't flag, even if all-zero.
    assert is_em_overconversion_read([0, 0], min_cpgs=3) is False
    # -1 (no-data) is not "unmethylated".
    assert is_em_overconversion_read([0, 0, -1], min_cpgs=3) is False
    assert is_em_overconversion_read([], min_cpgs=1) is False


# ---------------------------------------------------------------------------
# Non-converted filter (Bismark XM-tag path)
# ---------------------------------------------------------------------------


def test_non_converted_filter_bismark_drops_retained(tmp_path) -> None:
    """Bismark read with 3 retained non-CpG Cs is dropped when filter is on."""
    # Reference: CpG at pos 10-11 (1-based)
    ref_seq = "N" * 9 + "CG" + "N" * 19
    seq_len = len(ref_seq)
    emb = _make_embedding(tmp_path, ref_seq, "bismark_drop")

    # Read covers entire contig; XM has 3 H's (retained CHH) plus one Z.
    query = "A" * seq_len
    xm = "." * 9 + "Z." + "HHH" + "." * (seq_len - 14)
    assert len(xm) == seq_len

    bam_path = _write_bam(
        tmp_path,
        seq_len,
        [
            {
                "query_name": "read_retained",
                "query_sequence": query,
                "reference_start": 0,
                "xm": xm,
                "md": str(seq_len),
            }
        ],
    )

    # Filter on: dropped
    r = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_non_converted=True,
        non_converted_threshold=3,
    )
    assert r.matrix.shape[0] == 0
    assert len(r.tlen) == 0

    # Filter off: kept
    r = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_non_converted=False,
    )
    assert r.matrix.shape[0] == 1
    assert len(r.tlen) == 1


def test_non_converted_filter_bismark_below_threshold_keeps(tmp_path) -> None:
    """Bismark read with 2 retained non-CpG Cs is kept (below threshold)."""
    ref_seq = "N" * 9 + "CG" + "N" * 19
    seq_len = len(ref_seq)
    emb = _make_embedding(tmp_path, ref_seq, "bismark_keep")

    query = "A" * seq_len
    xm = "." * 9 + "Z." + "HH" + "." * (seq_len - 13)
    assert len(xm) == seq_len

    bam_path = _write_bam(
        tmp_path,
        seq_len,
        [
            {
                "query_name": "read_subthreshold",
                "query_sequence": query,
                "reference_start": 0,
                "xm": xm,
                "md": str(seq_len),
            }
        ],
    )

    r = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_non_converted=True,
        non_converted_threshold=3,
    )
    assert r.matrix.shape[0] == 1
    assert len(r.tlen) == 1


# ---------------------------------------------------------------------------
# Non-converted filter (Biscuit/bwameth/gem3 reference-validated path)
# ---------------------------------------------------------------------------


def test_non_converted_filter_biscuit_drops_retained(tmp_path) -> None:
    """Biscuit forward-strand read with 4 retained non-CpG Cs is dropped."""
    # Ref: 4 isolated Cs at pos 6-9 (1-based), CpG at pos 15-16.
    ref_seq = "N" * 5 + "CCCC" + "N" * 5 + "CG" + "N" * 6
    seq_len = len(ref_seq)
    assert seq_len == 22
    emb = _make_embedding(tmp_path, ref_seq, "biscuit_drop")

    # Read matches reference exactly -> all 4 Cs retained, MD is "22".
    query = ref_seq.replace("N", "A")  # pysam requires a real sequence

    bam_path = _write_bam(
        tmp_path,
        seq_len,
        [
            {
                "query_name": "read_retained_biscuit",
                "query_sequence": query,
                "reference_start": 0,
                "yd": "f",
                "md": str(seq_len),
            }
        ],
    )

    # Sanity: count directly from the aligned segment.
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        seg = next(bam.fetch("chr1"))
    assert count_non_cpg_retained_reference(seg, is_reverse_parent_strand=False) == 4

    # Filter on: dropped
    r = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_non_converted=True,
        non_converted_threshold=3,
    )
    assert r.matrix.shape[0] == 0

    # Filter off: kept
    r = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_non_converted=False,
    )
    assert r.matrix.shape[0] == 1


def test_non_converted_filter_biscuit_snps_rejected(tmp_path) -> None:
    """Candidate Cs that are actually SNPs against ref are not counted."""
    # Ref: only 1 real non-CpG C at pos 6, SNPs at pos 7 and 8.
    ref_seq = "N" * 5 + "CAT" + "N" * 6 + "CG" + "N" * 6
    seq_len = len(ref_seq)
    assert seq_len == 22
    emb = _make_embedding(tmp_path, ref_seq, "biscuit_snp")

    # Read has C at 6, 7, 8 — positions 7 (ref=A) and 8 (ref=T) are SNPs.
    # Everything else matches.
    query_list = list(ref_seq.replace("N", "A"))
    query_list[6] = "C"
    query_list[7] = "C"
    query = "".join(query_list)
    # MD: 6 matches, A mismatch (ref=A, query=C), 0 matches, T mismatch
    # (ref=T, query=C), remaining matches.
    remaining = seq_len - 8
    md = f"6A0T{remaining}"

    bam_path = _write_bam(
        tmp_path,
        seq_len,
        [
            {
                "query_name": "read_with_snps",
                "query_sequence": query,
                "reference_start": 0,
                "yd": "f",
                "md": md,
            }
        ],
    )

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        seg = next(bam.fetch("chr1"))
    # Only pos 6 is a validated retained C; pos 7/8 are SNPs.
    assert count_non_cpg_retained_reference(seg, is_reverse_parent_strand=False) == 1

    # Threshold 3, validated count 1 -> kept.
    r = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_non_converted=True,
        non_converted_threshold=3,
    )
    assert r.matrix.shape[0] == 1


# ---------------------------------------------------------------------------
# EM over-conversion filter
# ---------------------------------------------------------------------------


def _make_multi_cpg_embedding(tmp_path, name: str):
    """Reference with three CpGs spaced across 30bp."""
    # CpGs at 1-based positions 5-6, 15-16, 25-26.
    ref_seq = "N" * 4 + "CG" + "N" * 8 + "CG" + "N" * 8 + "CG" + "N" * 4
    return _make_embedding(tmp_path, ref_seq, name), ref_seq


def _query_for_ref_with_cpg_bases(ref_seq: str, cpg_base_for_c: str) -> str:
    """Build a query sequence: N→A, C in CpG→cpg_base_for_c, G→G (match)."""
    out = []
    i = 0
    while i < len(ref_seq):
        c = ref_seq[i]
        if c == "C" and i + 1 < len(ref_seq) and ref_seq[i + 1] == "G":
            out.append(cpg_base_for_c)
            out.append("G")
            i += 2
        elif c == "N":
            out.append("A")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def test_em_overconversion_filter_drops_all_unmethylated(tmp_path) -> None:
    """Read with 3 CpGs all unmethylated (T) is dropped when filter is on."""
    emb, ref_seq = _make_multi_cpg_embedding(tmp_path, "em_drop")
    seq_len = len(ref_seq)

    # Replace each CpG C with T in the query -> all 3 CpGs call as unmethylated.
    query = _query_for_ref_with_cpg_bases(ref_seq, "T")

    bam_path = _write_bam(
        tmp_path,
        seq_len,
        [
            {
                "query_name": "read_all_unmethylated",
                "query_sequence": query,
                "reference_start": 0,
                "yd": "f",
                # 3 C→T mismatches at positions 4, 14, 24 (0-based).
                "md": "4C9C9C4",
            }
        ],
    )

    # Filter on: dropped
    r = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_em_overconversion=True,
        em_overconversion_min_cpgs=3,
    )
    assert r.matrix.shape[0] == 0
    assert len(r.tlen) == 0

    # Filter off: kept
    r = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_em_overconversion=False,
    )
    assert r.matrix.shape[0] == 1
    assert r.matrix.nnz == 3
    # All 3 values are 0 (unmethylated).
    assert set(r.matrix.data.tolist()) == {0}


def test_em_overconversion_filter_keeps_mixed(tmp_path) -> None:
    """Read with one methylated CpG is not dropped."""
    emb, ref_seq = _make_multi_cpg_embedding(tmp_path, "em_mixed")
    seq_len = len(ref_seq)

    # Unmethylated at first and third CpG (T), methylated at middle CpG (C).
    q_list = list(_query_for_ref_with_cpg_bases(ref_seq, "T"))
    # Middle CpG C is at 0-based position 14.
    q_list[14] = "C"
    query = "".join(q_list)

    bam_path = _write_bam(
        tmp_path,
        seq_len,
        [
            {
                "query_name": "read_mixed",
                "query_sequence": query,
                "reference_start": 0,
                "yd": "f",
                # Mismatches only at pos 4 and 24 (middle CpG matches ref's C).
                "md": f"4C19C{seq_len - 25}",
            }
        ],
    )

    r = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_em_overconversion=True,
        em_overconversion_min_cpgs=3,
    )
    assert r.matrix.shape[0] == 1
    assert sorted(r.matrix.data.tolist()) == [0, 0, 1]


def test_em_overconversion_filter_respects_min_cpgs(tmp_path) -> None:
    """With min_cpgs=6, a 3-CpG all-unmethylated read is kept."""
    emb, ref_seq = _make_multi_cpg_embedding(tmp_path, "em_minkeep")
    seq_len = len(ref_seq)

    query = _query_for_ref_with_cpg_bases(ref_seq, "T")

    bam_path = _write_bam(
        tmp_path,
        seq_len,
        [
            {
                "query_name": "read_below_min",
                "query_sequence": query,
                "reference_start": 0,
                "yd": "f",
                "md": "4C9C9C4",
            }
        ],
    )

    r = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_em_overconversion=True,
        em_overconversion_min_cpgs=6,
    )
    assert r.matrix.shape[0] == 1


# ---------------------------------------------------------------------------
# tlen and ExtractionResult stay in sync with drops
# ---------------------------------------------------------------------------


def test_drops_keep_tlen_in_sync(tmp_path) -> None:
    """When the EM filter drops one read of two, tlen and matrix rows match."""
    emb, ref_seq = _make_multi_cpg_embedding(tmp_path, "em_tlen")
    seq_len = len(ref_seq)

    keeper_query = _query_for_ref_with_cpg_bases(ref_seq, "C")  # all methylated
    dropper_query = _query_for_ref_with_cpg_bases(ref_seq, "T")  # all unmethylated

    bam_path = _write_bam(
        tmp_path,
        seq_len,
        [
            {
                "query_name": "keeper",
                "query_sequence": keeper_query,
                "reference_start": 0,
                "yd": "f",
                "md": str(seq_len),  # keeper matches ref at CpG Cs
            },
            {
                "query_name": "dropper",
                "query_sequence": dropper_query,
                "reference_start": 0,
                "yd": "f",
                "md": "4C9C9C4",
            },
        ],
    )

    r = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_em_overconversion=True,
        em_overconversion_min_cpgs=3,
    )
    assert r.matrix.shape[0] == 1
    assert len(r.tlen) == 1
    assert r.tlen.dtype == np.int32


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------


def test_metadata_round_trip(tmp_path) -> None:
    """Filter settings written by the CLI flow round-trip through read_npz_metadata."""
    emb, ref_seq = _make_multi_cpg_embedding(tmp_path, "em_meta")
    seq_len = len(ref_seq)
    query = _query_for_ref_with_cpg_bases(ref_seq, "C")  # all methylated, all kept

    bam_path = _write_bam(
        tmp_path,
        seq_len,
        [
            {
                "query_name": "meta_read",
                "query_sequence": query,
                "reference_start": 0,
                "yd": "f",
                "md": str(seq_len),
            }
        ],
    )

    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_non_converted=True,
        non_converted_threshold=4,
        filter_em_overconversion=True,
        em_overconversion_min_cpgs=5,
    )

    out_npz = tmp_path / "out.methylation.npz"
    scipy.sparse.save_npz(str(out_npz), result.matrix, compressed=True)
    write_npz_tlen(str(out_npz), result.tlen)
    write_npz_metadata(
        str(out_npz),
        {
            "bam2tensor_version": __version__,
            "genome_name": "em_meta",
            "expected_chromosomes": ["chr1"],
            "total_cpg_sites": emb.total_cpg_sites,
            "cpg_index_crc32": compute_cpg_index_crc32(emb),
            "filters": {
                "non_converted_reads": {
                    "enabled": True,
                    "threshold": 4,
                },
                "em_overconversion": {
                    "enabled": True,
                    "min_cpgs": 5,
                },
            },
        },
    )

    meta = read_npz_metadata(str(out_npz))
    assert meta is not None
    assert meta["filters"]["non_converted_reads"] == {
        "enabled": True,
        "threshold": 4,
    }
    assert meta["filters"]["em_overconversion"] == {
        "enabled": True,
        "min_cpgs": 5,
    }

    # tlen also round-trips (sanity for the v2.4 feature).
    tlen = read_npz_tlen(str(out_npz))
    assert tlen is not None
    assert len(tlen) == result.matrix.shape[0]


# ---------------------------------------------------------------------------
# Regression: default flags produce unchanged output on test.bam
# ---------------------------------------------------------------------------


def test_default_flags_match_pre_change_behavior() -> None:
    """Both filters default to off and output is unchanged vs. historical."""
    emb = embedding.GenomeMethylationEmbedding(
        genome_name="test_regression",
        expected_chromosomes=["chr1", "chr2", "chr3"],
        fasta_source=TEST_FASTA,
        skip_cache=True,
        verbose=False,
    )
    r = functions.extract_methylation_data_from_bam(
        input_bam=TEST_BAM,
        genome_methylation_embedding=emb,
        quality_limit=20,
    )
    # These numbers are the pre-change output of tests/test.bam against
    # tests/test_fasta.fa with default settings.
    assert r.matrix.shape == (1, 37489)
    assert r.matrix.nnz == 3
    assert len(r.tlen) == 1


# ---------------------------------------------------------------------------
# Non-converted filter: reads that cannot be evaluated (no MD tag)
# ---------------------------------------------------------------------------


def _biscuit_non_converted_bam(tmp_path, name: str, with_md: bool, n_reads: int = 1):
    """A Biscuit-style BAM of fully non-converted reads, with or without MD.

    The reference has 4 isolated non-CpG Cs and one CpG; each read matches the
    reference exactly, so all 4 non-CpG Cs are retained (>= the default
    threshold of 3).
    """
    ref_seq = "N" * 5 + "CCCC" + "N" * 5 + "CG" + "N" * 6
    emb = _make_embedding(tmp_path, ref_seq, name)
    spec = {
        "query_sequence": ref_seq.replace("N", "A"),
        "reference_start": 0,
        "yd": "f",
    }
    if with_md:
        spec["md"] = str(len(ref_seq))
    reads = [dict(spec, query_name=f"read{i}") for i in range(n_reads)]
    return emb, _write_bam(tmp_path, len(ref_seq), reads)


def test_non_converted_filter_warns_when_md_tag_missing(tmp_path) -> None:
    """Without MD the filter cannot run; it must say so instead of silently passing.

    Regression: on a BAM without MD tags ``--filter-non-converted`` quietly
    kept every read — indistinguishable from a clean sample.
    """
    emb, bam_path = _biscuit_non_converted_bam(
        tmp_path, "md_missing", with_md=False, n_reads=3
    )

    with pytest.warns(RuntimeWarning) as recorded:
        result = functions.extract_methylation_data_from_bam(
            input_bam=bam_path,
            genome_methylation_embedding=emb,
            filter_non_converted=True,
        )

    # Exactly one warning, reporting how many reads were affected and the fix.
    assert len(recorded) == 1
    message = str(recorded[0].message)
    assert "3 read(s)" in message
    assert bam_path in message
    assert "no MD tag" in message
    assert "samtools calmd" in message

    # The unevaluable reads are kept rather than dropped.
    assert result.matrix.shape[0] == 3


def test_non_converted_filter_md_warning_counts_across_workers(tmp_path) -> None:
    """The count of unevaluable reads is aggregated from worker processes."""
    emb, bam_path = _biscuit_non_converted_bam(
        tmp_path, "md_missing_parallel", with_md=False, n_reads=5
    )
    with pytest.warns(RuntimeWarning, match=r"5 read\(s\).*no MD tag"):
        result = functions.extract_methylation_data_from_bam(
            input_bam=bam_path,
            genome_methylation_embedding=emb,
            filter_non_converted=True,
            threads=2,
        )
    assert result.matrix.shape[0] == 5


def test_no_md_warning_when_md_present_or_filter_off(tmp_path) -> None:
    """No warning when the filter can run, or when it was not requested."""
    emb_md, bam_md = _biscuit_non_converted_bam(tmp_path, "md_ok", with_md=True)
    md_free = tmp_path / "md_free"
    md_free.mkdir()
    emb_no_md, bam_no_md = _biscuit_non_converted_bam(md_free, "md_off", with_md=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # MD present: the filter evaluates (and drops) the read, silently.
        result = functions.extract_methylation_data_from_bam(
            input_bam=bam_md,
            genome_methylation_embedding=emb_md,
            filter_non_converted=True,
        )
        assert result.matrix.shape[0] == 0

        # MD absent but the filter is off: nothing to warn about.
        result = functions.extract_methylation_data_from_bam(
            input_bam=bam_no_md,
            genome_methylation_embedding=emb_no_md,
            filter_non_converted=False,
        )
        assert result.matrix.shape[0] == 1


def test_no_md_warning_for_bismark_reads(tmp_path) -> None:
    """Bismark reads are filtered via XM and never need (or warn about) MD."""
    ref_seq = "N" * 5 + "CCCC" + "N" * 5 + "CG" + "N" * 6
    emb = _make_embedding(tmp_path, ref_seq, "bismark_no_md")
    xm = "." * 5 + "HHHH" + "." * 5 + "Z." + "." * 6
    bam_path = _write_bam(
        tmp_path,
        len(ref_seq),
        [
            {
                "query_name": "bismark_read",
                "query_sequence": ref_seq.replace("N", "A"),
                "reference_start": 0,
                "xm": xm,  # no MD tag
            }
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = functions.extract_methylation_data_from_bam(
            input_bam=bam_path,
            genome_methylation_embedding=emb,
            filter_non_converted=True,
        )
    assert result.matrix.shape[0] == 0  # dropped: 4 retained non-CpG Cs


def test_cli_reports_md_warning_per_bam(tmp_path) -> None:
    """The CLI shows the MD warning in the BAM's own report section."""
    from click.testing import CliRunner

    from bam2tensor import __main__

    _, bam_path = _biscuit_non_converted_bam(
        tmp_path, "md_cli", with_md=False, n_reads=2
    )
    result = CliRunner().invoke(
        __main__.main,
        [
            "--input-path",
            bam_path,
            "--reference-fasta",
            str(tmp_path / "ref.fa"),
            "--genome-name",
            "md_cli",
            "--expected-chromosomes",
            "chr1",
            "--filter-non-converted",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "WARNING: 2 read(s)" in result.output
    assert "no MD tag" in result.output
    # A warning is not an error: the BAM is still processed and written.
    assert "Processed:     1 BAM(s)" in result.output
    assert (tmp_path / "out" / "test.methylation.npz").exists()
