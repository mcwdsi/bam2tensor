import pysam
import pytest
from bam2tensor import embedding
from bam2tensor import functions
from bam2tensor.functions import check_chromosome_overlap
from bam2tensor.functions import detect_aligner

TEST_EMBEDDING = embedding.GenomeMethylationEmbedding(
    "test_genome",
    expected_chromosomes=["chr1", "chr2", "chr3"],
    fasta_source="tests/test_fasta.fa",
    window_size=150,
    skip_cache=False,
    verbose=True,
)


def test_extract_methylation_data_from_bam() -> None:
    """Test extract_methylation_data_from_bam."""

    test_coo_matrix = functions.extract_methylation_data_from_bam(
        input_bam="tests/test.bam",
        genome_methylation_embedding=TEST_EMBEDDING,
        quality_limit=20,
        verbose=True,
        debug=False,
    )

    # Need to make a better test bam & fa pair...
    assert test_coo_matrix.shape == (1, 37489)


def test_extract_methylation_data_from_bam_debug() -> None:
    """Test extract_methylation_data_from_bam with debug flag set."""

    test_coo_matrix = functions.extract_methylation_data_from_bam(
        input_bam="tests/test.bam",
        genome_methylation_embedding=TEST_EMBEDDING,
        quality_limit=20,
        verbose=True,
        debug=True,
    )

    assert test_coo_matrix.shape == (1, 37489)


def test_check_chromosome_overlap_matching() -> None:
    """No error when chromosomes overlap."""
    check_chromosome_overlap(
        bam_references=("chr1", "chr2", "chr3", "chrM"),
        embedding_chromosomes=["chr1", "chr2", "chr3"],
    )


def test_check_chromosome_overlap_partial() -> None:
    """No error when there is partial overlap."""
    check_chromosome_overlap(
        bam_references=("chr1", "chr2", "chrM", "chrUn_gl000220"),
        embedding_chromosomes=["chr1", "chr2", "chr3"],
    )


def test_check_chromosome_overlap_bam_missing_chr_prefix() -> None:
    """Error when BAM uses Ensembl-style (no chr) and embedding uses UCSC-style."""
    with pytest.raises(ValueError, match="Chromosome name mismatch"):
        check_chromosome_overlap(
            bam_references=("1", "2", "3", "MT"),
            embedding_chromosomes=["chr1", "chr2", "chr3"],
        )


def test_check_chromosome_overlap_bam_has_chr_prefix() -> None:
    """Error when BAM uses UCSC-style (chr) and embedding uses Ensembl-style."""
    with pytest.raises(ValueError, match="Chromosome name mismatch"):
        check_chromosome_overlap(
            bam_references=("chr1", "chr2", "chr3"),
            embedding_chromosomes=["1", "2", "3"],
        )


def test_check_chromosome_overlap_completely_different() -> None:
    """Error with descriptive message for completely unrelated names."""
    with pytest.raises(ValueError, match="No overlapping chromosome names"):
        check_chromosome_overlap(
            bam_references=("scaffold_1", "scaffold_2"),
            embedding_chromosomes=["chr1", "chr2"],
        )


def test_chr_prefix_mismatch_in_extraction(tmp_path) -> None:
    """Integration test: extract_methylation_data_from_bam detects chr prefix mismatch."""
    # Create a FASTA with chr1
    fasta_path = tmp_path / "ref.fa"
    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write("ACGTCGACGT" * 15 + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_chr_mismatch",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    # Create BAM with Ensembl-style chromosome name "1" (no chr prefix)
    seq = "ACGTCGACGT" * 15
    bam_path = tmp_path / "test.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": len(seq), "SN": "1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = seq
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, len(seq))]
        a.set_tag("MD", str(len(seq)))
        a.set_tag("YD", "f")
        out_bam.write(a)

    pysam.index(str(bam_path))

    with pytest.raises(ValueError, match="Chromosome name mismatch"):
        functions.extract_methylation_data_from_bam(
            input_bam=str(bam_path),
            genome_methylation_embedding=emb,
        )


def _make_bam_with_reads(tmp_path, reads, seq_len=150):
    """Helper: create a BAM + index with given reads on chr1.

    Each entry in reads is a dict with keys:
        name, flag, mapq, tags (list of (tag, value) tuples)
        Optional: seq (defaults to 'N' * seq_len), start (defaults to 0)
    """
    fasta_path = tmp_path / "ref.fa"
    if not fasta_path.exists():
        # Place CpGs at positions 10, 20, 100, 110 (1-based)
        seq = (
            "N" * 9
            + "CG"
            + "N" * 9
            + "CG"
            + "N" * 79
            + "CG"
            + "N" * 9
            + "CG"
            + "N" * 40
        )
        with open(fasta_path, "w") as f:
            f.write(">chr1\n" + seq + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_perf",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    bam_path = tmp_path / "test.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": seq_len, "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        for r in reads:
            a = pysam.AlignedSegment()
            a.query_name = r["name"]
            a.query_sequence = r.get("seq", "N" * seq_len)
            a.flag = r["flag"]
            a.reference_id = 0
            a.reference_start = r.get("start", 0)
            a.mapping_quality = r["mapq"]
            a.cigartuples = [(0, seq_len)]
            for tag, val in r.get("tags", []):
                a.set_tag(tag, val)
            out_bam.write(a)

    pysam.index(str(bam_path))
    return emb, str(bam_path)


# ======================================================================
# B: Bitwise flag filtering tests
# ======================================================================


def test_duplicate_reads_skipped(tmp_path):
    """Reads marked as duplicates (flag 0x400) are excluded."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "dup_read",
                "flag": 0x400,  # duplicate
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 0


def test_qcfail_reads_skipped(tmp_path):
    """Reads marked as QC-failed (flag 0x200) are excluded."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "qcfail_read",
                "flag": 0x200,  # qcfail
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 0


def test_secondary_reads_skipped(tmp_path):
    """Reads marked as secondary (flag 0x100) are excluded."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "secondary_read",
                "flag": 0x100,  # secondary
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 0


def test_supplementary_reads_skipped(tmp_path):
    """Reads marked as supplementary (flag 0x800) are excluded."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "supplementary_read",
                "flag": 0x800,  # supplementary
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 0


def test_combined_skip_flags(tmp_path):
    """Reads with multiple skip flags set are excluded."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "multi_flag_read",
                "flag": 0x400 | 0x200,  # duplicate + qcfail
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 0


def test_normal_read_not_skipped(tmp_path):
    """A normal forward-strand read (flag=0) with YD=f is processed."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "good_read",
                "flag": 0,
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 1


def test_paired_read_flag_not_skipped(tmp_path):
    """Paired-end flag (0x1) and proper-pair (0x2) should NOT be skipped."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "paired_read",
                "flag": 0x1 | 0x2,  # paired + proper pair
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 1


# ======================================================================
# C: Strand check early-exit tests
# ======================================================================


def test_daughter_strand_skipped_yd(tmp_path):
    """Forward read with YD=r (reverse parent) is on daughter strand and skipped."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "daughter_yd",
                "flag": 0,  # forward
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "r")],  # reverse parent strand
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 0


def test_parent_strand_kept_yd_reverse(tmp_path):
    """Reverse read with YD=r is on the parent strand and processed."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "parent_rev",
                "flag": 0x10,  # reverse strand
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "r")],  # reverse parent strand
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 1


def test_daughter_strand_skipped_xb(tmp_path):
    """Forward read with XB=G (reverse parent) is on daughter strand and skipped."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "daughter_xb",
                "flag": 0,  # forward
                "mapq": 60,
                "tags": [("MD", "150"), ("XB", "G")],  # reverse parent strand
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 0


def test_parent_strand_kept_xb_forward(tmp_path):
    """Forward read with XB=C (forward parent) is on the parent strand."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "parent_xb",
                "flag": 0,  # forward
                "mapq": 60,
                "tags": [("MD", "150"), ("XB", "C")],  # forward parent strand
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 1


def test_no_strand_tag_skipped(tmp_path):
    """Read with no YD or XB tag has bisulfite_parent_strand_is_reverse=None, skipped."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "no_tag_read",
                "flag": 0,
                "mapq": 60,
                "tags": [("MD", "150")],  # No YD or XB
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    # bisulfite_parent_strand_is_reverse=None != is_reverse=False, so skipped
    assert matrix.shape[0] == 0


def test_low_mapq_skipped_before_strand_check(tmp_path):
    """Low MAPQ read is filtered before strand/CpG logic runs."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "low_mapq",
                "flag": 0,
                "mapq": 5,  # below default threshold of 20
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 0


def test_mixed_reads_correct_count(tmp_path):
    """Mix of good, flagged, and daughter-strand reads yields correct count."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            # Good: forward read, YD=f, parent strand
            {
                "name": "good1",
                "flag": 0,
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
            # Good: reverse read, YD=r, parent strand
            {
                "name": "good2",
                "flag": 0x10,
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "r")],
            },
            # Bad: duplicate
            {
                "name": "bad_dup",
                "flag": 0x400,
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
            # Bad: daughter strand (forward read + YD=r)
            {
                "name": "bad_daughter",
                "flag": 0,
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "r")],
            },
            # Bad: low MAPQ
            {
                "name": "bad_mapq",
                "flag": 0,
                "mapq": 1,
                "tags": [("MD", "150"), ("YD", "f")],
            },
            # Bad: secondary
            {
                "name": "bad_secondary",
                "flag": 0x100,
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 2  # only good1 and good2


def test_skip_flags_constant():
    """Verify _SKIP_FLAGS matches the expected BAM flag bits."""
    from bam2tensor.functions import _SKIP_FLAGS

    assert _SKIP_FLAGS == 0x400 | 0x200 | 0x100 | 0x800
    assert _SKIP_FLAGS == 0xF00


# ======================================================================
# Bismark XM tag tests
# ======================================================================


def _make_bismark_bam(tmp_path, reads, seq_len=154):
    """Helper: create a BAM with Bismark-style XM tags.

    The reference FASTA has CpGs at 0-based C positions 9, 20, 101, 112
    (1-based: 10, 21, 102, 113). Sequence is 154bp.
    Each read dict needs: name, flag, mapq, xm_string.
    Optional: seq, start, extra_tags.
    """
    fasta_path = tmp_path / "ref.fa"
    if not fasta_path.exists():
        seq = (
            "N" * 9
            + "CG"
            + "N" * 9
            + "CG"
            + "N" * 79
            + "CG"
            + "N" * 9
            + "CG"
            + "N" * 40
        )
        with open(fasta_path, "w") as f:
            f.write(">chr1\n" + seq + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_bismark",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    bam_path = tmp_path / "bismark.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": seq_len, "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        for r in reads:
            a = pysam.AlignedSegment()
            a.query_name = r["name"]
            a.query_sequence = r.get("seq", "N" * seq_len)
            a.flag = r["flag"]
            a.reference_id = 0
            a.reference_start = r.get("start", 0)
            a.mapping_quality = r["mapq"]
            a.cigartuples = [(0, seq_len)]
            a.set_tag("XM", r["xm_string"])
            for tag, val in r.get("extra_tags", []):
                a.set_tag(tag, val)
            out_bam.write(a)

    pysam.index(str(bam_path))
    return emb, str(bam_path)


# 0-based query positions of the C in each CpG for the test FASTA.
# The FASTA is: N*9 + CG + N*9 + CG + N*79 + CG + N*9 + CG + N*40 (154bp)
_BISMARK_CPG_QPOS = [9, 20, 101, 112]


def _make_xm_string(length, cpg_calls):
    """Build an XM string of given length with CpG calls at specific 0-based positions.

    cpg_calls is a dict of {0-based position: 'Z' or 'z'}.
    All other positions are '.'.
    """
    chars = ["."] * length
    for pos, call in cpg_calls.items():
        chars[pos] = call
    return "".join(chars)


def test_bismark_methylated_cpgs(tmp_path):
    """Bismark XM tag: Z at CpG positions → methylated (1)."""
    xm = _make_xm_string(154, {p: "Z" for p in _BISMARK_CPG_QPOS})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "read1", "flag": 0, "mapq": 60, "xm_string": xm}],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 1
    # All 4 CpG sites should be methylated (value=1)
    assert matrix.nnz == 4
    assert list(matrix.data) == [1, 1, 1, 1]


def test_bismark_unmethylated_cpgs(tmp_path):
    """Bismark XM tag: z at CpG positions → unmethylated (0)."""
    xm = _make_xm_string(154, {p: "z" for p in _BISMARK_CPG_QPOS})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "read1", "flag": 0, "mapq": 60, "xm_string": xm}],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 1
    # Unmethylated is stored as 0 in COO — but COO doesn't store literal zeros by default.
    # scipy COO *does* store explicit zeros; they are in .data.
    assert matrix.nnz == 4
    assert list(matrix.data) == [0, 0, 0, 0]


def test_bismark_mixed_methylation(tmp_path):
    """Bismark XM tag: mix of Z and z at CpG positions."""
    xm = _make_xm_string(154, {9: "Z", 20: "z", 101: "Z", 112: "z"})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "read1", "flag": 0, "mapq": 60, "xm_string": xm}],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 1
    assert matrix.nnz == 4
    assert list(matrix.data) == [1, 0, 1, 0]


def test_bismark_no_cpg_in_xm(tmp_path):
    """Bismark read with no CpG calls (all dots) → -1 for each CpG site."""
    xm = "." * 154  # No CpG calls at all
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "read1", "flag": 0, "mapq": 60, "xm_string": xm}],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    # Dots at CpG positions → -1 values, but the read should still be counted
    # if any CpG data was recorded
    # Actually: '.' is non-CpG context → gets -1 at CpG reference positions
    assert matrix.shape[0] == 1
    assert all(v == -1 for v in matrix.data)


def test_bismark_non_cpg_context_at_cpg_site(tmp_path):
    """Bismark XM tag with non-CpG context chars (H, h, X, x) at CpG positions → -1."""
    # Put CHH/CHG context markers at CpG reference positions
    xm = _make_xm_string(154, {9: "H", 20: "h", 101: "X", 112: "x"})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "read1", "flag": 0, "mapq": 60, "xm_string": xm}],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 1
    assert all(v == -1 for v in matrix.data)


def test_bismark_duplicate_flag_skipped(tmp_path):
    """Bismark reads with duplicate flag are still filtered out."""
    xm = _make_xm_string(154, {9: "Z", 20: "Z"})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "dup", "flag": 0x400, "mapq": 60, "xm_string": xm}],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 0


def test_bismark_low_mapq_skipped(tmp_path):
    """Bismark reads with low MAPQ are still filtered out."""
    xm = _make_xm_string(154, {9: "Z", 20: "Z"})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "lowq", "flag": 0, "mapq": 5, "xm_string": xm}],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 0


def test_bismark_both_strands_processed(tmp_path):
    """Bismark processes both forward and reverse reads (no strand filtering)."""
    xm = _make_xm_string(154, {9: "Z", 20: "z"})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [
            # Forward read
            {"name": "fwd", "flag": 0, "mapq": 60, "xm_string": xm},
            # Reverse read — Bismark should still process it
            {"name": "rev", "flag": 0x10, "mapq": 60, "xm_string": xm},
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    # Both reads should be processed (no daughter-strand filtering for Bismark)
    assert matrix.shape[0] == 2


def test_bismark_xm_takes_priority_over_yd(tmp_path):
    """If a read has both XM and YD tags, the Bismark (XM) path is used."""
    # XM says methylated at all CpG positions
    xm = _make_xm_string(154, {p: "Z" for p in _BISMARK_CPG_QPOS})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [
            {
                "name": "both_tags",
                "flag": 0,
                "mapq": 60,
                "xm_string": xm,
                # YD=r + flag=0 (forward) → would be daughter strand → skipped by YD path
                "extra_tags": [("YD", "r")],
            },
        ],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    # XM path should process it (not skip as daughter strand)
    assert matrix.shape[0] == 1
    assert all(v == 1 for v in matrix.data)


def test_bismark_debug_mode(tmp_path):
    """Bismark path works correctly with debug=True."""
    xm = _make_xm_string(154, {p: "Z" for p in _BISMARK_CPG_QPOS})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "debug_read", "flag": 0, "mapq": 60, "xm_string": xm}],
    )
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb, debug=True
    )
    assert matrix.shape[0] == 1
    assert list(matrix.data) == [1, 1, 1, 1]


def test_bismark_read_no_cpg_overlap(tmp_path):
    """Bismark read that doesn't overlap any CpG site → 0 rows."""
    fasta_path = tmp_path / "ref_nocpg.fa"
    # No CpG sites in first 50bp
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + "A" * 200 + "CG" + "A" * 48 + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_bismark_nocpg",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    xm = "." * 50
    bam_path = tmp_path / "nocpg.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": 250, "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "A" * 50
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0  # Covers 0-49 (0-based), no CpG there
        a.mapping_quality = 60
        a.cigartuples = [(0, 50)]
        a.set_tag("XM", xm)
        out_bam.write(a)

    pysam.index(str(bam_path))

    matrix = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path), genome_methylation_embedding=emb
    )
    assert matrix.shape[0] == 0


def test_detect_aligner_yd_tag():
    """detect_aligner identifies Biscuit/bwameth from the test BAM (has YD tag)."""
    result = detect_aligner("tests/test.bam")
    assert "Biscuit" in result or "bwameth" in result
    assert "YD" in result


def test_detect_aligner_xm_tag(tmp_path):
    """detect_aligner identifies Bismark from a BAM with XM tags."""
    bam_path = tmp_path / "bismark.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": 100, "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "A" * 100
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, 100)]
        a.set_tag("XM", "." * 100)
        out_bam.write(a)
    pysam.index(str(bam_path))

    result = detect_aligner(str(bam_path))
    assert "Bismark" in result
    assert "XM" in result


def test_detect_aligner_no_tags(tmp_path):
    """detect_aligner returns Unknown for BAMs without methylation tags."""
    bam_path = tmp_path / "plain.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": 100, "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "A" * 100
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, 100)]
        out_bam.write(a)
    pysam.index(str(bam_path))

    result = detect_aligner(str(bam_path))
    assert "Unknown" in result


def test_extract_methylation_missing_index(tmp_path) -> None:
    """Test that missing BAM index file raises FileNotFoundError."""
    import shutil

    # Copy a real BAM file but not its index
    bam_file = tmp_path / "no_index.bam"
    shutil.copy("tests/test.bam", bam_file)
    # Don't copy the .bai file

    with pytest.raises(FileNotFoundError, match="Index missing for bam file"):
        functions.extract_methylation_data_from_bam(
            input_bam=str(bam_file),
            genome_methylation_embedding=TEST_EMBEDDING,
            quality_limit=20,
            verbose=False,
            debug=False,
        )
