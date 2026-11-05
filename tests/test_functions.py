import numpy as np
import pysam
import pytest
from bam2tensor import embedding
from bam2tensor import functions
from bam2tensor.functions import ExtractionResult, check_chromosome_overlap
from bam2tensor.functions import detect_aligner
from tests.paths import TEST_BAM, TEST_FASTA

TEST_EMBEDDING = embedding.GenomeMethylationEmbedding(
    "test_genome",
    expected_chromosomes=["chr1", "chr2", "chr3"],
    fasta_source=TEST_FASTA,
    skip_cache=True,  # built at import time: must not write a cache file
    verbose=True,
)


def test_extract_methylation_data_from_bam() -> None:
    """Test extract_methylation_data_from_bam."""

    result = functions.extract_methylation_data_from_bam(
        input_bam=TEST_BAM,
        genome_methylation_embedding=TEST_EMBEDDING,
        quality_limit=20,
        verbose=True,
        debug=False,
    )

    assert isinstance(result, ExtractionResult)
    # Need to make a better test bam & fa pair...
    assert result.matrix.shape == (1, 37489)
    assert len(result.tlen) == result.matrix.shape[0]
    assert result.tlen.dtype == np.int32


def test_extract_methylation_data_from_bam_debug() -> None:
    """Test extract_methylation_data_from_bam with debug flag set."""

    result = functions.extract_methylation_data_from_bam(
        input_bam=TEST_BAM,
        genome_methylation_embedding=TEST_EMBEDDING,
        quality_limit=20,
        verbose=True,
        debug=True,
    )

    assert result.matrix.shape == (1, 37489)
    assert len(result.tlen) == 1


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
        # CpGs with their C at 1-based positions 10, 21, 102, 113
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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0


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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0


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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0


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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0


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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0


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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1


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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1


# ======================================================================
# C: Bisulfite strand and alignment orientation
# ======================================================================

# 0-based reference positions of the C and the G of each CpG in the
# _make_bam_with_reads reference (N*9 CG N*9 CG N*79 CG N*9 CG N*40).
_CPG_C_POSITIONS = [9, 20, 101, 112]
_CPG_G_POSITIONS = [10, 21, 102, 113]


def _read_with_bases(positions, bases, seq_len=150):
    """A read sequence of N with the given bases at the given 0-based positions."""
    seq = ["N"] * seq_len
    for position, base in zip(positions, bases):
        seq[position] = base
    return "".join(seq)


@pytest.mark.parametrize("flag", [0, 0x10], ids=["forward", "reverse"])
@pytest.mark.parametrize("tag", [("YD", "f"), ("XB", "C")], ids=["YD=f", "XB=C"])
def test_top_strand_fragment_is_read_at_the_c(tmp_path, flag, tag):
    """A top-strand fragment is decoded at the C of each CpG, in either orientation.

    BAM stores reads in top-strand orientation, so the forward mate (read 1)
    and the reverse mate (read 2) of a top-strand pair both show C
    (methylated) or T (unmethylated) at the C.
    """
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "top",
                "flag": flag,
                "mapq": 60,
                "seq": _read_with_bases(_CPG_C_POSITIONS, "CTCT"),
                "tags": [("MD", "150"), tag],
            },
        ],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1
    assert result.matrix.col.tolist() == [0, 1, 2, 3]
    assert result.matrix.data.tolist() == [1, 0, 1, 0]


@pytest.mark.parametrize("flag", [0, 0x10], ids=["forward", "reverse"])
@pytest.mark.parametrize("tag", [("YD", "r"), ("XB", "G")], ids=["YD=r", "XB=G"])
def test_bottom_strand_fragment_is_read_at_the_g(tmp_path, flag, tag):
    """A bottom-strand fragment is decoded at the G of each CpG, in either orientation.

    Regression: a forward read tagged as bottom strand (read 2 of a
    bottom-strand pair) used to be discarded as an uninformative "daughter
    strand", as was a reverse read tagged as top strand. Both mates of a pair
    carry the same strand tag and the same information, so half of every
    paired-end library was thrown away.
    """
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "bottom",
                "flag": flag,
                "mapq": 60,
                # C at every CpG's C position (it is not informative here),
                # G / A at the G positions.
                "seq": _read_with_bases(
                    _CPG_C_POSITIONS + _CPG_G_POSITIONS, "CCCC" + "GAGA"
                ),
                "tags": [("MD", "150"), tag],
            },
        ],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1
    assert result.matrix.col.tolist() == [0, 1, 2, 3]
    assert result.matrix.data.tolist() == [1, 0, 1, 0]


def test_strand_tag_decides_which_base_is_read(tmp_path):
    """The same read decodes differently under the two strand tags.

    With C at every C and A at every G: methylated everywhere if the fragment
    is top strand, unmethylated everywhere if it is bottom strand.
    """
    seq = _read_with_bases(_CPG_C_POSITIONS + _CPG_G_POSITIONS, "CCCC" + "AAAA")
    for tag, expected in ((("YD", "f"), [1, 1, 1, 1]), (("YD", "r"), [0, 0, 0, 0])):
        directory = tmp_path / tag[1]
        directory.mkdir()
        emb, bam_path = _make_bam_with_reads(
            directory,
            [{"name": "read", "flag": 0, "mapq": 60, "seq": seq, "tags": [tag]}],
        )
        result = functions.extract_methylation_data_from_bam(
            input_bam=bam_path, genome_methylation_embedding=emb
        )
        assert result.matrix.data.tolist() == expected


def test_no_strand_tag_skipped(tmp_path):
    """A read with no strand tag (no XM, YD or XB) cannot be decoded and is skipped."""
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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0


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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0


def test_mixed_reads_correct_count(tmp_path):
    """Mix of usable and filtered reads yields the correct number of rows."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            # Good: forward read of a top-strand fragment
            {
                "name": "good1",
                "flag": 0,
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
            # Good: reverse read of a bottom-strand fragment
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
            # Good: forward read of a bottom-strand fragment (its read 2)
            {
                "name": "good3",
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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 3  # good1, good2 and good3


def test_skip_flags_constant():
    """Verify _SKIP_FLAGS matches the expected BAM flag bits."""
    from bam2tensor.functions import _SKIP_FLAGS

    # duplicate | qcfail | secondary | supplementary | unmapped
    assert _SKIP_FLAGS == 0x400 | 0x200 | 0x100 | 0x800 | 0x4
    assert _SKIP_FLAGS == 0xF04


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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1
    # All 4 CpG sites should be methylated (value=1)
    assert result.matrix.nnz == 4
    assert list(result.matrix.data) == [1, 1, 1, 1]


def test_bismark_unmethylated_cpgs(tmp_path):
    """Bismark XM tag: z at CpG positions → unmethylated (0)."""
    xm = _make_xm_string(154, {p: "z" for p in _BISMARK_CPG_QPOS})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "read1", "flag": 0, "mapq": 60, "xm_string": xm}],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1
    # Unmethylated is stored as 0 in COO — but COO doesn't store literal zeros by default.
    # scipy COO *does* store explicit zeros; they are in .data.
    assert result.matrix.nnz == 4
    assert list(result.matrix.data) == [0, 0, 0, 0]


def test_bismark_mixed_methylation(tmp_path):
    """Bismark XM tag: mix of Z and z at CpG positions."""
    xm = _make_xm_string(154, {9: "Z", 20: "z", 101: "Z", 112: "z"})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "read1", "flag": 0, "mapq": 60, "xm_string": xm}],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1
    assert result.matrix.nnz == 4
    assert list(result.matrix.data) == [1, 0, 1, 0]


def test_bismark_no_cpg_in_xm(tmp_path):
    """Bismark read with no CpG calls (all dots) → -1 for each CpG site."""
    xm = "." * 154  # No CpG calls at all
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "read1", "flag": 0, "mapq": 60, "xm_string": xm}],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    # Dots at CpG positions → -1 values, but the read should still be counted
    # if any CpG data was recorded
    # Actually: '.' is non-CpG context → gets -1 at CpG reference positions
    assert result.matrix.shape[0] == 1
    assert all(v == -1 for v in result.matrix.data)


def test_bismark_non_cpg_context_at_cpg_site(tmp_path):
    """Bismark XM tag with non-CpG context chars (H, h, X, x) at CpG positions → -1."""
    # Put CHH/CHG context markers at CpG reference positions
    xm = _make_xm_string(154, {9: "H", 20: "h", 101: "X", 112: "x"})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "read1", "flag": 0, "mapq": 60, "xm_string": xm}],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1
    assert all(v == -1 for v in result.matrix.data)


def test_bismark_duplicate_flag_skipped(tmp_path):
    """Bismark reads with duplicate flag are still filtered out."""
    xm = _make_xm_string(154, {9: "Z", 20: "Z"})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "dup", "flag": 0x400, "mapq": 60, "xm_string": xm}],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0


def test_bismark_low_mapq_skipped(tmp_path):
    """Bismark reads with low MAPQ are still filtered out."""
    xm = _make_xm_string(154, {9: "Z", 20: "Z"})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "lowq", "flag": 0, "mapq": 5, "xm_string": xm}],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0


def test_bismark_both_strands_processed(tmp_path):
    """Bismark reads are used whether they align forward or reverse."""
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
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    # Two unpaired reads with different names: two fragments.
    assert result.matrix.shape[0] == 2


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
                # YD=r would mean "bottom strand, read the G" on the YD path
                "extra_tags": [("YD", "r")],
            },
        ],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    # The XM calls (at the C positions) are used, not the bases under YD.
    assert result.matrix.shape[0] == 1
    assert all(v == 1 for v in result.matrix.data)


def test_bismark_debug_mode(tmp_path):
    """Bismark path works correctly with debug=True."""
    xm = _make_xm_string(154, {p: "Z" for p in _BISMARK_CPG_QPOS})
    emb, bam_path = _make_bismark_bam(
        tmp_path,
        [{"name": "debug_read", "flag": 0, "mapq": 60, "xm_string": xm}],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb, debug=True
    )
    assert result.matrix.shape[0] == 1
    assert list(result.matrix.data) == [1, 1, 1, 1]


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

    result = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path), genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0


def test_detect_aligner_yd_tag():
    """detect_aligner identifies Biscuit/bwameth from the test BAM (has YD tag)."""
    result = detect_aligner(TEST_BAM)
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
    shutil.copy(TEST_BAM, bam_file)
    # Don't copy the .bai file

    with pytest.raises(FileNotFoundError, match="Index missing for bam file"):
        functions.extract_methylation_data_from_bam(
            input_bam=str(bam_file),
            genome_methylation_embedding=TEST_EMBEDDING,
            quality_limit=20,
            verbose=False,
            debug=False,
        )


# ======================================================================
# detect_aligner edge case tests
# ======================================================================


def test_detect_aligner_xb_tag(tmp_path):
    """detect_aligner identifies gem3/Blueprint from XB tag."""
    bam_path = tmp_path / "gem3.bam"
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
        a.set_tag("XB", "C")
        out_bam.write(a)
    pysam.index(str(bam_path))

    result = detect_aligner(str(bam_path))
    assert "gem3" in result or "Blueprint" in result
    assert "XB" in result


def test_detect_aligner_cannot_open():
    """detect_aligner returns Unknown when BAM cannot be opened."""
    result = detect_aligner("/nonexistent/path/to/file.bam")
    assert "Unknown" in result
    assert "could not open" in result


def test_detect_aligner_sample_size_limit(tmp_path):
    """detect_aligner stops after sample_size reads."""
    bam_path = tmp_path / "many_reads.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": 100, "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        for i in range(10):
            a = pysam.AlignedSegment()
            a.query_name = f"read{i}"
            a.query_sequence = "A" * 100
            a.flag = 0
            a.reference_id = 0
            a.reference_start = 0
            a.mapping_quality = 60
            a.cigartuples = [(0, 100)]
            out_bam.write(a)
    pysam.index(str(bam_path))

    result = detect_aligner(str(bam_path), sample_size=3)
    assert "Unknown" in result
    assert "no XM/YD/XB" in result


def test_detect_aligner_skips_flagged_reads(tmp_path):
    """detect_aligner skips reads with skip flags before checking tags."""
    bam_path = tmp_path / "flagged.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": 100, "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        # Duplicate read with XM tag — should be skipped
        a = pysam.AlignedSegment()
        a.query_name = "dup_read"
        a.query_sequence = "A" * 100
        a.flag = 0x400  # duplicate
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, 100)]
        a.set_tag("XM", "." * 100)
        out_bam.write(a)
        # Good read with YD tag
        a2 = pysam.AlignedSegment()
        a2.query_name = "good_read"
        a2.query_sequence = "A" * 100
        a2.flag = 0
        a2.reference_id = 0
        a2.reference_start = 0
        a2.mapping_quality = 60
        a2.cigartuples = [(0, 100)]
        a2.set_tag("YD", "f")
        out_bam.write(a2)
    pysam.index(str(bam_path))

    result = detect_aligner(str(bam_path))
    assert "Biscuit" in result or "bwameth" in result


# ======================================================================
# Chromosome-not-in-BAM tests
# ======================================================================


def test_chromosome_not_in_bam_skipped(tmp_path):
    """Chromosomes in embedding but not in BAM are skipped gracefully."""
    fasta_path = tmp_path / "ref.fa"
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + "N" * 9 + "CG" + "N" * 139 + "\n")
        f.write(">chr2\n" + "N" * 9 + "CG" + "N" * 139 + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_chrom_skip",
        expected_chromosomes=["chr1", "chr2"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    # BAM only has chr1, not chr2
    bam_path = tmp_path / "test.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": 150, "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "N" * 150
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, 150)]
        a.set_tag("MD", "150")
        a.set_tag("YD", "f")
        out_bam.write(a)
    pysam.index(str(bam_path))

    result = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path),
        genome_methylation_embedding=emb,
        verbose=True,
    )
    # Should still process chr1 reads
    assert result.matrix.shape[0] == 1


# ======================================================================
# Biscuit/bwameth path: C and T base coverage
# ======================================================================


def test_biscuit_methylated_and_unmethylated_bases(tmp_path):
    """Biscuit path: C at CpG = methylated (1), T at CpG = unmethylated (0)."""
    fasta_path = tmp_path / "ref.fa"
    # CpGs at 1-based positions 10, 21 (0-based C positions 9, 20)
    seq = "N" * 9 + "CG" + "N" * 9 + "CG" + "N" * 128
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + seq + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_biscuit_ct",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    bam_path = tmp_path / "test.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": len(seq), "SN": "chr1"}]}
    # Build a read with C at position 9 (methylated) and T at position 20 (unmethylated)
    read_seq = list("N" * len(seq))
    read_seq[9] = "C"  # methylated at CpG pos 10
    read_seq[20] = "T"  # unmethylated at CpG pos 21
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "".join(read_seq)
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, len(seq))]
        a.set_tag("MD", str(len(seq)))
        a.set_tag("YD", "f")
        out_bam.write(a)
    pysam.index(str(bam_path))

    result = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path),
        genome_methylation_embedding=emb,
    )
    assert result.matrix.shape[0] == 1
    assert result.matrix.nnz == 2
    data = list(result.matrix.data)
    assert 1 in data  # methylated
    assert 0 in data  # unmethylated


def test_biscuit_debug_mode_ct_bases(tmp_path):
    """Biscuit debug path prints methylation status for C and T bases."""
    fasta_path = tmp_path / "ref.fa"
    seq = "N" * 9 + "CG" + "N" * 9 + "CG" + "N" * 128
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + seq + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_biscuit_debug_ct",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    bam_path = tmp_path / "test.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": len(seq), "SN": "chr1"}]}
    read_seq = list("N" * len(seq))
    read_seq[9] = "C"
    read_seq[20] = "T"
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "".join(read_seq)
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, len(seq))]
        a.set_tag("MD", str(len(seq)))
        a.set_tag("YD", "f")
        out_bam.write(a)
    pysam.index(str(bam_path))

    result = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path),
        genome_methylation_embedding=emb,
        debug=True,
    )
    assert result.matrix.shape[0] == 1


def test_biscuit_ob_strand_methylation_extraction(tmp_path):
    """Biscuit/bwameth OB-strand (YD=r, is_reverse=True) reads must read the
    methylation-informative base at ref_pos+1 (G=methylated, A=unmethylated),
    not ref_pos (which is always C in BAM SEQ regardless of methylation state).

    Regression for the bug where OB reads were extracted with C/T logic at
    ref_pos and thus scored as universally methylated.
    """
    fasta_path = tmp_path / "ref.fa"
    # CpGs at 1-based positions 10, 21 (top-strand C at 0-based 9, 20; G at 10, 21).
    seq = "N" * 9 + "CG" + "N" * 9 + "CG" + "N" * 128
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + seq + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_biscuit_ob",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    bam_path = tmp_path / "test.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": len(seq), "SN": "chr1"}]}
    # OB read: BAM SEQ is reference-oriented. The C of each top-strand CG is
    # always C in BAM (bottom-strand G reverse-complemented). The G of each
    # top-strand CG is what carries the methylation signal: G=methylated,
    # A=unmethylated.
    read_seq = list("N" * len(seq))
    read_seq[9] = "C"  # top-strand C of CpG#1 (always C in BAM for OB)
    read_seq[10] = "G"  # methylated → G at ref_pos+1
    read_seq[20] = "C"  # top-strand C of CpG#2 (always C in BAM for OB)
    read_seq[21] = "A"  # unmethylated → A at ref_pos+1
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "ob_read"
        a.query_sequence = "".join(read_seq)
        a.flag = 0x10  # reverse-mapped
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, len(seq))]
        a.set_tag("MD", str(len(seq)))
        a.set_tag("YD", "r")  # OB / reverse parent strand
        out_bam.write(a)
    pysam.index(str(bam_path))

    result = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path),
        genome_methylation_embedding=emb,
    )
    assert result.matrix.shape[0] == 1
    assert result.matrix.nnz == 2
    data = sorted(result.matrix.data)
    assert data == [0, 1], (
        f"Expected one methylated (1) and one unmethylated (0) call, got {data}. "
        "If this is all 1s, the OB-strand base lookup regression has returned."
    )


def test_biscuit_ot_and_ob_share_cpg_columns(tmp_path):
    """OT and OB reads at the same CpG must land in the same embedding column
    (canonical CpG site = top-strand C, ref_pos+1 in 1-based coordinates).
    """
    fasta_path = tmp_path / "ref.fa"
    seq = "N" * 9 + "CG" + "N" * 9 + "CG" + "N" * 128
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + seq + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_biscuit_ot_ob_columns",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    bam_path = tmp_path / "test.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": len(seq), "SN": "chr1"}]}
    # OT read: C at top-strand C positions = methylated at both CpGs.
    ot_seq = list("N" * len(seq))
    ot_seq[9] = "C"
    ot_seq[20] = "C"
    # OB read (BAM in reference orientation): G at top-strand G positions
    # = methylated at both CpGs.
    ob_seq = list("N" * len(seq))
    ob_seq[9] = "C"
    ob_seq[10] = "G"
    ob_seq[20] = "C"
    ob_seq[21] = "G"
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "ot_read"
        a.query_sequence = "".join(ot_seq)
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, len(seq))]
        a.set_tag("MD", str(len(seq)))
        a.set_tag("YD", "f")
        out_bam.write(a)
        b = pysam.AlignedSegment()
        b.query_name = "ob_read"
        b.query_sequence = "".join(ob_seq)
        b.flag = 0x10
        b.reference_id = 0
        b.reference_start = 0
        b.mapping_quality = 60
        b.cigartuples = [(0, len(seq))]
        b.set_tag("MD", str(len(seq)))
        b.set_tag("YD", "r")
        out_bam.write(b)
    pysam.index(str(bam_path))

    result = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path),
        genome_methylation_embedding=emb,
    )
    assert result.matrix.shape[0] == 2
    # Both reads call both CpGs methylated, so we expect two reads × two CpGs
    # in the same two columns, all with value 1.
    coo = result.matrix.tocoo()
    ot_cols = sorted(int(c) for r, c in zip(coo.row, coo.col) if r == 0)
    ob_cols = sorted(int(c) for r, c in zip(coo.row, coo.col) if r == 1)
    assert ot_cols == ob_cols, f"OT and OB columns diverged: OT={ot_cols} OB={ob_cols}"
    assert list(result.matrix.data) == [1, 1, 1, 1]


# ======================================================================
# XB tag (gem3/Blueprint) extraction tests
# ======================================================================


def test_xb_tag_forward_strand_extraction(tmp_path):
    """gem3 XB=C tag on forward read processes correctly."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "xb_forward",
                "flag": 0,
                "mapq": 60,
                "tags": [("MD", "150"), ("XB", "C")],
            },
        ],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1


def test_xb_tag_reverse_strand_extraction(tmp_path):
    """gem3 XB=G tag on reverse read processes correctly."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "xb_reverse",
                "flag": 0x10,  # reverse
                "mapq": 60,
                "tags": [("MD", "150"), ("XB", "G")],
            },
        ],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1


# ======================================================================
# Bismark edge cases for get_aligned_pairs filtering
# ======================================================================


def test_bismark_cpgs_in_range_but_no_aligned_pairs(tmp_path):
    """Bismark read overlaps CpG range but get_aligned_pairs has no CpG matches."""
    fasta_path = tmp_path / "ref.fa"
    # CpG at position 51 (1-based), read covers positions 0-49
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + "A" * 50 + "CG" + "A" * 48 + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_bismark_no_pairs",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    # Read covers 0-99, CpG at pos 51 is within range.
    # Use an insertion in CIGAR so aligned pairs skip the CpG position.
    bam_path = tmp_path / "no_pairs.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": 100, "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "A" * 100
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        # 50M match + 50I insertion means reference_end=50, covering 0-49
        # CpG at ref pos 50 (0-based) is NOT covered
        a.cigartuples = [(0, 50), (1, 50)]
        a.set_tag("XM", "." * 100)
        out_bam.write(a)
    pysam.index(str(bam_path))

    result = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path), genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0


# ======================================================================
# TLEN (template length) tests
# ======================================================================


def test_tlen_shape_matches_reads(tmp_path):
    """TLEN array length matches the number of rows in the matrix."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "read1",
                "flag": 0,
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert len(result.tlen) == result.matrix.shape[0]


def test_tlen_dtype_is_int32(tmp_path):
    """TLEN array dtype is int32."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "read1",
                "flag": 0,
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.tlen.dtype == np.int32


def test_tlen_with_template_length(tmp_path):
    """TLEN captures the template_length set on the aligned segment."""
    fasta_path = tmp_path / "ref.fa"
    seq = "N" * 9 + "CG" + "N" * 9 + "CG" + "N" * 79 + "CG" + "N" * 9 + "CG" + "N" * 40
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + seq + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_tlen",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    bam_path = tmp_path / "test.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": 150, "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "N" * 150
        a.flag = 0x1 | 0x2  # paired, proper pair
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, 150)]
        a.template_length = 300
        a.set_tag("MD", "150")
        a.set_tag("YD", "f")
        out_bam.write(a)

    pysam.index(str(bam_path))

    result = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path), genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1
    assert result.tlen[0] == 300


def test_tlen_zero_for_single_end(tmp_path):
    """TLEN is 0 for single-end reads (default template_length)."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "se_read",
                "flag": 0,  # single-end
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1
    assert result.tlen[0] == 0


def test_tlen_excluded_for_filtered_reads(tmp_path):
    """Filtered reads (dup, qcfail, etc.) are not included in TLEN array."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "good_read",
                "flag": 0,
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
            {
                "name": "dup_read",
                "flag": 0x400,  # duplicate — should be filtered
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 1
    assert len(result.tlen) == 1


def test_tlen_empty_for_no_reads(tmp_path):
    """TLEN array is empty when no reads pass filters."""
    emb, bam_path = _make_bam_with_reads(
        tmp_path,
        [
            {
                "name": "dup_read",
                "flag": 0x400,
                "mapq": 60,
                "tags": [("MD", "150"), ("YD", "f")],
            },
        ],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb
    )
    assert result.matrix.shape[0] == 0
    assert len(result.tlen) == 0
    assert result.tlen.dtype == np.int32


# ======================================================================
# Reads without a usable alignment
# ======================================================================


def _make_bam_from_segments(tmp_path, build_segments, seq_len=150):
    """Helper: write an indexed single-contig BAM from hand-built segments.

    ``build_segments`` receives the BAM header and returns the
    ``pysam.AlignedSegment`` objects to write, in coordinate order. Uses the
    same reference as ``_make_bam_with_reads`` (CpG cytosines at 1-based 10,
    21, 102, 113).
    """
    fasta_path = tmp_path / "ref.fa"
    seq = "N" * 9 + "CG" + "N" * 9 + "CG" + "N" * 79 + "CG" + "N" * 9 + "CG" + "N" * 40
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + seq + "\n")
    emb = embedding.GenomeMethylationEmbedding(
        "test_segments",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    bam_path = tmp_path / "segments.bam"
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.0", "SO": "coordinate"}, "SQ": [{"LN": seq_len, "SN": "chr1"}]}
    )
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        for segment in build_segments(header):
            out_bam.write(segment)
    pysam.index(str(bam_path))
    return emb, str(bam_path)


def _mapped_segment(header, name, flag=0, mapq=60, seq_len=150):
    """A full-length forward-parent (YD=f) read covering all four CpGs."""
    a = pysam.AlignedSegment(header)
    a.query_name = name
    a.query_sequence = "N" * seq_len
    a.flag = flag
    a.reference_id = 0
    a.reference_start = 0
    a.mapping_quality = mapq
    a.cigartuples = [(0, seq_len)]
    a.set_tag("MD", str(seq_len))
    a.set_tag("YD", "f")
    return a


def _unmapped_placed_mate(header, name, strand_tag=("YD", "f")):
    """An unmapped read placed at its mapped mate's position.

    This is how aligners emit a pair in which only one mate aligned: the
    unmapped mate inherits the mate's reference and position (so a region
    fetch returns it) but has flag 0x4, MAPQ 0 and no CIGAR.
    """
    a = pysam.AlignedSegment(header)
    a.query_name = name
    a.query_sequence = "N" * 50
    a.flag = 0x1 | 0x4 | 0x40  # paired, unmapped, first in pair
    a.reference_id = 0
    a.reference_start = 0
    a.mapping_quality = 0
    a.next_reference_id = 0
    a.next_reference_start = 0
    if strand_tag is not None:
        a.set_tag(*strand_tag)
    return a


@pytest.mark.parametrize(
    "strand_tag",
    [("YD", "f"), ("XB", "C"), ("XM", "." * 50), None],
    ids=["biscuit-YD", "gem3-XB", "bismark-XM", "untagged"],
)
def test_unmapped_placed_mate_is_skipped(tmp_path, strand_tag):
    """An unmapped mate placed at its partner's position is ignored, not fatal.

    Regression: with ``quality_limit=0`` the MAPQ filter no longer hid these
    reads, and their missing reference end crashed the CpG lookup with
    ``TypeError: '<' not supported between instances of 'NoneType' and 'int'``.
    """
    emb, bam_path = _make_bam_from_segments(
        tmp_path,
        lambda header: [
            _unmapped_placed_mate(header, "pair1", strand_tag),
            _mapped_segment(header, "pair1", flag=0x1 | 0x8 | 0x80),
        ],
    )

    for quality_limit in (0, 20):
        result = functions.extract_methylation_data_from_bam(
            input_bam=bam_path,
            genome_methylation_embedding=emb,
            quality_limit=quality_limit,
        )
        # Only the mapped mate contributes a row.
        assert result.matrix.shape[0] == 1
        assert len(result.tlen) == 1


def test_unmapped_placed_mate_is_skipped_in_parallel_path(tmp_path):
    """The unmapped-read skip applies in worker processes too."""
    emb, bam_path = _make_bam_from_segments(
        tmp_path,
        lambda header: [
            _unmapped_placed_mate(header, "pair1"),
            _mapped_segment(header, "pair1", flag=0x1 | 0x8 | 0x80),
        ],
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        quality_limit=0,
        threads=2,
    )
    assert result.matrix.shape[0] == 1


def test_mapped_read_without_cigar_is_skipped(tmp_path):
    """A read flagged as mapped but with no CIGAR has no span to extract."""

    def build(header):
        no_cigar = pysam.AlignedSegment(header)
        no_cigar.query_name = "no_cigar"
        no_cigar.query_sequence = "N" * 50
        no_cigar.flag = 0
        no_cigar.reference_id = 0
        no_cigar.reference_start = 0
        no_cigar.mapping_quality = 60
        no_cigar.set_tag("YD", "f")
        return [no_cigar, _mapped_segment(header, "good")]

    emb, bam_path = _make_bam_from_segments(tmp_path, build)
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb, quality_limit=0
    )
    assert result.matrix.shape[0] == 1


def test_detect_aligner_ignores_unmapped_reads(tmp_path):
    """Aligner detection samples only mapped reads."""
    _, bam_path = _make_bam_from_segments(
        tmp_path,
        # The unmapped read carries a Bismark-style tag; the mapped one YD.
        lambda header: [
            _unmapped_placed_mate(header, "pair1", ("XM", "." * 50)),
            _mapped_segment(header, "pair1", flag=0x1 | 0x8 | 0x80),
        ],
    )
    assert detect_aligner(bam_path) == "Biscuit/bwameth (YD tag)"
