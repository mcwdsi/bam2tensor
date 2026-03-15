import pysam
import pytest
from bam2tensor import embedding
from bam2tensor import functions
from bam2tensor.functions import check_chromosome_overlap

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
