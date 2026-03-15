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
