import pytest
from bam2tensor import embedding
from bam2tensor import functions


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
