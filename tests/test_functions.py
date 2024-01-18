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
