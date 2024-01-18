from bam2tensor import embedding

test_embedding = embedding.GenomeMethylationEmbedding(
    "test_genome",
    expected_chromosomes=["chr1", "chr2", "chr3", "chrX", "chrY"],
    fasta_source="tests/test_fasta.fa",
    window_size=150,
    skip_cache=False,
    verbose=False,
)


def test_genome_methylation_embedding_init() -> None:
    """Test GenomeMethylationEmbedding."""

    assert test_embedding.genome_name == "test_genome"
    assert test_embedding.total_cpg_sites == 0

    assert test_embedding.cpgs_per_chr_cumsum[-1] == test_embedding.total_cpg_sites


def test_embedding_to_genomic_position() -> None:
    """Test embedding_to_genomic_position."""

    assert test_embedding.embedding_to_genomic_position(0) == (
        "chr1",
        test_embedding.cpg_sites_dict["chr1"][0],
    )
    assert test_embedding.embedding_to_genomic_position(1) == (
        "chr1",
        test_embedding.cpg_sites_dict["chr1"][1],
    )

    # Edges
    assert test_embedding.embedding_to_genomic_position(
        test_embedding.cpgs_per_chr_cumsum[0]
    ) == ("chr2", test_embedding.cpg_sites_dict["chr2"][0])
    assert test_embedding.embedding_to_genomic_position(
        test_embedding.cpgs_per_chr_cumsum[-1] - 1
    ) == ("chrY", test_embedding.cpg_sites_dict["chrY"][-1])


def genomic_position_to_embedding() -> None:
    """Test genomic_position_to_embedding."""

    assert (
        test_embedding.genomic_position_to_embedding(
            "chr1", test_embedding.cpg_sites_dict["chr1"][0]
        )
        == 0
    )
    assert (
        test_embedding.genomic_position_to_embedding(
            "chr1", test_embedding.cpg_sites_dict["chr1"][1]
        )
        == 1
    )

    # Edges
    assert (
        test_embedding.genomic_position_to_embedding(
            "chr2", test_embedding.cpg_sites_dict["chr2"][0]
        )
        == test_embedding.cpgs_per_chr_cumsum[0]
    )
    assert (
        test_embedding.genomic_position_to_embedding(
            "chrY", test_embedding.cpg_sites_dict["chrY"][-1]
        )
        == test_embedding.cpgs_per_chr_cumsum[-1] - 1
    )
