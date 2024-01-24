from bam2tensor import embedding

# Generate a fresh, uncached embedding
test_embedding = embedding.GenomeMethylationEmbedding(
    "test_genome",
    expected_chromosomes=["chr1", "chr2", "chr3"],
    fasta_source="tests/test_fasta.fa",
    window_size=150,
    skip_cache=True,
    verbose=True,
)

# Generate a cached version of the embedding
test_embedding_cache_generate = embedding.GenomeMethylationEmbedding(
    "test_genome",
    expected_chromosomes=["chr1", "chr2", "chr3"],
    fasta_source="tests/test_fasta.fa",
    window_size=150,
    skip_cache=False,
    verbose=True,
)

# Load the cached version of the embedding
test_embedding_cache_load = embedding.GenomeMethylationEmbedding(
    "test_genome",
    expected_chromosomes=["chr1", "chr2", "chr3"],
    fasta_source="tests/test_fasta.fa",
    window_size=150,
    skip_cache=False,
    verbose=False,
)

TEST_EMBEDDINGS = [
    test_embedding,
    test_embedding_cache_generate,
    test_embedding_cache_load,
]


def test_genome_methylation_embedding_init() -> None:
    """Test GenomeMethylationEmbedding."""

    assert test_embedding.genome_name == "test_genome"
    assert test_embedding.total_cpg_sites == 37489  # Known from test_fasta.fa

    assert test_embedding.expected_chromosomes == ["chr1", "chr2", "chr3"]

    assert test_embedding.cpgs_per_chr_cumsum[-1] == test_embedding.total_cpg_sites


def test_cached_embedding() -> None:
    """Test cached embedding."""

    assert test_embedding.total_cpg_sites == 37489
    assert test_embedding_cache_generate.total_cpg_sites == 37489
    assert test_embedding_cache_load.total_cpg_sites == 37489


def test_embedding_to_genomic_position() -> None:
    """Test embedding_to_genomic_position."""

    for embedding_obj in TEST_EMBEDDINGS:
        assert embedding_obj.embedding_to_genomic_position(0) == (
            "chr1",
            embedding_obj.cpg_sites_dict["chr1"][0],
        )
        assert embedding_obj.embedding_to_genomic_position(1) == (
            "chr1",
            embedding_obj.cpg_sites_dict["chr1"][1],
        )

        # Edges
        assert embedding_obj.embedding_to_genomic_position(
            embedding_obj.cpgs_per_chr_cumsum[0]
        ) == ("chr2", embedding_obj.cpg_sites_dict["chr2"][0])
        assert embedding_obj.embedding_to_genomic_position(
            embedding_obj.cpgs_per_chr_cumsum[-1] - 1
        ) == ("chr3", embedding_obj.cpg_sites_dict["chr3"][-1])


def test_genomic_position_to_embedding() -> None:
    """Test genomic_position_to_embedding."""

    for embedding_obj in TEST_EMBEDDINGS:
        assert (
            embedding_obj.genomic_position_to_embedding(
                "chr1", embedding_obj.cpg_sites_dict["chr1"][0]
            )
            == 0
        )
        assert (
            embedding_obj.genomic_position_to_embedding(
                "chr1", embedding_obj.cpg_sites_dict["chr1"][1]
            )
            == 1
        )

        # Edges
        assert (
            embedding_obj.genomic_position_to_embedding(
                "chr2", embedding_obj.cpg_sites_dict["chr2"][0]
            )
            == embedding_obj.cpgs_per_chr_cumsum[0]
        )
        assert (
            embedding_obj.genomic_position_to_embedding(
                "chr3", embedding_obj.cpg_sites_dict["chr3"][-1]
            )
            == embedding_obj.cpgs_per_chr_cumsum[-1] - 1
        )
