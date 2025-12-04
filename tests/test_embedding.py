import os
import pytest
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


def test_empty_chromosomes_raises_error() -> None:
    """Test that empty expected_chromosomes raises ValueError."""
    with pytest.raises(ValueError, match="Expected chromosomes cannot be empty"):
        embedding.GenomeMethylationEmbedding(
            "test_genome",
            expected_chromosomes=[],
            fasta_source="tests/test_fasta.fa",
            window_size=150,
            skip_cache=True,
            verbose=False,
        )


def test_missing_fasta_raises_error() -> None:
    """Test that missing fasta file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Cannot read fasta file"):
        embedding.GenomeMethylationEmbedding(
            "test_genome",
            expected_chromosomes=["chr1"],
            fasta_source="/nonexistent/file.fa",
            window_size=150,
            skip_cache=True,
            verbose=False,
        )


def test_cache_chromosome_mismatch(tmp_path) -> None:
    """Test that mismatched chromosomes between cache and request raises AssertionError."""
    # First create a cache with chr1,chr2,chr3
    cache_file = tmp_path / "mismatch_test.cache.json.gz"

    # Create embedding with original chromosomes (creates cache)
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        embedding.GenomeMethylationEmbedding(
            "mismatch_test",
            expected_chromosomes=["chr1", "chr2", "chr3"],
            fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
            window_size=150,
            skip_cache=False,
            verbose=False,
        )
        assert cache_file.exists()

        # Now try to load with different chromosomes - should raise AssertionError
        with pytest.raises(AssertionError, match="Expected chromosomes do not match"):
            embedding.GenomeMethylationEmbedding(
                "mismatch_test",
                expected_chromosomes=["chr1"],  # Different!
                fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
                window_size=150,
                skip_cache=False,
                verbose=False,
            )
    finally:
        os.chdir(original_cwd)


def test_cache_window_size_mismatch(tmp_path) -> None:
    """Test that mismatched window_size between cache and request raises AssertionError."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Create embedding with window_size=150 (creates cache)
        embedding.GenomeMethylationEmbedding(
            "window_test",
            expected_chromosomes=["chr1", "chr2", "chr3"],
            fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
            window_size=150,
            skip_cache=False,
            verbose=False,
        )

        # Now try to load with different window_size - should raise AssertionError
        with pytest.raises(AssertionError, match="Window size does not match"):
            embedding.GenomeMethylationEmbedding(
                "window_test",
                expected_chromosomes=["chr1", "chr2", "chr3"],
                fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
                window_size=200,  # Different!
                skip_cache=False,
                verbose=False,
            )
    finally:
        os.chdir(original_cwd)


def test_cache_not_found_verbose(tmp_path, capsys) -> None:
    """Test verbose message when cache file not found."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Create embedding with verbose=True and no existing cache
        embedding.GenomeMethylationEmbedding(
            "no_cache_test",
            expected_chromosomes=["chr1", "chr2", "chr3"],
            fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
            window_size=150,
            skip_cache=False,
            verbose=True,
        )
        captured = capsys.readouterr()
        # The verbose message about cache not found should appear
        assert "Could not load" in captured.out or "No cache available" in captured.out
    finally:
        os.chdir(original_cwd)


def test_save_cache_verbose(tmp_path, capsys) -> None:
    """Test verbose messages when saving cache."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Create embedding with verbose=True (will save cache)
        embedding.GenomeMethylationEmbedding(
            "verbose_save_test",
            expected_chromosomes=["chr1", "chr2", "chr3"],
            fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
            window_size=150,
            skip_cache=False,
            verbose=True,
        )
        captured = capsys.readouterr()
        # Check for verbose save messages
        assert "Saving embedding to cache" in captured.out
        assert "Saved embedding cache" in captured.out
    finally:
        os.chdir(original_cwd)
