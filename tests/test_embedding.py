import gzip
import json
import os

import pytest

from bam2tensor import __version__, embedding

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


# -- Regression: soft-masked FASTAs must enumerate every CpG -----------------


def _write_softmasked_fasta(path: str) -> None:
    """Write a tiny mixed-case FASTA with CG/cg/Cg/cG to ``path``.

    UCSC's default ``hg38.fa.gz`` is soft-masked (lowercase = RepeatMasker
    / Tandem Repeats Finder). Prior to v2.7 the CpG enumeration used a
    case-sensitive search and silently dropped roughly half the CpGs in
    soft-masked references — this fixture pins the regression.
    """
    # 1-based positions of the C in each C[gG] (0-based + 1):
    #   CG @ pos  6, cg @ pos 11, Cg @ pos 17, cG @ pos 23
    seq = "AAAAACGAAAcgAAAACgAAAAcGAAAAA"
    with open(path, "w") as fh:
        fh.write(">chr1\n")
        fh.write(seq + "\n")


def test_soft_masked_fasta_enumerates_all_cpgs(tmp_path) -> None:
    """All CpGs are found regardless of case (regression for v2.7 case fix).

    Bug: prior to v2.7, ``sequence.find('CG')`` on a soft-masked FASTA
    missed every lowercase or mixed-case ``cg``/``Cg``/``cG``. On real
    UCSC hg38 that silently dropped ~53% of all CpG sites.
    """
    fasta = tmp_path / "softmasked.fa"
    _write_softmasked_fasta(str(fasta))

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        emb = embedding.GenomeMethylationEmbedding(
            "softmasked_test",
            expected_chromosomes=["chr1"],
            fasta_source=str(fasta),
            window_size=150,
            skip_cache=True,
            verbose=False,
        )
    finally:
        os.chdir(original_cwd)

    assert emb.total_cpg_sites == 4
    assert emb.cpg_sites_dict["chr1"] == [6, 11, 17, 23]


# -- Cache provenance: stamp + reject stale ---------------------------------


def _read_cache(path: str) -> dict:
    """Read and decode a gzipped JSON cache file."""
    with gzip.open(path, "rt") as f:
        return json.load(f)


def test_cache_stamps_provenance(tmp_path) -> None:
    """Saved cache embeds bam2tensor_version, fasta_sha256, total_cpg_sites."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        emb = embedding.GenomeMethylationEmbedding(
            "provenance_test",
            expected_chromosomes=["chr1", "chr2", "chr3"],
            fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
            window_size=150,
            skip_cache=False,
            verbose=False,
        )
        data = _read_cache("provenance_test.cache.json.gz")
    finally:
        os.chdir(original_cwd)

    assert data["bam2tensor_version"] == __version__
    assert data["total_cpg_sites"] == emb.total_cpg_sites
    assert isinstance(data["fasta_sha256"], str)
    assert len(data["fasta_sha256"]) == 64  # hex-encoded SHA-256


def test_cache_rejected_on_fasta_sha256_mismatch(tmp_path, capsys) -> None:
    """A cache whose ``fasta_sha256`` no longer matches the current FASTA is discarded.

    This is the primary safety net for the upgrade to v2.7: any user
    with a pre-v2.7 cache (or any user who switches between soft- and
    hard-masked FASTAs) will get a clean re-parse rather than silently
    misaligned column indices.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # First run: generate cache against the real test FASTA.
        emb1 = embedding.GenomeMethylationEmbedding(
            "sha_mismatch_test",
            expected_chromosomes=["chr1", "chr2", "chr3"],
            fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
            window_size=150,
            skip_cache=False,
            verbose=False,
        )
        cache_path = "sha_mismatch_test.cache.json.gz"
        assert os.path.exists(cache_path)

        # Tamper with the cached fasta_sha256 so it can no longer match
        # the on-disk FASTA. This simulates a user upgrading bam2tensor
        # with an inherited v2.5 cache whose enumeration is stale.
        data = _read_cache(cache_path)
        data["fasta_sha256"] = "0" * 64
        with gzip.open(cache_path, "wt") as f:
            json.dump(data, f)

        # Second run: the stale cache must be rejected and regenerated.
        emb2 = embedding.GenomeMethylationEmbedding(
            "sha_mismatch_test",
            expected_chromosomes=["chr1", "chr2", "chr3"],
            fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
            window_size=150,
            skip_cache=False,
            verbose=False,
        )

        captured = capsys.readouterr()
        assert "Discarding stale embedding cache" in captured.out
        assert "FASTA SHA-256 mismatch" in captured.out

        # Regenerated cache: same content as the first run.
        assert emb2.total_cpg_sites == emb1.total_cpg_sites
        # And the on-disk cache now stamps the real SHA-256, not the tampered one.
        refreshed = _read_cache(cache_path)
        assert refreshed["fasta_sha256"] != "0" * 64
    finally:
        os.chdir(original_cwd)


def test_cache_rejected_on_version_mismatch(tmp_path, capsys) -> None:
    """A cache stamped with an older bam2tensor version is discarded."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        embedding.GenomeMethylationEmbedding(
            "version_mismatch_test",
            expected_chromosomes=["chr1", "chr2", "chr3"],
            fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
            window_size=150,
            skip_cache=False,
            verbose=False,
        )
        cache_path = "version_mismatch_test.cache.json.gz"

        # Roll the version back to a release that predates the
        # case-sensitivity fix; any non-current version should trigger
        # rejection.
        data = _read_cache(cache_path)
        data["bam2tensor_version"] = "2.5"
        with gzip.open(cache_path, "wt") as f:
            json.dump(data, f)

        embedding.GenomeMethylationEmbedding(
            "version_mismatch_test",
            expected_chromosomes=["chr1", "chr2", "chr3"],
            fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
            window_size=150,
            skip_cache=False,
            verbose=False,
        )

        captured = capsys.readouterr()
        assert "Discarding stale embedding cache" in captured.out
        assert "bam2tensor" in captured.out
        assert "2.5" in captured.out
    finally:
        os.chdir(original_cwd)


def test_cache_rejected_when_provenance_absent(tmp_path, capsys) -> None:
    """A pre-v2.7 cache without provenance fields is treated as stale.

    Users on v2.5 / v2.6 have caches with no ``bam2tensor_version`` or
    ``fasta_sha256``. After upgrading, those caches must be discarded so
    the case-sensitivity fix actually takes effect on their data.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        embedding.GenomeMethylationEmbedding(
            "legacy_cache_test",
            expected_chromosomes=["chr1", "chr2", "chr3"],
            fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
            window_size=150,
            skip_cache=False,
            verbose=False,
        )
        cache_path = "legacy_cache_test.cache.json.gz"

        # Strip provenance fields to mimic a v2.5-era cache file.
        data = _read_cache(cache_path)
        data.pop("bam2tensor_version", None)
        data.pop("fasta_sha256", None)
        data.pop("total_cpg_sites", None)
        with gzip.open(cache_path, "wt") as f:
            json.dump(data, f)

        embedding.GenomeMethylationEmbedding(
            "legacy_cache_test",
            expected_chromosomes=["chr1", "chr2", "chr3"],
            fasta_source=os.path.join(original_cwd, "tests/test_fasta.fa"),
            window_size=150,
            skip_cache=False,
            verbose=False,
        )

        captured = capsys.readouterr()
        assert "Discarding stale embedding cache" in captured.out
    finally:
        os.chdir(original_cwd)
