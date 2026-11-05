"""Test cases for the embedding module.

Every test runs in its own temporary working directory (see ``conftest.py``),
so the ``<genome>.cache.json.gz`` files written by ``skip_cache=False`` land
there rather than in the repository checkout.
"""

import gzip
import json
import os

import pytest

from bam2tensor import __version__, embedding
from tests.paths import TEST_FASTA

CHROMOSOMES = ["chr1", "chr2", "chr3"]
TOTAL_CPG_SITES = 37489  # Known from test_fasta.fa


def _make_embedding(
    genome_name, expected_chromosomes=None, fasta_source=TEST_FASTA, **kwargs
):
    """Build an embedding of the bundled test FASTA (chr1-chr3 by default)."""
    return embedding.GenomeMethylationEmbedding(
        genome_name,
        expected_chromosomes=(
            list(CHROMOSOMES) if expected_chromosomes is None else expected_chromosomes
        ),
        fasta_source=fasta_source,
        **kwargs,
    )


def _read_cache(path: str) -> dict:
    """Read and decode a gzipped JSON cache file."""
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _write_cache(path: str, data: dict) -> None:
    """Overwrite a gzipped JSON cache file with ``data``."""
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


@pytest.fixture(scope="module")
def test_embeddings(tmp_path_factory):
    """The test FASTA embedded three ways: uncached, cache-writing, cache-loading.

    Module-scoped fixtures are set up before the per-test working-directory
    fixture, so this one manages its own working directory.
    """
    cache_dir = tmp_path_factory.mktemp("embedding_cache")
    original_cwd = os.getcwd()
    os.chdir(cache_dir)
    try:
        # Generate a fresh, uncached embedding
        fresh = _make_embedding("test_genome", skip_cache=True, verbose=True)
        assert not (cache_dir / "test_genome.cache.json.gz").exists()

        # Generate a cached version of the embedding
        cache_generate = _make_embedding("test_genome", skip_cache=False, verbose=True)
        assert (cache_dir / "test_genome.cache.json.gz").exists()

        # Load the cached version of the embedding
        cache_load = _make_embedding("test_genome", skip_cache=False, verbose=False)
    finally:
        os.chdir(original_cwd)
    return [fresh, cache_generate, cache_load]


def test_genome_methylation_embedding_init(test_embeddings) -> None:
    """Test GenomeMethylationEmbedding."""
    test_embedding = test_embeddings[0]

    assert test_embedding.genome_name == "test_genome"
    assert test_embedding.total_cpg_sites == TOTAL_CPG_SITES

    assert test_embedding.expected_chromosomes == CHROMOSOMES

    assert test_embedding.cpgs_per_chr_cumsum[-1] == test_embedding.total_cpg_sites


def test_cached_embedding(test_embeddings) -> None:
    """Fresh, cache-writing and cache-loading embeddings are identical."""
    fresh, cache_generate, cache_load = test_embeddings

    for embedding_obj in test_embeddings:
        assert embedding_obj.total_cpg_sites == TOTAL_CPG_SITES
        assert embedding_obj.expected_chromosomes == CHROMOSOMES
        assert embedding_obj.fasta_source == TEST_FASTA

    assert cache_generate.cpg_sites_dict == fresh.cpg_sites_dict
    assert cache_load.cpg_sites_dict == fresh.cpg_sites_dict
    assert list(cache_load.cpg_sites_dict) == list(fresh.cpg_sites_dict)


def test_cache_is_loaded_on_second_run(capsys) -> None:
    """The second construction reads the cache instead of re-parsing the FASTA."""
    first = _make_embedding("round_trip")
    assert "No cache available" in capsys.readouterr().out
    assert os.path.exists("round_trip.cache.json.gz")

    second = _make_embedding("round_trip")
    captured = capsys.readouterr()
    assert "Loaded methylation embedding from cache" in captured.out
    assert "No cache available" not in captured.out
    assert second.cpg_sites_dict == first.cpg_sites_dict


def test_embedding_to_genomic_position(test_embeddings) -> None:
    """Test embedding_to_genomic_position."""

    for embedding_obj in test_embeddings:
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


def test_genomic_position_to_embedding(test_embeddings) -> None:
    """Test genomic_position_to_embedding."""

    for embedding_obj in test_embeddings:
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


def test_position_conversions_round_trip(test_embeddings) -> None:
    """Column -> (chromosome, position) -> column is the identity everywhere."""
    embedding_obj = test_embeddings[0]
    # Every 97th column plus both ends of each chromosome's column block.
    columns = set(range(0, embedding_obj.total_cpg_sites, 97))
    for boundary in embedding_obj.cpgs_per_chr_cumsum:
        columns.update({int(boundary) - 1, int(boundary)})
    columns.discard(embedding_obj.total_cpg_sites)

    for column in sorted(columns):
        chrom, pos = embedding_obj.embedding_to_genomic_position(column)
        assert embedding_obj.genomic_position_to_embedding(chrom, pos) == column


def test_genomic_position_to_embedding_rejects_non_cpg_positions(
    test_embeddings,
) -> None:
    """A position that is not a CpG site (or an unknown chromosome) is a KeyError."""
    embedding_obj = test_embeddings[0]
    first_cpg = embedding_obj.cpg_sites_dict["chr1"][0]
    # The G of the first CpG is, by construction, not itself a CpG C position
    # unless the sequence is CGCG..., so pick a position that is certainly absent.
    non_cpg = next(
        pos
        for pos in range(1, first_cpg + 2)
        if pos not in set(embedding_obj.cpg_sites_dict["chr1"][:5])
    )
    with pytest.raises(KeyError):
        embedding_obj.genomic_position_to_embedding("chr1", non_cpg)
    with pytest.raises(KeyError):
        embedding_obj.genomic_position_to_embedding("chrNOPE", first_cpg)


def test_empty_chromosomes_raises_error() -> None:
    """Test that empty expected_chromosomes raises ValueError."""
    with pytest.raises(ValueError, match="Expected chromosomes cannot be empty"):
        _make_embedding("test_genome", expected_chromosomes=[], skip_cache=True)


def test_missing_fasta_raises_error() -> None:
    """Test that missing fasta file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Cannot read fasta file"):
        _make_embedding(
            "test_genome",
            expected_chromosomes=["chr1"],
            fasta_source="/nonexistent/file.fa",
            skip_cache=True,
        )


# -- Expected chromosomes that the FASTA does not contain ---------------------


@pytest.mark.parametrize("skip_cache", [True, False])
def test_expected_chromosome_missing_from_fasta(skip_cache) -> None:
    """A chromosome absent from the FASTA is a clear ValueError, not a KeyError.

    Regression: this used to surface as a bare ``KeyError: 'chrM'`` from the
    column-layout code — and, with caching on, only *after* a cache lacking
    the chromosome had already been written to disk.
    """
    with pytest.raises(ValueError) as excinfo:
        _make_embedding(
            "missing_chrom",
            expected_chromosomes=["chr1", "chrM", "chr2", "chrUn"],
            skip_cache=skip_cache,
        )

    message = str(excinfo.value)
    assert "Expected chromosome(s) not found in reference FASTA" in message
    assert TEST_FASTA in message
    assert "chrM, chrUn" in message  # the missing names, in the order requested
    # ...and what the FASTA does offer.
    assert "4 sequence(s): chr1, chr2, chr3, chrIGNORED" in message
    assert "--expected-chromosomes" in message

    # No half-built cache may be left behind for the next run to trip over.
    assert not os.path.exists("missing_chrom.cache.json.gz")
    assert os.listdir(".") == []


def test_no_expected_chromosome_in_fasta() -> None:
    """A FASTA sharing no names with the expected list is reported the same way.

    Regression: with caching on this used to be ``AssertionError: CpG sites
    dict is empty!``.
    """
    with pytest.raises(ValueError, match="not found in reference FASTA"):
        _make_embedding("no_overlap", expected_chromosomes=["scaffold_1", "scaffold_2"])
    assert os.listdir(".") == []


def test_missing_chromosome_hint_fasta_is_ensembl_style(tmp_path) -> None:
    """UCSC names requested against an Ensembl-named FASTA: say so."""
    fasta = tmp_path / "ensembl.fa"
    fasta.write_text(">1\nACGTACGT\n>2\nACGTACGT\n>MT\nACGT\n")

    with pytest.raises(ValueError) as excinfo:
        _make_embedding(
            "ensembl_fasta",
            expected_chromosomes=["chr1", "chr2"],
            fasta_source=str(fasta),
            skip_cache=True,
        )
    message = str(excinfo.value)
    assert "Ensembl-style names (no 'chr' prefix)" in message
    assert "--expected-chromosomes '1,2,3,...'" in message


def test_missing_chromosome_hint_fasta_is_ucsc_style(tmp_path) -> None:
    """Ensembl names requested against a UCSC-named FASTA: say so."""
    fasta = tmp_path / "ucsc.fa"
    fasta.write_text(">chr1\nACGTACGT\n>chr2\nACGTACGT\n")

    with pytest.raises(ValueError) as excinfo:
        _make_embedding(
            "ucsc_fasta",
            expected_chromosomes=["1", "2"],
            fasta_source=str(fasta),
            skip_cache=True,
        )
    message = str(excinfo.value)
    assert "UCSC-style names ('chr' prefix)" in message
    assert "--expected-chromosomes 'chr1,chr2,chr3,...'" in message


def test_missing_chromosome_message_truncates_long_lists(tmp_path) -> None:
    """A FASTA with many sequences (alt contigs) does not flood the message."""
    fasta = tmp_path / "many.fa"
    fasta.write_text("".join(f">contig{i}\nACGT\n" for i in range(50)))

    with pytest.raises(ValueError) as excinfo:
        _make_embedding(
            "many_contigs",
            expected_chromosomes=["chr1"],
            fasta_source=str(fasta),
            skip_cache=True,
        )
    message = str(excinfo.value)
    assert "50 sequence(s): contig0, contig1" in message
    assert "... (50 total)" in message
    assert "contig49" not in message


def test_cache_lacking_expected_chromosome_is_regenerated(capsys) -> None:
    """A cache written without one of its own chromosomes is discarded.

    Releases up to 2.8 saved the cache *before* discovering that an expected
    chromosome was absent from the FASTA, leaving a cache that could never be
    used. Such a cache must not poison later runs.
    """
    _make_embedding("poisoned")
    cache_path = "poisoned.cache.json.gz"
    data = _read_cache(cache_path)
    del data["cpg_sites_dict"]["chr3"]
    _write_cache(cache_path, data)
    capsys.readouterr()

    emb = _make_embedding("poisoned")

    captured = capsys.readouterr()
    assert "Discarding stale embedding cache" in captured.out
    assert "no CpG sites stored for chr3" in captured.out
    assert emb.total_cpg_sites == TOTAL_CPG_SITES
    assert sorted(_read_cache(cache_path)["cpg_sites_dict"]) == CHROMOSOMES


# -- Cache configuration mismatches -------------------------------------------


def test_cache_chromosome_mismatch() -> None:
    """A cache built for a different chromosome list is an explicit ValueError.

    The cache is keyed only by genome name, so the same name used with two
    chromosome lists is ambiguous. It must neither be used (wrong columns)
    nor silently overwritten — and it must be a real exception rather than an
    ``assert``, which ``python -O`` would strip.
    """
    _make_embedding("mismatch_test")
    cache_path = "mismatch_test.cache.json.gz"
    assert os.path.exists(cache_path)
    before = _read_cache(cache_path)

    with pytest.raises(ValueError) as excinfo:
        _make_embedding("mismatch_test", expected_chromosomes=["chr1"])  # Different!

    message = str(excinfo.value)
    assert "Expected chromosomes do not match the cache" in message
    assert "built for [chr1, chr2, chr3] but [chr1] were requested" in message
    # Tells the user how to get unstuck.
    assert "different genome name" in message
    assert "--skip-cache" in message

    # The existing cache is left exactly as it was.
    assert _read_cache(cache_path) == before


def test_cache_chromosome_order_mismatch() -> None:
    """Chromosome order defines the column layout, so a reordering is a mismatch."""
    _make_embedding("order_test")
    with pytest.raises(ValueError, match="Expected chromosomes do not match"):
        _make_embedding("order_test", expected_chromosomes=["chr3", "chr2", "chr1"])


def test_cache_chromosome_mismatch_can_be_bypassed_with_skip_cache() -> None:
    """``skip_cache=True`` ignores the mismatched cache, as the message advises."""
    _make_embedding("bypass_test")
    emb = _make_embedding("bypass_test", expected_chromosomes=["chr1"], skip_cache=True)
    assert emb.expected_chromosomes == ["chr1"]
    assert emb.total_cpg_sites == len(emb.cpg_sites_dict["chr1"])


def test_cache_window_size_mismatch() -> None:
    """Test that mismatched window_size between cache and request raises ValueError."""
    _make_embedding("window_test", window_size=150)

    with pytest.raises(ValueError, match="Window size does not match the cache"):
        _make_embedding("window_test", window_size=200)  # Different!


# -- Verbose output ------------------------------------------------------------


def test_cache_not_found_verbose(capsys) -> None:
    """Test verbose message when cache file not found."""
    _make_embedding("no_cache_test", verbose=True)
    captured = capsys.readouterr()
    # The verbose message about cache not found should appear
    assert "Could not load" in captured.out or "No cache available" in captured.out


def test_save_cache_verbose(capsys) -> None:
    """Test verbose messages when saving cache."""
    _make_embedding("verbose_save_test", verbose=True)
    captured = capsys.readouterr()
    # Check for verbose save messages
    assert "Saving embedding to cache" in captured.out
    assert "Saved embedding cache" in captured.out


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

    emb = _make_embedding(
        "softmasked_test",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta),
        skip_cache=True,
    )

    assert emb.total_cpg_sites == 4
    assert emb.cpg_sites_dict["chr1"] == [6, 11, 17, 23]


# -- Cache provenance: stamp + reject stale ---------------------------------


def test_cache_stamps_provenance() -> None:
    """Saved cache embeds bam2tensor_version, fasta_sha256, total_cpg_sites."""
    emb = _make_embedding("provenance_test")
    data = _read_cache("provenance_test.cache.json.gz")

    assert data["bam2tensor_version"] == __version__
    assert data["total_cpg_sites"] == emb.total_cpg_sites
    assert isinstance(data["fasta_sha256"], str)
    assert len(data["fasta_sha256"]) == 64  # hex-encoded SHA-256
    assert data["genome_name"] == "provenance_test"
    assert data["fasta_source"] == TEST_FASTA
    assert data["expected_chromosomes"] == CHROMOSOMES


def test_cache_rejected_on_fasta_sha256_mismatch(capsys) -> None:
    """A cache whose ``fasta_sha256`` no longer matches the current FASTA is discarded.

    This is the primary safety net for the upgrade to v2.7: any user
    with a pre-v2.7 cache (or any user who switches between soft- and
    hard-masked FASTAs) will get a clean re-parse rather than silently
    misaligned column indices.
    """
    # First run: generate cache against the real test FASTA.
    emb1 = _make_embedding("sha_mismatch_test")
    cache_path = "sha_mismatch_test.cache.json.gz"
    assert os.path.exists(cache_path)

    # Tamper with the cached fasta_sha256 so it can no longer match
    # the on-disk FASTA. This simulates a user upgrading bam2tensor
    # with an inherited v2.5 cache whose enumeration is stale.
    data = _read_cache(cache_path)
    data["fasta_sha256"] = "0" * 64
    _write_cache(cache_path, data)

    # Second run: the stale cache must be rejected and regenerated.
    emb2 = _make_embedding("sha_mismatch_test")

    captured = capsys.readouterr()
    assert "Discarding stale embedding cache" in captured.out
    assert "FASTA SHA-256 mismatch" in captured.out

    # Regenerated cache: same content as the first run.
    assert emb2.total_cpg_sites == emb1.total_cpg_sites
    # And the on-disk cache now stamps the real SHA-256, not the tampered one.
    refreshed = _read_cache(cache_path)
    assert refreshed["fasta_sha256"] != "0" * 64


def test_cache_rejected_when_fasta_content_changes(tmp_path, capsys) -> None:
    """Editing the FASTA in place invalidates the cache built from it."""
    fasta = tmp_path / "ref.fa"
    fasta.write_text(">chr1\nAACGAACGAA\n")
    emb1 = _make_embedding(
        "content_change", expected_chromosomes=["chr1"], fasta_source=str(fasta)
    )
    assert emb1.cpg_sites_dict["chr1"] == [3, 7]

    # Same path, same length, different CpG content (hard-masking one CpG).
    fasta.write_text(">chr1\nAACGAANNAA\n")
    capsys.readouterr()
    emb2 = _make_embedding(
        "content_change", expected_chromosomes=["chr1"], fasta_source=str(fasta)
    )

    assert "Discarding stale embedding cache" in capsys.readouterr().out
    assert emb2.cpg_sites_dict["chr1"] == [3]


def test_cache_rejected_on_version_mismatch(capsys) -> None:
    """A cache stamped with an older bam2tensor version is discarded."""
    _make_embedding("version_mismatch_test")
    cache_path = "version_mismatch_test.cache.json.gz"

    # Roll the version back to a release that predates the
    # case-sensitivity fix; any non-current version should trigger
    # rejection.
    data = _read_cache(cache_path)
    data["bam2tensor_version"] = "2.5"
    _write_cache(cache_path, data)

    _make_embedding("version_mismatch_test")

    captured = capsys.readouterr()
    assert "Discarding stale embedding cache" in captured.out
    assert "bam2tensor" in captured.out
    assert "2.5" in captured.out
    assert _read_cache(cache_path)["bam2tensor_version"] == __version__


def test_cache_rejected_when_provenance_absent(capsys) -> None:
    """A pre-v2.7 cache without provenance fields is treated as stale.

    Users on v2.5 / v2.6 have caches with no ``bam2tensor_version`` or
    ``fasta_sha256``. After upgrading, those caches must be discarded so
    the case-sensitivity fix actually takes effect on their data.
    """
    _make_embedding("legacy_cache_test")
    cache_path = "legacy_cache_test.cache.json.gz"

    # Strip provenance fields to mimic a v2.5-era cache file.
    data = _read_cache(cache_path)
    data.pop("bam2tensor_version", None)
    data.pop("fasta_sha256", None)
    data.pop("total_cpg_sites", None)
    _write_cache(cache_path, data)

    _make_embedding("legacy_cache_test")

    captured = capsys.readouterr()
    assert "Discarding stale embedding cache" in captured.out


# -- Unreadable caches: recover instead of crashing -----------------------------


def _assert_cache_was_regenerated(genome_name: str, capsys) -> None:
    """Build the embedding over a broken cache and check that it recovered."""
    cache_path = f"{genome_name}.cache.json.gz"
    capsys.readouterr()

    emb = _make_embedding(genome_name)

    captured = capsys.readouterr()
    assert "Discarding stale embedding cache" in captured.out
    assert "Unreadable cache" in captured.out
    assert emb.total_cpg_sites == TOTAL_CPG_SITES

    # The broken file has been replaced by a complete, loadable cache...
    assert _read_cache(cache_path)["total_cpg_sites"] == TOTAL_CPG_SITES
    # ...which the next run then uses.
    _make_embedding(genome_name)
    assert "Loaded methylation embedding from cache" in capsys.readouterr().out


def test_truncated_cache_is_regenerated(capsys) -> None:
    """A cache cut short mid-write is discarded and rebuilt.

    Regression: interrupting a run while the cache was being written left a
    truncated gzip, and every later run died with ``EOFError`` until the user
    worked out which file to delete.
    """
    _make_embedding("truncated")
    cache_path = "truncated.cache.json.gz"
    with open(cache_path, "rb") as f:
        complete = f.read()
    with open(cache_path, "wb") as f:
        f.write(complete[: len(complete) // 2])

    _assert_cache_was_regenerated("truncated", capsys)


def test_non_gzip_cache_is_regenerated(capsys) -> None:
    """A cache file that is not gzip data at all is discarded and rebuilt."""
    with open("garbage.cache.json.gz", "wb") as f:
        f.write(b"this is not a gzip file")

    _assert_cache_was_regenerated("garbage", capsys)


def test_empty_cache_file_is_regenerated(capsys) -> None:
    """A zero-byte cache file is discarded and rebuilt."""
    open("empty.cache.json.gz", "wb").close()

    _assert_cache_was_regenerated("empty", capsys)


def test_non_json_cache_is_regenerated(capsys) -> None:
    """Valid gzip holding invalid JSON is discarded and rebuilt."""
    with gzip.open("notjson.cache.json.gz", "wt") as f:
        f.write('{"genome_name": "notjson", "cpg_sites_dict": {"chr1": [1, 2')

    _assert_cache_was_regenerated("notjson", capsys)


def test_non_object_json_cache_is_regenerated(capsys) -> None:
    """Valid JSON of the wrong shape (not an object) is discarded and rebuilt."""
    _write_cache("wrongshape.cache.json.gz", [1, 2, 3])  # type: ignore[arg-type]

    _assert_cache_was_regenerated("wrongshape", capsys)


def test_cache_without_cpg_site_data_is_regenerated(capsys) -> None:
    """A cache that passes provenance checks but holds no CpG data is rebuilt."""
    _make_embedding("hollow")
    data = _read_cache("hollow.cache.json.gz")
    del data["cpg_sites_dict"]
    _write_cache("hollow.cache.json.gz", data)

    _assert_cache_was_regenerated("hollow", capsys)


# -- Atomic cache writes ---------------------------------------------------------


def test_cache_write_leaves_no_temporary_files() -> None:
    """A successful save leaves exactly the cache file behind."""
    _make_embedding("tidy")
    assert os.listdir(".") == ["tidy.cache.json.gz"]


def test_interrupted_cache_write_leaves_no_partial_cache(monkeypatch) -> None:
    """A save that dies part-way leaves no cache (and no temp file) behind.

    The cache is written to a temporary file and renamed into place, so a
    failure — or Ctrl-C — while serialising must not leave a truncated
    ``<genome>.cache.json.gz`` for the next run to choke on.
    """

    def exploding_dump(obj, fp, *args, **kwargs):
        fp.write('{"genome_name": "interrupted", "cpg_sites_dict": {"chr1": [1, 2, 3')
        raise KeyboardInterrupt

    monkeypatch.setattr(embedding.json, "dump", exploding_dump)

    with pytest.raises(KeyboardInterrupt):
        _make_embedding("interrupted")

    assert os.listdir(".") == []


def test_interrupted_cache_rewrite_keeps_previous_cache(monkeypatch, capsys) -> None:
    """A failed rewrite of a stale cache leaves the previous file untouched."""
    _make_embedding("rewrite")
    cache_path = "rewrite.cache.json.gz"
    # Make the cache stale so the next run regenerates and re-saves it.
    data = _read_cache(cache_path)
    data["bam2tensor_version"] = "2.5"
    _write_cache(cache_path, data)
    with open(cache_path, "rb") as f:
        before = f.read()

    real_dump = json.dump

    def exploding_dump(obj, fp, *args, **kwargs):
        fp.write("{")
        raise OSError("No space left on device")

    monkeypatch.setattr(embedding.json, "dump", exploding_dump)
    with pytest.raises(OSError, match="No space left on device"):
        _make_embedding("rewrite")

    # Byte-for-byte the file that was there before, and nothing else.
    with open(cache_path, "rb") as f:
        assert f.read() == before
    assert os.listdir(".") == [cache_path]

    # Once the write can succeed again, the stale cache is replaced.
    monkeypatch.setattr(embedding.json, "dump", real_dump)
    _make_embedding("rewrite")
    assert _read_cache(cache_path)["bam2tensor_version"] == __version__


# -- Resource handling -----------------------------------------------------------


@pytest.mark.filterwarnings("error::ResourceWarning")
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_fasta_handle_is_closed_after_parsing(tmp_path) -> None:
    """Parsing closes the FASTA, both on success and when it raises."""
    import gc

    _make_embedding("handle_ok", skip_cache=True)
    with pytest.raises(ValueError, match="not found in reference FASTA"):
        _make_embedding("handle_err", expected_chromosomes=["chrM"], skip_cache=True)
    gc.collect()  # surface any ResourceWarning from an unclosed handle now
