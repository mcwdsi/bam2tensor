"""Test cases for the metadata module."""

import json
import zipfile

import numpy as np
import scipy.sparse

from bam2tensor import embedding
from bam2tensor.metadata import (
    compute_cpg_index_crc32,
    read_npz_metadata,
    read_npz_tlen,
    write_npz_metadata,
    write_npz_tlen,
)
from tests.paths import TEST_BAM, TEST_BAM_INDEX, TEST_FASTA

TEST_EMBEDDING = embedding.GenomeMethylationEmbedding(
    "test_genome",
    expected_chromosomes=["chr1", "chr2", "chr3"],
    fasta_source=TEST_FASTA,
    skip_cache=True,  # built at import time: must not write a cache file
    verbose=False,
)


# -- compute_cpg_index_crc32 -------------------------------------------------


def test_cpg_index_crc32_deterministic() -> None:
    """Same embedding always produces the same CRC32."""
    assert compute_cpg_index_crc32(TEST_EMBEDDING) == compute_cpg_index_crc32(
        TEST_EMBEDDING
    )


def test_cpg_index_crc32_format() -> None:
    """CRC32 is an 8-character hex string."""
    crc = compute_cpg_index_crc32(TEST_EMBEDDING)
    assert len(crc) == 8
    int(crc, 16)  # must be valid hex


def test_cpg_index_crc32_differs_for_different_embeddings(tmp_path) -> None:
    """Different chromosome lists produce different CRC32 values."""
    emb_subset = embedding.GenomeMethylationEmbedding(
        "test_subset",
        expected_chromosomes=["chr1"],
        fasta_source=TEST_FASTA,
        skip_cache=True,
        verbose=False,
    )
    assert compute_cpg_index_crc32(TEST_EMBEDDING) != compute_cpg_index_crc32(
        emb_subset
    )


def test_cpg_index_crc32_value_is_stable() -> None:
    """The checksum of the bundled FASTA's CpG index is pinned.

    The checksum is written into every output file and compared across
    files, so its definition must never change between releases. This value
    was produced by bam2tensor 2.8.
    """
    assert compute_cpg_index_crc32(TEST_EMBEDDING) == "30241ef1"


def test_cpg_index_crc32_matches_its_definition(tmp_path) -> None:
    """The incremental computation equals CRC32 of the whole text at once."""
    import zlib

    for chromosomes in (["chr1", "chr2", "chr3"], ["chr3", "chr1"], ["chr2"]):
        emb = embedding.GenomeMethylationEmbedding(
            "crc_definition",
            expected_chromosomes=chromosomes,
            fasta_source=TEST_FASTA,
            skip_cache=True,
            verbose=False,
        )
        text = "\n".join(
            chrom + "\t" + ",".join(str(pos) for pos in emb.cpg_sites_dict[chrom])
            for chrom in chromosomes
        )
        expected = format(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF, "08x")
        assert compute_cpg_index_crc32(emb) == expected


def test_cpg_index_crc32_depends_on_chromosome_order() -> None:
    """Reordering chromosomes reorders columns, so the checksum must differ."""
    forward = embedding.GenomeMethylationEmbedding(
        "crc_forward",
        expected_chromosomes=["chr1", "chr3"],
        fasta_source=TEST_FASTA,
        skip_cache=True,
        verbose=False,
    )
    reverse = embedding.GenomeMethylationEmbedding(
        "crc_reverse",
        expected_chromosomes=["chr3", "chr1"],
        fasta_source=TEST_FASTA,
        skip_cache=True,
        verbose=False,
    )
    assert compute_cpg_index_crc32(forward) != compute_cpg_index_crc32(reverse)


# -- write / read round-trip -------------------------------------------------


def test_write_then_read_metadata(tmp_path) -> None:
    """Metadata survives a write-then-read round trip."""
    npz_path = str(tmp_path / "matrix.npz")
    matrix = scipy.sparse.coo_matrix(([1, 0, -1], ([0, 0, 1], [0, 2, 1])), shape=(2, 4))
    scipy.sparse.save_npz(npz_path, matrix)

    metadata = {
        "bam2tensor_version": "2.2",
        "genome_name": "hg38",
        "cpg_index_crc32": "deadbeef",
        "total_cpg_sites": 4,
        "expected_chromosomes": ["chr1", "chr2"],
    }
    write_npz_metadata(npz_path, metadata)

    loaded = read_npz_metadata(npz_path)
    assert loaded == metadata


def test_scipy_load_unaffected_by_metadata(tmp_path) -> None:
    """scipy.sparse.load_npz still works after metadata is appended."""
    npz_path = str(tmp_path / "matrix.npz")
    data = [1, 0, -1, 1, 0]
    row = [0, 0, 1, 1, 2]
    col = [0, 2, 1, 3, 2]
    matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=(3, 5))
    scipy.sparse.save_npz(npz_path, matrix)

    write_npz_metadata(npz_path, {"genome_name": "hg38"})

    loaded = scipy.sparse.load_npz(npz_path)
    assert (loaded.toarray() == matrix.toarray()).all()
    assert loaded.shape == matrix.shape
    assert loaded.nnz == matrix.nnz


def test_read_metadata_returns_none_without_metadata(tmp_path) -> None:
    """read_npz_metadata returns None for files without metadata."""
    npz_path = str(tmp_path / "plain.npz")
    matrix = scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(1, 1))
    scipy.sparse.save_npz(npz_path, matrix)

    assert read_npz_metadata(npz_path) is None


def test_metadata_accessible_via_zipfile(tmp_path) -> None:
    """Metadata is plain JSON readable with standard zipfile tools."""
    npz_path = str(tmp_path / "matrix.npz")
    matrix = scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(1, 1))
    scipy.sparse.save_npz(npz_path, matrix)

    write_npz_metadata(npz_path, {"genome_name": "mm10", "total_cpg_sites": 42})

    with zipfile.ZipFile(npz_path, "r") as zf:
        assert "metadata.json" in zf.namelist()
        raw = json.loads(zf.read("metadata.json"))
        assert raw["genome_name"] == "mm10"
        assert raw["total_cpg_sites"] == 42


# -- CLI integration (end-to-end) -------------------------------------------


def test_main_writes_metadata(tmp_path) -> None:
    """The CLI embeds metadata in the output .npz file."""
    import shutil
    from click.testing import CliRunner
    from bam2tensor import __main__

    shutil.copy(TEST_BAM, tmp_path / "test.bam")
    shutil.copy(TEST_BAM_INDEX, tmp_path / "test.bam.bai")

    runner = CliRunner()
    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            str(tmp_path / "test.bam"),
            "--reference-fasta",
            TEST_FASTA,
            "--allow-length-mismatch",
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            "--output-dir",
            str(tmp_path / "out"),
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    npz_path = str(tmp_path / "out" / "test.methylation.npz")
    meta = read_npz_metadata(npz_path)
    assert meta is not None
    assert meta["genome_name"] == "test"
    assert meta["expected_chromosomes"] == ["chr1", "chr2", "chr3"]
    assert meta["total_cpg_sites"] == TEST_EMBEDDING.total_cpg_sites
    assert len(meta["cpg_index_crc32"]) == 8
    assert "bam2tensor_version" in meta

    # Verify the sparse matrix is still loadable
    mat = scipy.sparse.load_npz(npz_path)
    assert mat.shape[1] == TEST_EMBEDDING.total_cpg_sites


# -- TLEN write / read round-trip ----------------------------------------------


def test_write_then_read_tlen(tmp_path) -> None:
    """TLEN array survives a write-then-read round trip."""
    npz_path = str(tmp_path / "matrix.npz")
    matrix = scipy.sparse.coo_matrix(([1, 0], ([0, 1], [0, 1])), shape=(2, 3))
    scipy.sparse.save_npz(npz_path, matrix)

    tlen = np.array([300, -300], dtype=np.int32)
    write_npz_tlen(npz_path, tlen)

    loaded = read_npz_tlen(npz_path)
    assert loaded is not None
    assert np.array_equal(loaded, tlen)
    assert loaded.dtype == np.int32


def test_scipy_load_unaffected_by_tlen(tmp_path) -> None:
    """scipy.sparse.load_npz still works after tlen.npy is appended."""
    npz_path = str(tmp_path / "matrix.npz")
    matrix = scipy.sparse.coo_matrix(([1, 0, -1], ([0, 0, 1], [0, 2, 1])), shape=(2, 4))
    scipy.sparse.save_npz(npz_path, matrix)

    write_npz_tlen(npz_path, np.array([250, -250], dtype=np.int32))

    loaded = scipy.sparse.load_npz(npz_path)
    assert (loaded.toarray() == matrix.toarray()).all()
    assert loaded.shape == matrix.shape


def test_read_tlen_returns_none_without_tlen(tmp_path) -> None:
    """read_npz_tlen returns None for files without tlen.npy."""
    npz_path = str(tmp_path / "plain.npz")
    matrix = scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(1, 1))
    scipy.sparse.save_npz(npz_path, matrix)

    assert read_npz_tlen(npz_path) is None


def test_tlen_and_metadata_coexist(tmp_path) -> None:
    """Both tlen.npy and metadata.json can be read from the same file."""
    npz_path = str(tmp_path / "matrix.npz")
    matrix = scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(1, 2))
    scipy.sparse.save_npz(npz_path, matrix)

    tlen = np.array([200], dtype=np.int32)
    write_npz_tlen(npz_path, tlen)
    write_npz_metadata(npz_path, {"genome_name": "hg38"})

    loaded_tlen = read_npz_tlen(npz_path)
    loaded_meta = read_npz_metadata(npz_path)

    assert loaded_tlen is not None
    assert np.array_equal(loaded_tlen, tlen)
    assert loaded_meta is not None
    assert loaded_meta["genome_name"] == "hg38"
