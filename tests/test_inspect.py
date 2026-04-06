"""Test cases for the inspect module."""

import shutil

import numpy as np
import scipy.sparse
from click.testing import CliRunner

from bam2tensor import __main__
from bam2tensor.inspect import _format_size
from bam2tensor.inspect import main as inspect_main
from bam2tensor.metadata import write_npz_metadata, write_npz_tlen


def test_inspect_with_metadata(tmp_path) -> None:
    """Inspect prints metadata fields when present."""
    npz_path = str(tmp_path / "sample.methylation.npz")
    matrix = scipy.sparse.coo_matrix(([1, 0, -1], ([0, 0, 1], [0, 2, 1])), shape=(2, 5))
    scipy.sparse.save_npz(npz_path, matrix)
    write_npz_metadata(
        npz_path,
        {
            "bam2tensor_version": "2.4",
            "genome_name": "hg38",
            "expected_chromosomes": ["chr1", "chr2", "chrX", "chrY"],
            "total_cpg_sites": 5,
            "cpg_index_crc32": "deadbeef",
        },
    )

    runner = CliRunner()
    result = runner.invoke(inspect_main, [npz_path])
    assert result.exit_code == 0
    assert "hg38" in result.output
    assert "Reads:" in result.output
    assert "2" in result.output  # 2 reads
    assert "CpG sites:" in result.output
    assert "deadbeef" in result.output
    assert "v2.4" in result.output
    assert "chr1, chr2, chrX, chrY" in result.output


def test_inspect_without_metadata(tmp_path) -> None:
    """Inspect works on files without metadata (older bam2tensor)."""
    npz_path = str(tmp_path / "old.methylation.npz")
    matrix = scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(1, 100))
    scipy.sparse.save_npz(npz_path, matrix)

    runner = CliRunner()
    result = runner.invoke(inspect_main, [npz_path])
    assert result.exit_code == 0
    assert "Reads:" in result.output
    assert "older bam2tensor" in result.output
    # Should NOT have genome or CRC lines
    assert "Genome:" not in result.output


def test_inspect_multiple_files(tmp_path) -> None:
    """Inspect handles multiple files with blank line separator."""
    paths = []
    for name in ["a.npz", "b.npz"]:
        p = str(tmp_path / name)
        matrix = scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(1, 10))
        scipy.sparse.save_npz(p, matrix)
        paths.append(p)

    runner = CliRunner()
    result = runner.invoke(inspect_main, paths)
    assert result.exit_code == 0
    assert "a.npz" in result.output
    assert "b.npz" in result.output


def test_inspect_many_chromosomes(tmp_path) -> None:
    """Chromosome list is summarised when > 4 entries."""
    npz_path = str(tmp_path / "matrix.npz")
    matrix = scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(1, 10))
    scipy.sparse.save_npz(npz_path, matrix)
    chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    write_npz_metadata(
        npz_path,
        {
            "expected_chromosomes": chroms,
            "genome_name": "hg38",
        },
    )

    runner = CliRunner()
    result = runner.invoke(inspect_main, [npz_path])
    assert "24 (" in result.output
    assert "chrY" in result.output


def test_inspect_end_to_end(tmp_path) -> None:
    """Full pipeline: bam2tensor produces file, bam2tensor-inspect reads it."""
    shutil.copy("tests/test.bam", tmp_path / "test.bam")
    shutil.copy("tests/test.bam.bai", tmp_path / "test.bam.bai")

    runner = CliRunner()
    # Run extraction
    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            str(tmp_path / "test.bam"),
            "--reference-fasta",
            "tests/test_fasta.fa",
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            "--output-dir",
            str(tmp_path / "out"),
            "--overwrite",
        ],
    )
    assert result.exit_code == 0

    # Inspect the output
    npz_path = str(tmp_path / "out" / "test.methylation.npz")
    result = runner.invoke(inspect_main, [npz_path])
    assert result.exit_code == 0
    assert "test" in result.output  # genome_name
    assert "CpG index CRC32:" in result.output
    assert "v2.4" in result.output


def test_format_size_bytes() -> None:
    """_format_size handles small byte counts."""
    assert _format_size(500) == "500 bytes"


def test_format_size_kb() -> None:
    """_format_size handles kilobyte range."""
    assert _format_size(2048) == "2.0 KB"


def test_format_size_mb() -> None:
    """_format_size handles megabyte range."""
    result = _format_size(14_200_000)
    assert "MB" in result


def test_format_size_gb() -> None:
    """_format_size handles gigabyte range."""
    result = _format_size(2_500_000_000)
    assert "GB" in result


def test_inspect_with_tlen(tmp_path) -> None:
    """Inspect prints fragment length stats when tlen.npy is present."""
    npz_path = str(tmp_path / "sample.methylation.npz")
    matrix = scipy.sparse.coo_matrix(([1, 0], ([0, 1], [0, 1])), shape=(2, 5))
    scipy.sparse.save_npz(npz_path, matrix)
    write_npz_tlen(npz_path, np.array([300, -300], dtype=np.int32))

    runner = CliRunner()
    result = runner.invoke(inspect_main, [npz_path])
    assert result.exit_code == 0
    assert "Fragment len:" in result.output
    assert "median 300" in result.output


def test_inspect_without_tlen(tmp_path) -> None:
    """Inspect does not crash on files without tlen.npy (backward compat)."""
    npz_path = str(tmp_path / "old.methylation.npz")
    matrix = scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(1, 100))
    scipy.sparse.save_npz(npz_path, matrix)

    runner = CliRunner()
    result = runner.invoke(inspect_main, [npz_path])
    assert result.exit_code == 0
    assert "Fragment len:" not in result.output


def test_inspect_tlen_all_zero(tmp_path) -> None:
    """Inspect reports single-end when all TLEN values are zero."""
    npz_path = str(tmp_path / "se.methylation.npz")
    matrix = scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(1, 10))
    scipy.sparse.save_npz(npz_path, matrix)
    write_npz_tlen(npz_path, np.array([0], dtype=np.int32))

    runner = CliRunner()
    result = runner.invoke(inspect_main, [npz_path])
    assert result.exit_code == 0
    assert "single-end" in result.output
