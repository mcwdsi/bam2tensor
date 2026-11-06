"""Test cases for the inspect module."""

import shutil

import numpy as np
import scipy.sparse
from click.testing import CliRunner

from bam2tensor import __main__, __version__
from bam2tensor.inspect import _format_size
from bam2tensor.inspect import main as inspect_main
from bam2tensor.metadata import write_npz_metadata, write_npz_tlen
from tests.paths import TEST_BAM, TEST_BAM_INDEX, TEST_FASTA


def test_inspect_with_metadata(tmp_path) -> None:
    """Inspect prints metadata fields when present."""
    npz_path = str(tmp_path / "sample.methylation.npz")
    matrix = scipy.sparse.coo_matrix(([1, 0, -1], ([0, 0, 1], [0, 2, 1])), shape=(2, 5))
    scipy.sparse.save_npz(npz_path, matrix)
    write_npz_metadata(
        npz_path,
        {
            "bam2tensor_version": "2.5",
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
    assert "v2.5" in result.output
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
    shutil.copy(TEST_BAM, tmp_path / "test.bam")
    shutil.copy(TEST_BAM_INDEX, tmp_path / "test.bam.bai")

    runner = CliRunner()
    # Run extraction
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
    assert result.exit_code == 0

    # Inspect the output
    npz_path = str(tmp_path / "out" / "test.methylation.npz")
    result = runner.invoke(inspect_main, [npz_path])
    assert result.exit_code == 0
    assert "test" in result.output  # genome_name
    assert "CpG index CRC32:" in result.output
    assert f"v{__version__}" in result.output


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


def test_inspect_exits_nonzero_for_an_unreadable_file(tmp_path) -> None:
    """A corrupt file is reported and makes the command fail.

    Regression: the error was printed but the exit status stayed 0, so a
    pipeline validating its outputs with this command saw success.
    """
    corrupt = tmp_path / "corrupt.methylation.npz"
    corrupt.write_bytes(b"this is not a zip archive")

    result = CliRunner().invoke(inspect_main, [str(corrupt)])
    assert result.exit_code == 1
    assert "corrupt.methylation.npz" in result.output
    assert "Error:" in result.output


def test_inspect_reports_good_files_alongside_a_bad_one(tmp_path) -> None:
    """One unreadable file does not stop the others; the status is still 1."""
    good = str(tmp_path / "good.methylation.npz")
    scipy.sparse.save_npz(
        good, scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(1, 9))
    )
    truncated = tmp_path / "truncated.methylation.npz"
    with open(good, "rb") as source:
        truncated.write_bytes(source.read()[:40])
    also_good = str(tmp_path / "also_good.methylation.npz")
    shutil.copy(good, also_good)

    result = CliRunner().invoke(inspect_main, [good, str(truncated), also_good])
    assert result.exit_code == 1
    assert "good.methylation.npz" in result.output
    assert "also_good.methylation.npz" in result.output
    assert result.output.count("CpG sites:       9") == 2
    assert "truncated.methylation.npz" in result.output


def test_inspect_shows_the_call_filters(tmp_path) -> None:
    """The settings a file was made with are part of its summary."""
    npz_path = str(tmp_path / "sample.methylation.npz")
    scipy.sparse.save_npz(
        npz_path, scipy.sparse.coo_matrix(([1], ([0], [0])), shape=(1, 9))
    )
    write_npz_metadata(
        npz_path,
        {
            "filters": {
                "min_mapping_quality": 30,
                "min_base_quality": 20,
                "mask_3prime": 10,
                "keep_improper_pairs": False,
                "non_converted_reads": {"enabled": False, "threshold": 3},
                "em_overconversion": {"enabled": False, "min_cpgs": 3},
            }
        },
    )
    result = CliRunner().invoke(inspect_main, [npz_path])
    assert result.exit_code == 0
    assert "Filters:         MAPQ >= 30" in result.output
    assert "base quality >= 20" in result.output
    assert "improper pairs dropped" in result.output
    assert "3' end mask (10 bp)" in result.output
