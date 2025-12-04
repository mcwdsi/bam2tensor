"""Test cases for the __main__ module."""

import os
import pytest
from click.testing import CliRunner

from bam2tensor import __main__


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_main_succeeds(runner: CliRunner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(__main__.main, ["--help"])
    assert result.exit_code == 0


def test_get_input_bams() -> None:
    """Test get_input_bams."""

    assert __main__.get_input_bams("tests/test.bam") == ["tests/test.bam"]
    assert __main__.get_input_bams("tests/") == ["tests/test.bam"]


def test_get_input_bams_invalid_path() -> None:
    """Test get_input_bams raises ValueError for invalid path."""
    with pytest.raises(ValueError, match="is not a file or a directory"):
        __main__.get_input_bams("/nonexistent/path/to/nowhere")


def test_validate_input_output() -> None:
    """Test validate_input_output."""

    __main__.validate_input_output(["tests/test.bam"], overwrite=True)


def test_validate_input_output_unreadable(tmp_path) -> None:
    """Test validate_input_output raises ValueError for unreadable file."""
    # Create a file that we'll make unreadable
    unreadable_file = tmp_path / "unreadable.bam"
    unreadable_file.touch()
    os.chmod(unreadable_file, 0o000)

    try:
        with pytest.raises(ValueError, match="Input file is not readable"):
            __main__.validate_input_output([str(unreadable_file)], overwrite=False)
    finally:
        # Restore permissions for cleanup
        os.chmod(unreadable_file, 0o644)


def test_validate_input_output_existing_no_overwrite(tmp_path) -> None:
    """Test validate_input_output when output exists but no overwrite flag."""
    # Create a .bam file and its corresponding .methylation.npz
    bam_file = tmp_path / "test.bam"
    bam_file.touch()
    npz_file = tmp_path / "test.methylation.npz"
    npz_file.touch()

    # Should not raise, just silently pass (output exists but no overwrite message)
    __main__.validate_input_output([str(bam_file)], overwrite=False)


def test_main(runner: CliRunner) -> None:
    """Test main() call, which runs everything, with some accepted defaults."""

    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            "tests/test.bam",
            "--overwrite",
            "--reference-fasta",
            "tests/test_fasta.fa",
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
        ],
    )

    print(result.output)
    assert result.exit_code == 0, f"Failed with: {result.output}"


def test_main_skip_existing(runner: CliRunner, tmp_path) -> None:
    """Test main skips BAMs when output exists and --overwrite not set."""
    import shutil

    # Copy test files to tmp_path
    shutil.copy("tests/test.bam", tmp_path / "test.bam")
    shutil.copy("tests/test.bam.bai", tmp_path / "test.bam.bai")

    # Create an existing output file
    (tmp_path / "test.methylation.npz").touch()

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
            # Note: no --overwrite flag
        ],
    )

    assert result.exit_code == 0
    assert "Skipping this .bam" in result.output or "skipped" in result.output.lower()


def test_main_missing_bam_index(runner: CliRunner, tmp_path) -> None:
    """Test main handles missing BAM index file gracefully (reports error)."""
    import shutil

    # Copy BAM file but NOT the index
    shutil.copy("tests/test.bam", tmp_path / "no_index.bam")
    # Don't copy the .bai file

    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            str(tmp_path / "no_index.bam"),
            "--reference-fasta",
            "tests/test_fasta.fa",
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            "--overwrite",
        ],
    )

    # The main function catches FileNotFoundError and reports it
    assert result.exit_code == 0  # Should complete (with errors logged)
    assert "Error" in result.output or "error" in result.output.lower()
    assert "1 errors occurred" in result.output or "had errors" in result.output
