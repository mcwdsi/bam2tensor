"""Test cases for the __main__ module."""

import pytest
from click.testing import CliRunner

from bam2tensor import __main__


@pytest.fixture()
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


def test_validate_input_output() -> None:
    """Test validate_input_output."""

    __main__.validate_input_output(["tests/test.bam"], overwrite=True)


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
