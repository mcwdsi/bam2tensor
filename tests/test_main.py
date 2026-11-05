"""Test cases for the __main__ module."""

import os
import shutil

import numpy as np
import pysam
import pytest
import scipy.sparse
from click.testing import CliRunner

from bam2tensor import __main__
from bam2tensor.__main__ import get_output_path
from bam2tensor.functions import ExtractionResult
from bam2tensor.metadata import read_npz_metadata, read_npz_tlen
from tests.paths import (
    SARSCOV2_BISMARK_BAM,
    SARSCOV2_BWAMETH_BAM,
    TESTS_DIR,
    TEST_BAM,
    TEST_BAM_INDEX,
    TEST_FASTA,
)


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

    assert __main__.get_input_bams(TEST_BAM) == [TEST_BAM]
    # A directory is searched recursively: the bundled fixtures in tests/
    # and tests/data/.
    assert sorted(__main__.get_input_bams(str(TESTS_DIR))) == sorted(
        [TEST_BAM, SARSCOV2_BISMARK_BAM, SARSCOV2_BWAMETH_BAM]
    )


def test_get_input_bams_recurses_into_subdirectories(tmp_path) -> None:
    """Directories are searched recursively and only *.bam files are returned."""
    (tmp_path / "nested" / "deeper").mkdir(parents=True)
    (tmp_path / "a.bam").touch()
    (tmp_path / "a.bam.bai").touch()
    (tmp_path / "nested" / "b.bam").touch()
    (tmp_path / "nested" / "deeper" / "c.bam").touch()
    (tmp_path / "nested" / "notes.txt").touch()

    found = __main__.get_input_bams(str(tmp_path))
    assert sorted(found) == sorted(
        [
            str(tmp_path / "a.bam"),
            str(tmp_path / "nested" / "b.bam"),
            str(tmp_path / "nested" / "deeper" / "c.bam"),
        ]
    )


def test_get_input_bams_invalid_path() -> None:
    """Test get_input_bams raises ValueError for invalid path."""
    with pytest.raises(ValueError, match="is not a file or a directory"):
        __main__.get_input_bams("/nonexistent/path/to/nowhere")


def test_validate_input_output() -> None:
    """Test validate_input_output."""

    __main__.validate_input_output([TEST_BAM], overwrite=True)


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


def test_main(runner: CliRunner, tmp_path) -> None:
    """Test main() call, which runs everything, with some accepted defaults."""

    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            TEST_BAM,
            "--overwrite",
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    print(result.output)
    assert result.exit_code == 0, f"Failed with: {result.output}"
    assert (tmp_path / "out" / "test.methylation.npz").exists()
    # The CpG index cache is written to the working directory.
    assert (tmp_path / "test.cache.json.gz").exists()


def test_main_output_defaults_to_input_directory(runner: CliRunner, tmp_path) -> None:
    """Without --output-dir the .npz is written next to the input BAM."""
    shutil.copy(TEST_BAM, tmp_path / "sample.bam")
    shutil.copy(TEST_BAM_INDEX, tmp_path / "sample.bam.bai")

    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            str(tmp_path / "sample.bam"),
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
        ],
    )

    assert result.exit_code == 0, f"Failed with: {result.output}"
    assert (tmp_path / "sample.methylation.npz").exists()


def test_main_skip_existing(runner: CliRunner, tmp_path) -> None:
    """Test main skips BAMs when output exists and --overwrite not set."""
    # Copy test files to tmp_path
    shutil.copy(TEST_BAM, tmp_path / "test.bam")
    shutil.copy(TEST_BAM_INDEX, tmp_path / "test.bam.bai")

    # Create an existing output file
    (tmp_path / "test.methylation.npz").touch()

    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            str(tmp_path / "test.bam"),
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            # Note: no --overwrite flag
        ],
    )

    assert result.exit_code == 0
    assert "Skipped" in result.output or "skipped" in result.output.lower()


def test_main_missing_bam_index(runner: CliRunner, tmp_path) -> None:
    """Test main handles missing BAM index file gracefully (reports error)."""
    # Copy BAM file but NOT the index
    shutil.copy(TEST_BAM, tmp_path / "no_index.bam")
    # Don't copy the .bai file

    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            str(tmp_path / "no_index.bam"),
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            "--overwrite",
        ],
    )

    # The failure is reported in the per-BAM section and in the summary, and
    # the exit status is non-zero so schedulers/pipelines see the failure.
    assert result.exit_code == 1
    assert "ERROR: Index missing for bam file" in result.output
    assert "Errors:        1" in result.output
    assert "Processed:     0 BAM(s)" in result.output


def test_get_output_path_default() -> None:
    """get_output_path returns file next to input when no output_dir."""
    result = get_output_path("/data/sample.bam")
    assert result == "/data/sample.methylation.npz"


def test_get_output_path_with_output_dir() -> None:
    """get_output_path places file in output_dir when specified."""
    result = get_output_path("/data/sample.bam", "/output")
    assert result == "/output/sample.methylation.npz"


def test_get_output_path_preserves_basename() -> None:
    """get_output_path uses only basename, not full input path."""
    result = get_output_path("/a/b/c/deep/sample.bam", "/out")
    assert result == "/out/sample.methylation.npz"


def test_validate_input_output_with_output_dir(tmp_path) -> None:
    """validate_input_output works with output_dir specified."""
    bam_file = tmp_path / "test.bam"
    bam_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    __main__.validate_input_output(
        [str(bam_file)], overwrite=False, output_dir=str(output_dir)
    )


def test_main_with_output_dir(runner: CliRunner, tmp_path) -> None:
    """Test end-to-end CLI with --output-dir flag."""
    # Copy test files to tmp_path
    shutil.copy(TEST_BAM, tmp_path / "test.bam")
    shutil.copy(TEST_BAM_INDEX, tmp_path / "test.bam.bai")

    output_dir = tmp_path / "output"

    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            str(tmp_path / "test.bam"),
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ],
    )

    print(result.output)
    assert result.exit_code == 0, f"Failed with: {result.output}"
    assert (output_dir / "test.methylation.npz").exists()


def test_main_output_dir_created(runner: CliRunner, tmp_path) -> None:
    """Test that --output-dir creates the directory if it doesn't exist."""
    shutil.copy(TEST_BAM, tmp_path / "test.bam")
    shutil.copy(TEST_BAM_INDEX, tmp_path / "test.bam.bai")

    output_dir = tmp_path / "nonexistent" / "nested" / "dir"

    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            str(tmp_path / "test.bam"),
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, f"Failed with: {result.output}"
    assert output_dir.exists()
    assert (output_dir / "test.methylation.npz").exists()


def test_main_list_genomes(runner: CliRunner) -> None:
    """Test --list-genomes prints available genomes."""
    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            TEST_BAM,
            "--list-genomes",
        ],
    )

    # sys.exit(0) causes SystemExit which Click catches
    assert result.exit_code == 0
    assert "hg38" in result.output
    assert "hg19" in result.output
    assert "mm10" in result.output


def test_main_missing_reference_fasta(runner: CliRunner) -> None:
    """Test that missing both --reference-fasta and --download-reference gives error."""
    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            TEST_BAM,
            "--genome-name",
            "test",
        ],
    )

    assert result.exit_code != 0
    assert "reference-fasta" in result.output or "download-reference" in result.output


def test_format_elapsed_minutes() -> None:
    """Test _format_elapsed with durations over 60 seconds."""
    result = __main__._format_elapsed(125.5)
    assert "2 min" in result
    assert "5.50 sec" in result


def test_main_missing_genome_name(runner: CliRunner) -> None:
    """Test that missing --genome-name without --download-reference gives error."""
    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            TEST_BAM,
            "--reference-fasta",
            TEST_FASTA,
        ],
    )

    assert result.exit_code != 0
    assert "genome-name" in result.output


def test_main_default_expected_chromosomes(runner: CliRunner, tmp_path) -> None:
    """Test that omitting --expected-chromosomes uses hg38 defaults (24 chroms)."""
    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            TEST_BAM,
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test_default_chroms",
            "--skip-cache",
            "--overwrite",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    # The default hg38 chromosomes (24) are displayed in the config section
    # before the embedding is loaded, even though the test FASTA only has 3 chroms.
    assert "24 (" in result.output
    assert "chrY" in result.output

    # The test FASTA lacks chr4..chrY, which must be reported as a clean
    # usage error naming the missing chromosomes — not as a KeyError traceback.
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Error: Expected chromosome(s) not found in reference FASTA" in result.output
    assert "chr4" in result.output
    assert not (tmp_path / "out" / "test.methylation.npz").exists()


def test_main_few_chromosomes_display(runner: CliRunner, tmp_path) -> None:
    """Test that <= 4 chromosomes are displayed directly (not summarized)."""
    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            TEST_BAM,
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test_few_chroms",
            "--expected-chromosomes",
            "chr1,chr2",
            "--skip-cache",
            "--overwrite",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    # With <= 4 chroms, they're shown directly as the comma-separated string
    assert "chr1,chr2" in result.output


def test_main_download_reference(runner: CliRunner, tmp_path) -> None:
    """Test --download-reference flag triggers download and sets defaults."""
    from unittest.mock import patch

    # Mock download_reference to return a fake FASTA path
    fake_fasta = tmp_path / "hg38.fa"
    # Create a FASTA with chr1 so the embedding can be built
    fake_fasta.write_text(">chr1\n" + "ACGTCGACGT" * 15 + "\n")

    with patch("bam2tensor.__main__.download_reference_fn", return_value=fake_fasta):
        result = runner.invoke(
            __main__.main,
            [
                "--input-path",
                TEST_BAM,
                "--download-reference",
                "hg38",
                "--expected-chromosomes",
                "chr1",
                "--skip-cache",
                "--overwrite",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

    # --download-reference sets genome_name to "hg38" automatically
    assert result.exit_code == 0, f"Failed with: {result.output}"
    assert "hg38" in result.output


def test_main_download_reference_with_genome_name(runner: CliRunner, tmp_path) -> None:
    """Test --download-reference with explicit --genome-name and --expected-chromosomes."""
    from unittest.mock import patch

    fake_fasta = tmp_path / "hg38.fa"
    fake_fasta.write_text(">chr1\n" + "ACGTCGACGT" * 15 + "\n")

    with patch("bam2tensor.__main__.download_reference_fn", return_value=fake_fasta):
        result = runner.invoke(
            __main__.main,
            [
                "--input-path",
                TEST_BAM,
                "--download-reference",
                "hg38",
                "--genome-name",
                "custom_name",
                "--expected-chromosomes",
                "chr1",
                "--skip-cache",
                "--overwrite",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0
    assert "custom_name" in result.output


def test_main_verbose_flag(runner: CliRunner, tmp_path) -> None:
    """Test --verbose flag enables verbose output."""
    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            TEST_BAM,
            "--overwrite",
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            "--verbose",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert "Total reads" in result.output


def test_validate_input_output_unwritable_dir(tmp_path) -> None:
    """Test validate_input_output raises ValueError for unwritable output dir."""
    bam_file = tmp_path / "test.bam"
    bam_file.touch()

    unwritable_dir = tmp_path / "locked"
    unwritable_dir.mkdir()
    os.chmod(unwritable_dir, 0o444)

    try:
        with pytest.raises(ValueError, match="Output file path is not writable"):
            __main__.validate_input_output(
                [str(bam_file)], overwrite=False, output_dir=str(unwritable_dir)
            )
    finally:
        os.chmod(unwritable_dir, 0o755)


def test_main_writes_tlen(runner: CliRunner, tmp_path) -> None:
    """Test that the CLI writes TLEN data to the output .npz file."""
    shutil.copy(TEST_BAM, tmp_path / "test.bam")
    shutil.copy(TEST_BAM_INDEX, tmp_path / "test.bam.bai")

    output_dir = tmp_path / "output"

    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            str(tmp_path / "test.bam"),
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, f"Failed with: {result.output}"

    npz_path = str(output_dir / "test.methylation.npz")
    tlen = read_npz_tlen(npz_path)
    assert tlen is not None
    assert len(tlen) >= 1  # At least one read


# ======================================================================
# Output validation
# ======================================================================


def test_validate_input_output_existing_unwritable_with_overwrite(tmp_path) -> None:
    """--overwrite onto a read-only output fails up front, not after extraction.

    Regression: this case fell through a no-op branch, so the permission
    error only surfaced when the result was saved — after the (potentially
    hours-long) extraction had already run.
    """
    bam_file = tmp_path / "test.bam"
    bam_file.touch()
    npz_file = tmp_path / "test.methylation.npz"
    npz_file.touch()
    os.chmod(npz_file, 0o444)

    try:
        with pytest.raises(ValueError, match="Output file exists and is not writable"):
            __main__.validate_input_output([str(bam_file)], overwrite=True)

        # Without --overwrite the BAM is skipped, so a read-only output is fine.
        __main__.validate_input_output([str(bam_file)], overwrite=False)
    finally:
        os.chmod(npz_file, 0o644)


def test_validate_input_output_existing_writable_with_overwrite(tmp_path) -> None:
    """--overwrite onto a writable existing output passes validation."""
    bam_file = tmp_path / "test.bam"
    bam_file.touch()
    (tmp_path / "test.methylation.npz").touch()

    __main__.validate_input_output([str(bam_file)], overwrite=True)


def test_validate_input_output_existing_output_in_unwritable_dir(tmp_path) -> None:
    """Overwriting needs a writable directory, not just a writable file.

    The replacement is written beside the existing output and renamed over
    it, so that a failed overwrite cannot destroy the file being replaced.
    """
    bam_file = tmp_path / "test.bam"
    bam_file.touch()
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "test.methylation.npz").touch()
    os.chmod(locked, 0o555)

    try:
        with pytest.raises(ValueError, match="Output file path is not writable"):
            __main__.validate_input_output(
                [str(bam_file)], overwrite=True, output_dir=str(locked)
            )
        # Without --overwrite the existing output means the BAM is skipped.
        __main__.validate_input_output(
            [str(bam_file)], overwrite=False, output_dir=str(locked)
        )
    finally:
        os.chmod(locked, 0o755)


# ======================================================================
# Output files are written atomically
# ======================================================================


def _small_result():
    """A tiny ExtractionResult to write."""
    matrix = scipy.sparse.coo_matrix(
        (np.array([1, 0, -1]), (np.array([0, 0, 1]), np.array([0, 2, 1]))),
        shape=(2, 5),
    )
    return ExtractionResult(matrix=matrix, tlen=np.array([150, 0], dtype=np.int32))


def test_write_output_writes_matrix_tlen_and_metadata(tmp_path) -> None:
    """A successful write holds all three parts and leaves no temporary file."""
    output_file = str(tmp_path / "sample.methylation.npz")
    __main__.write_output(output_file, _small_result(), {"genome_name": "tiny"})

    assert os.listdir(tmp_path) == ["sample.methylation.npz"]
    loaded = scipy.sparse.load_npz(output_file)
    assert loaded.shape == (2, 5)
    assert loaded.toarray().tolist() == [[1, 0, 0, 0, 0], [0, -1, 0, 0, 0]]
    tlen = read_npz_tlen(output_file)
    assert tlen is not None
    assert tlen.tolist() == [150, 0]
    assert read_npz_metadata(output_file) == {"genome_name": "tiny"}


@pytest.mark.parametrize("failing_step", ["write_npz_tlen", "write_npz_metadata"])
def test_failed_write_leaves_no_output_file(
    tmp_path, monkeypatch, failing_step
) -> None:
    """A write that fails part-way leaves nothing behind.

    Regression: the matrix was saved to its final name first and the
    fragment lengths and metadata appended afterwards. A failure — or an
    interruption — in between left a file that loaded fine but was
    incomplete, and the next run skipped it as already done.
    """

    def fail(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(__main__, failing_step, fail)
    output_file = str(tmp_path / "sample.methylation.npz")

    with pytest.raises(OSError, match="No space left on device"):
        __main__.write_output(output_file, _small_result(), {"genome_name": "tiny"})

    assert os.listdir(tmp_path) == []


def test_failed_overwrite_keeps_the_existing_output(tmp_path, monkeypatch) -> None:
    """A failed overwrite leaves the previous file byte-for-byte intact."""
    output_file = str(tmp_path / "sample.methylation.npz")
    __main__.write_output(output_file, _small_result(), {"run": "first"})
    with open(output_file, "rb") as f:
        before = f.read()

    def fail(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(__main__, "write_npz_metadata", fail)
    with pytest.raises(KeyboardInterrupt):
        __main__.write_output(output_file, _small_result(), {"run": "second"})

    with open(output_file, "rb") as f:
        assert f.read() == before
    assert os.listdir(tmp_path) == ["sample.methylation.npz"]
    assert read_npz_metadata(output_file) == {"run": "first"}


def test_main_interrupted_save_is_redone_on_the_next_run(
    runner: CliRunner, tmp_path, monkeypatch
) -> None:
    """After a failed save, a plain re-run (no --overwrite) processes the BAM.

    Nothing was left behind for the "output already exists" check to find.
    """
    shutil.copy(TEST_BAM, tmp_path / "sample.bam")
    shutil.copy(TEST_BAM_INDEX, tmp_path / "sample.bam.bai")
    args = [
        "--input-path",
        str(tmp_path / "sample.bam"),
        "--reference-fasta",
        TEST_FASTA,
        "--genome-name",
        "test",
        "--expected-chromosomes",
        "chr1,chr2,chr3",
    ]
    real_write_metadata = __main__.write_npz_metadata

    def fail(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(__main__, "write_npz_metadata", fail)
    failed = runner.invoke(__main__.main, args)
    assert failed.exit_code != 0
    assert isinstance(failed.exception, OSError)
    assert not (tmp_path / "sample.methylation.npz").exists()
    assert sorted(p.name for p in tmp_path.glob("sample*")) == [
        "sample.bam",
        "sample.bam.bai",
    ]

    monkeypatch.setattr(__main__, "write_npz_metadata", real_write_metadata)
    retried = runner.invoke(__main__.main, args)
    assert retried.exit_code == 0, retried.output
    assert "Skipped" not in retried.output
    assert read_npz_metadata(str(tmp_path / "sample.methylation.npz")) is not None


# ======================================================================
# Per-BAM failures in a batch
# ======================================================================


def _write_ensembl_named_bam(path) -> None:
    """Write an indexed BAM whose only contig is named ``1`` (no chr prefix)."""
    seq = "ACGTCGACGT" * 15
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": len(seq), "SN": "1"}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = seq
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, len(seq))]
        a.set_tag("MD", str(len(seq)))
        a.set_tag("YD", "f")
        out_bam.write(a)
    pysam.index(str(path))


def test_main_continues_past_failing_bams(runner: CliRunner, tmp_path) -> None:
    """One unusable BAM is reported and skipped; the rest of the batch still runs.

    Regression: only a missing index was handled. A BAM whose chromosome
    names did not match the reference raised out of the loop, so a single bad
    file aborted the whole directory run with a traceback.
    """
    bam_dir = tmp_path / "bams"
    bam_dir.mkdir()
    shutil.copy(TEST_BAM, bam_dir / "good.bam")
    shutil.copy(TEST_BAM_INDEX, bam_dir / "good.bam.bai")
    shutil.copy(TEST_BAM, bam_dir / "no_index.bam")  # index deliberately absent
    _write_ensembl_named_bam(bam_dir / "wrong_names.bam")
    output_dir = tmp_path / "out"

    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            str(bam_dir),
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            "--output-dir",
            str(output_dir),
        ],
    )

    # Each failure is reported with its cause, not as a traceback.
    assert isinstance(result.exception, SystemExit)
    assert "ERROR: Index missing for bam file" in result.output
    assert "ERROR: Chromosome name mismatch" in result.output

    # The healthy BAM was still processed and written.
    assert (output_dir / "good.methylation.npz").exists()
    assert not (output_dir / "no_index.methylation.npz").exists()
    assert not (output_dir / "wrong_names.methylation.npz").exists()

    # The summary accounts for all three, and the exit status reports failure.
    assert "Found 3 BAM file(s)" in result.output
    assert "Processed:     1 BAM(s)" in result.output
    assert "Errors:        2" in result.output
    assert result.exit_code == 1


def test_main_exit_status_zero_when_all_bams_succeed(
    runner: CliRunner, tmp_path
) -> None:
    """A clean batch (including skipped existing outputs) exits 0."""
    bam_dir = tmp_path / "bams"
    bam_dir.mkdir()
    for name in ("a", "b"):
        shutil.copy(TEST_BAM, bam_dir / f"{name}.bam")
        shutil.copy(TEST_BAM_INDEX, bam_dir / f"{name}.bam.bai")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "b.methylation.npz").touch()  # pre-existing: will be skipped

    result = runner.invoke(
        __main__.main,
        [
            "--input-path",
            str(bam_dir),
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "test",
            "--expected-chromosomes",
            "chr1,chr2,chr3",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Processed:     1 BAM(s)" in result.output
    assert "Skipped:       1" in result.output
    assert "Errors" not in result.output


# ======================================================================
# Configuration errors are reported cleanly
# ======================================================================


def test_main_cache_built_for_other_chromosomes_is_clean_error(
    runner: CliRunner, tmp_path
) -> None:
    """Reusing a genome name with a different chromosome list explains itself."""
    common = [
        "--input-path",
        TEST_BAM,
        "--reference-fasta",
        TEST_FASTA,
        "--genome-name",
        "shared_name",
        "--output-dir",
        str(tmp_path / "out"),
        "--overwrite",
    ]
    first = runner.invoke(
        __main__.main, common + ["--expected-chromosomes", "chr1,chr2,chr3"]
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(__main__.main, common + ["--expected-chromosomes", "chr1"])

    assert second.exit_code == 1
    assert isinstance(second.exception, SystemExit)  # not an uncaught ValueError
    assert "Error: Expected chromosomes do not match the cache" in second.output
    assert "--skip-cache" in second.output

    # Following the advice works.
    third = runner.invoke(
        __main__.main, common + ["--expected-chromosomes", "chr1", "--skip-cache"]
    )
    assert third.exit_code == 0, third.output
