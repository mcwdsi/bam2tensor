"""Test cases for the reference module."""

import gzip
import os
from unittest.mock import patch

import pytest

from bam2tensor.reference import (
    KNOWN_GENOMES,
    _DownloadProgressBar,
    download_reference,
    get_cache_dir,
    get_cached_reference,
    list_available_genomes,
)


def test_known_genomes_structure():
    """All entries in KNOWN_GENOMES have required keys."""
    for name, info in KNOWN_GENOMES.items():
        assert "url" in info, f"{name} missing 'url'"
        assert "description" in info, f"{name} missing 'description'"
        assert "expected_chromosomes" in info, f"{name} missing 'expected_chromosomes'"
        assert info["url"].startswith("http"), f"{name} URL doesn't start with http"


def test_known_genomes_has_common_genomes():
    """KNOWN_GENOMES contains the expected common genome references."""
    assert "hg38" in KNOWN_GENOMES
    assert "hg19" in KNOWN_GENOMES
    assert "mm10" in KNOWN_GENOMES
    assert "T2T-CHM13" in KNOWN_GENOMES


def test_known_genomes_chromosomes():
    """Verify expected chromosomes are reasonable for each genome."""
    # hg38/hg19 should have chr1-chr22, chrX, chrY
    for name in ["hg38", "hg19"]:
        chroms = KNOWN_GENOMES[name]["expected_chromosomes"].split(",")
        assert len(chroms) == 24
        assert "chr1" in chroms
        assert "chrX" in chroms
        assert "chrY" in chroms

    # mm10 should have chr1-chr19, chrX, chrY
    mm10_chroms = KNOWN_GENOMES["mm10"]["expected_chromosomes"].split(",")
    assert len(mm10_chroms) == 21
    assert "chr1" in mm10_chroms
    assert "chr19" in mm10_chroms
    assert "chr20" not in mm10_chroms


def test_get_cache_dir(tmp_path):
    """get_cache_dir returns a valid path and creates it."""
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
        cache_dir = get_cache_dir()
        assert cache_dir == tmp_path / "bam2tensor"
        assert cache_dir.exists()


def test_get_cache_dir_default(tmp_path):
    """get_cache_dir uses ~/.cache when XDG_CACHE_HOME is not set."""
    env = os.environ.copy()
    env.pop("XDG_CACHE_HOME", None)
    with patch.dict(os.environ, env, clear=True):
        cache_dir = get_cache_dir()
        assert "bam2tensor" in str(cache_dir)


def test_get_cached_reference_missing(tmp_path):
    """get_cached_reference returns None for uncached genomes."""
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
        result = get_cached_reference("hg38")
        assert result is None


def test_get_cached_reference_exists(tmp_path):
    """get_cached_reference returns path when genome is cached."""
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
        # Create a fake cached FASTA
        genome_dir = tmp_path / "bam2tensor" / "hg38"
        genome_dir.mkdir(parents=True)
        fasta = genome_dir / "hg38.fa"
        fasta.write_text(">chr1\nACGT\n")

        result = get_cached_reference("hg38")
        assert result is not None
        assert result == fasta


def test_list_available_genomes():
    """list_available_genomes returns the KNOWN_GENOMES dict."""
    genomes = list_available_genomes()
    assert genomes is KNOWN_GENOMES
    assert "hg38" in genomes
    assert "description" in genomes["hg38"]


def test_download_reference_invalid_genome():
    """download_reference raises ValueError for unknown genome."""
    with pytest.raises(ValueError, match="Unknown genome"):
        download_reference("not_a_real_genome")


def test_download_reference_uses_cache(tmp_path):
    """download_reference returns cached path without downloading."""
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
        # Create a fake cached FASTA
        genome_dir = tmp_path / "bam2tensor" / "hg38"
        genome_dir.mkdir(parents=True)
        fasta = genome_dir / "hg38.fa"
        fasta.write_text(">chr1\nACGT\n")

        result = download_reference("hg38", verbose=True)
        assert result == fasta


def test_download_reference_uses_cache_non_verbose(tmp_path):
    """download_reference returns cached path without verbose output."""
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
        genome_dir = tmp_path / "bam2tensor" / "hg38"
        genome_dir.mkdir(parents=True)
        fasta = genome_dir / "hg38.fa"
        fasta.write_text(">chr1\nACGT\n")

        result = download_reference("hg38", verbose=False)
        assert result == fasta


def _create_fake_gzipped_fasta(path, content=">chr1\nACGTCGACGT\n"):
    """Helper: write a gzipped FASTA to the given path."""
    with gzip.open(path, "wt") as f:
        f.write(content)


def test_download_reference_downloads_and_decompresses(tmp_path):
    """download_reference downloads, decompresses, and validates a FASTA."""
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):

        def fake_urlretrieve(url, dest, reporthook=None):
            _create_fake_gzipped_fasta(dest)
            return dest, {}

        with patch("bam2tensor.reference.urllib.request.urlretrieve", fake_urlretrieve):
            result = download_reference("hg38", verbose=False)

        assert result.exists()
        assert result.name == "hg38.fa"
        content = result.read_text()
        assert content.startswith(">chr1")
        # Compressed file should be removed
        gz_path = result.parent / "hg38.fa.gz"
        assert not gz_path.exists()


def test_download_reference_verbose_with_progress_bar(tmp_path):
    """download_reference uses progress bar when verbose=True."""
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):

        def fake_urlretrieve(url, dest, reporthook=None):
            _create_fake_gzipped_fasta(dest)
            if reporthook:
                reporthook(1, 1024, 2048)
            return dest, {}

        with patch("bam2tensor.reference.urllib.request.urlretrieve", fake_urlretrieve):
            result = download_reference("hg38", verbose=True)

        assert result.exists()
        assert result.name == "hg38.fa"


def test_download_reference_invalid_fasta(tmp_path):
    """download_reference raises OSError when downloaded file is not valid FASTA."""
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):

        def fake_urlretrieve(url, dest, reporthook=None):
            _create_fake_gzipped_fasta(dest, content="NOT A FASTA FILE\n")
            return dest, {}

        with patch("bam2tensor.reference.urllib.request.urlretrieve", fake_urlretrieve):
            with pytest.raises(OSError, match="does not appear to be a valid FASTA"):
                download_reference("hg38", verbose=False)

        # Invalid file should be cleaned up
        fasta_path = tmp_path / "bam2tensor" / "hg38" / "hg38.fa"
        assert not fasta_path.exists()


def test_download_progress_bar_update_to():
    """_DownloadProgressBar.update_to sets total and updates progress."""
    bar = _DownloadProgressBar(total=0)
    bar.update_to(block_num=5, block_size=100, total_size=1000)
    assert bar.total == 1000
    assert bar.n == 500
    bar.close()


def test_download_progress_bar_update_to_no_total():
    """_DownloadProgressBar.update_to works when total_size is None."""
    bar = _DownloadProgressBar(total=100)
    bar.update_to(block_num=2, block_size=10, total_size=None)
    assert bar.total == 100  # unchanged
    assert bar.n == 20
    bar.close()
