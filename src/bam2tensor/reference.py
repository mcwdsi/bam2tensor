"""Reference genome download and caching utilities.

Provides functionality to download and cache reference genome FASTA files
from well-known sources (UCSC, NCBI) for common genomes like hg38, hg19,
mm10, and T2T-CHM13. This avoids the need for users to manually locate
and download reference FASTAs.

Example:
    >>> # xdoctest: +SKIP
    >>> from bam2tensor.reference import download_reference, list_available_genomes
    >>>
    >>> # See what genomes are available
    >>> genomes = list_available_genomes()
    >>> for name, info in genomes.items():
    ...     print(f"{name}: {info['description']}")
    >>>
    >>> # Download hg38
    >>> fasta_path = download_reference("hg38", verbose=True)
    >>> print(f"Reference saved to: {fasta_path}")
"""

import gzip
import os
import shutil
import urllib.request
from pathlib import Path

from tqdm import tqdm

KNOWN_GENOMES: dict[str, dict[str, str]] = {
    "hg38": {
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
        "description": "Human GRCh38/hg38 (UCSC)",
        "expected_chromosomes": ",".join(
            ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
        ),
    },
    "hg19": {
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
        "description": "Human GRCh37/hg19 (UCSC)",
        "expected_chromosomes": ",".join(
            ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
        ),
    },
    "mm10": {
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
        "description": "Mouse GRCm38/mm10 (UCSC)",
        "expected_chromosomes": ",".join(
            ["chr" + str(i) for i in range(1, 20)] + ["chrX", "chrY"]
        ),
    },
    "T2T-CHM13": {
        "url": "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz",
        "description": "T2T-CHM13v2.0 (Telomere-to-Telomere)",
        "expected_chromosomes": ",".join(
            ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
        ),
    },
}


class _DownloadProgressBar(tqdm):  # type: ignore
    """Progress bar for urllib downloads."""

    def update_to(
        self, block_num: int = 1, block_size: int = 1, total_size: int | None = None
    ) -> None:
        """Update progress bar with download progress.

        Called by urllib.request.urlretrieve as a reporthook callback.

        Args:
            block_num: Number of blocks transferred so far.
            block_size: Size of each block in bytes.
            total_size: Total size of the file in bytes, or None if unknown.
        """
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def get_cache_dir() -> Path:
    """Get the bam2tensor cache directory for downloaded references.

    Returns the platform-appropriate cache directory, creating it if it
    does not already exist. Uses ``~/.cache/bam2tensor`` by default, or
    respects the ``XDG_CACHE_HOME`` environment variable on Linux/macOS.

    Returns:
        Path to the bam2tensor cache directory. The directory is created
        if it does not exist.

    Example:
        >>> # xdoctest: +SKIP
        >>> cache_dir = get_cache_dir()
        >>> print(cache_dir)
        /home/user/.cache/bam2tensor
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        base = Path(xdg_cache)
    else:
        base = Path.home() / ".cache"
    cache_dir = base / "bam2tensor"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_reference(genome_name: str) -> Path | None:
    """Check if a reference genome FASTA is already cached locally.

    Looks for a previously downloaded and decompressed FASTA file in the
    bam2tensor cache directory.

    Args:
        genome_name: The genome identifier (e.g., "hg38", "mm10").
            Must be a key in KNOWN_GENOMES.

    Returns:
        Path to the cached FASTA file if it exists, or None if the
        genome has not been downloaded yet.

    Example:
        >>> # xdoctest: +SKIP
        >>> path = get_cached_reference("hg38")
        >>> if path:
        ...     print(f"Found cached reference: {path}")
        ... else:
        ...     print("Not cached yet")
    """
    cache_dir = get_cache_dir()
    fasta_path = cache_dir / genome_name / f"{genome_name}.fa"
    if fasta_path.exists():
        return fasta_path
    return None


def download_reference(genome_name: str, verbose: bool = False) -> Path:
    """Download and cache a reference genome FASTA file.

    Downloads a gzipped FASTA file from a well-known source (UCSC or
    NCBI), decompresses it, and stores it in the bam2tensor cache
    directory. If the genome has already been downloaded, returns the
    path to the cached file without re-downloading.

    Args:
        genome_name: The genome identifier to download. Must be one of
            the keys in KNOWN_GENOMES (e.g., "hg38", "hg19", "mm10",
            "T2T-CHM13").
        verbose: If True, display download progress and status messages.

    Returns:
        Path to the decompressed reference FASTA file in the cache
        directory.

    Raises:
        ValueError: If genome_name is not a recognized genome in
            KNOWN_GENOMES.
        OSError: If the download fails due to network issues or
            insufficient disk space.

    Example:
        >>> # xdoctest: +SKIP
        >>> fasta_path = download_reference("hg38", verbose=True)
        Downloading Human GRCh38/hg38 (UCSC)...
        >>> print(fasta_path)
        /home/user/.cache/bam2tensor/hg38/hg38.fa
    """
    if genome_name not in KNOWN_GENOMES:
        available = ", ".join(sorted(KNOWN_GENOMES.keys()))
        raise ValueError(
            f"Unknown genome '{genome_name}'. " f"Available genomes: {available}"
        )

    cached = get_cached_reference(genome_name)
    if cached is not None:
        if verbose:
            print(f"Using cached reference genome: {cached}")
        return cached

    genome_info = KNOWN_GENOMES[genome_name]
    cache_dir = get_cache_dir() / genome_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    gz_path = cache_dir / f"{genome_name}.fa.gz"
    fasta_path = cache_dir / f"{genome_name}.fa"

    url = genome_info["url"]
    description = genome_info["description"]

    print(f"Downloading {description} from:\n  {url}")
    print(f"Saving to: {cache_dir}")
    print("This may take a while for large genomes (~1 GB compressed)...")

    if verbose:
        with _DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=genome_name
        ) as progress:
            urllib.request.urlretrieve(  # noqa: S310
                url, str(gz_path), reporthook=progress.update_to
            )
    else:
        urllib.request.urlretrieve(url, str(gz_path))  # noqa: S310

    print("Download complete. Decompressing...")

    with gzip.open(gz_path, "rb") as f_in:
        with open(fasta_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove the compressed file to save disk space
    gz_path.unlink()

    # Validate the downloaded file looks like a FASTA
    with open(fasta_path) as f:
        first_line = f.readline()
        if not first_line.startswith(">"):
            fasta_path.unlink()
            raise OSError(
                f"Downloaded file does not appear to be a valid FASTA "
                f"(first line: {first_line[:50]!r})"
            )

    print(f"Reference genome ready: {fasta_path}")
    return fasta_path


def list_available_genomes() -> dict[str, dict[str, str]]:
    """List all known genome references available for download.

    Returns the dictionary of genomes that can be downloaded using
    download_reference(), along with their descriptions and URLs.

    Returns:
        A dictionary mapping genome names to their metadata. Each value
        is a dict with keys 'url', 'description', and
        'expected_chromosomes'.

    Example:
        >>> genomes = list_available_genomes()
        >>> "hg38" in genomes
        True
        >>> "description" in genomes["hg38"]
        True
    """
    return KNOWN_GENOMES
