"""Genome methylation embedding for CpG site indexing and coordinate conversion.

This module provides the GenomeMethylationEmbedding class, which manages the
mapping between genomic coordinates and matrix column indices for methylation
data. It parses reference genome FASTA files to identify all CpG sites and
provides efficient bidirectional lookup between:

- Genomic coordinates (chromosome, position)
- Matrix embedding indices (column numbers in the sparse output matrix)

The CpG site index is cached to disk as a gzipped JSON file, allowing fast
subsequent runs on the same reference genome.

Example:
    Create an embedding for the human genome:

    .. code-block:: python

        from bam2tensor.embedding import GenomeMethylationEmbedding

        embedding = GenomeMethylationEmbedding(
            genome_name="hg38",
            expected_chromosomes=["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"],
            fasta_source="/path/to/GRCh38.fa",
            verbose=True,
        )

        # Get total CpG count
        print(f"Total CpG sites: {embedding.total_cpg_sites:,}")

        # Convert between coordinate systems
        chrom, pos = embedding.embedding_to_genomic_position(12345)
        idx = embedding.genomic_position_to_embedding("chr1", 10525)

Attributes:
    CACHE_FILE_SUFFIX: The suffix appended to genome_name for cache files.
"""

import gzip
import json
import os
import zlib

import numpy as np

from tqdm import tqdm
from Bio import SeqIO

from bam2tensor import __version__
from bam2tensor.metadata import compute_fasta_sha256


def _format_names(names: list[str], limit: int = 6) -> str:
    """Format a list of sequence names for an error message.

    Args:
        names: Sequence (chromosome) names to display.
        limit: Maximum number of names to list before summarising.

    Returns:
        A comma-separated string; lists longer than ``limit`` are truncated
        and suffixed with the total count.

    Example:
        >>> _format_names(["chr1", "chr2"])
        'chr1, chr2'
        >>> _format_names(["1", "2", "3", "4"], limit=2)
        '1, 2, ... (4 total)'
    """
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", ... ({len(names)} total)"


def _missing_chromosomes_message(
    missing: list[str], fasta_names: list[str], fasta_source: str
) -> str:
    """Build the error message for expected chromosomes absent from a FASTA.

    Args:
        missing: Expected chromosome names that were not found.
        fasta_names: All sequence names present in the FASTA, in file order.
        fasta_source: Path to the FASTA file (for display).

    Returns:
        A message naming the missing chromosomes and the sequences that are
        available, with a hint when the cause is a ``chr`` prefix mismatch.
    """
    available = set(fasta_names)
    message = (
        f"Expected chromosome(s) not found in reference FASTA {fasta_source}: "
        f"{_format_names(missing)}. The FASTA contains {len(fasta_names)} "
        f"sequence(s): {_format_names(fasta_names)}."
    )
    if any(name.startswith("chr") and name[3:] in available for name in missing):
        message += (
            " The FASTA appears to use Ensembl-style names (no 'chr' prefix); "
            "set the expected chromosomes to match "
            "(e.g. --expected-chromosomes '1,2,3,...')."
        )
    elif any(f"chr{name}" in available for name in missing):
        message += (
            " The FASTA appears to use UCSC-style names ('chr' prefix); "
            "set the expected chromosomes to match "
            "(e.g. --expected-chromosomes 'chr1,chr2,chr3,...')."
        )
    else:
        message += (
            " Check that the expected chromosomes match the sequence names "
            "in the FASTA (--expected-chromosomes)."
        )
    return message


class GenomeMethylationEmbedding:
    """Manages CpG site positions and coordinate conversions for a reference genome.

    This class parses a reference genome FASTA file to identify all CpG sites
    (positions where cytosine is followed by guanine), creating a mapping between
    genomic coordinates and linear embedding indices. This mapping is essential
    for interpreting the sparse matrices produced by bam2tensor.

    The embedding assigns each CpG site a unique integer index, ordered by:
    1. Chromosome order (as specified in expected_chromosomes)
    2. Genomic position within each chromosome

    CpG site data is cached to disk after the first parse, significantly
    speeding up subsequent runs on the same genome.

    Attributes:
        genome_name: Identifier for this genome (used for cache file naming).
        fasta_source: Path to the reference FASTA file.
        expected_chromosomes: List of chromosome names to include.
        verbose: Whether to print progress information.
        cpg_sites_dict: Dictionary mapping chromosome names to lists of
            CpG positions (1-based coordinates).
        total_cpg_sites: Total number of CpG sites across all chromosomes.
        cache_file: Path to the cache file for this genome.

    Example:
        >>> # xdoctest: +SKIP
        >>> embedding = GenomeMethylationEmbedding(
        ...     genome_name="hg38",
        ...     expected_chromosomes=["chr1", "chr2"],
        ...     fasta_source="reference.fa",
        ... )
        >>> embedding.total_cpg_sites
        2847432
        >>> embedding.embedding_to_genomic_position(0)
        ('chr1', 10525)
        >>> embedding.genomic_position_to_embedding("chr1", 10525)
        0
    """

    def __init__(
        self,
        genome_name: str,
        expected_chromosomes: list[str],
        fasta_source: str,
        window_size: int = 150,
        skip_cache: bool = False,
        verbose: bool = False,
    ):
        """Initialize the methylation embedding by parsing or loading CpG sites.

        Creates a new GenomeMethylationEmbedding by either:
        1. Loading CpG site positions from a cached file (if available), or
        2. Parsing the reference FASTA to find all CpG sites (slow, ~10 min for human)

        After initialization, the embedding provides efficient bidirectional
        conversion between genomic coordinates and matrix column indices.

        Args:
            genome_name: A unique identifier for this genome configuration
                (e.g., "hg38", "mm10", "hg38-no-alt"). This is used to name
                the cache file: "{genome_name}.cache.json.gz".
            expected_chromosomes: List of chromosome names to include in the
                embedding. Chromosomes not in this list are skipped when parsing
                the FASTA. Order matters: chromosomes are processed in this order
                and the embedding indices follow this ordering.
            fasta_source: Path to the reference genome FASTA file. This should
                match the genome used for aligning the BAM files.
            window_size: Reserved for future use. Currently unused but stored
                in the cache for compatibility checking.
            skip_cache: If True, ignore any existing cache file and re-parse
                the FASTA. Useful when the reference has been modified or the
                chromosome list has changed.
            verbose: If True, print progress information during FASTA parsing
                and cache operations.

        Raises:
            ValueError: If expected_chromosomes is empty.

        Note:
            A ``ValueError`` also propagates from the cache loader when an
            existing cache was built for a different chromosome list or
            window size, and from the FASTA parser when an expected
            chromosome is absent from the reference.

        Example:
            >>> # xdoctest: +SKIP
            >>> # First run: parses FASTA and creates cache
            >>> embedding = GenomeMethylationEmbedding(
            ...     genome_name="hg38",
            ...     expected_chromosomes=["chr1", "chr2", "chr3"],
            ...     fasta_source="/data/GRCh38.fa",
            ...     verbose=True,
            ... )
            Parsing fasta file for CpG sites: /data/GRCh38.fa
            ...
            Saved embedding cache.

            >>> # Subsequent runs: loads from cache (fast)
            >>> embedding2 = GenomeMethylationEmbedding(
            ...     genome_name="hg38",
            ...     expected_chromosomes=["chr1", "chr2", "chr3"],
            ...     fasta_source="/data/GRCh38.fa",
            ... )
            Loaded methylation embedding from cache: hg38.cache.json.gz
        """
        self.genome_name = genome_name
        self.fasta_source = fasta_source
        self.expected_chromosomes = expected_chromosomes
        self.verbose = verbose
        self.window_size = window_size

        # Store the CpG sites in a dict per chromosome
        self.cpg_sites_dict: dict[str, list[int]] = {}

        self.cache_file = self.genome_name + ".cache.json.gz"

        # Check that the expected chromosomes are not empty
        if len(self.expected_chromosomes) == 0:
            raise ValueError("Expected chromosomes cannot be empty")

        # Try to load a cached embedding if it exists
        cache_available = False
        if not skip_cache:
            try:
                cache_available = self.load_embedding_cache()
            except FileNotFoundError as e:
                # Stale-cache rejections (version or FASTA SHA-256 mismatch,
                # unreadable or incomplete cache file) raise FileNotFoundError
                # too — always surface those so users are not silently
                # regenerating a cache they thought was valid.
                if os.path.exists(self.cache_file):
                    print(f"Discarding stale embedding cache: {e}")
                elif self.verbose:
                    print("Could not load methylation embedding from cache: " + str(e))

        if not cache_available:
            print("No cache available, generating fresh methylation embedding.")
            # Generate CpG sites if we don't have a cached embedding
            self.parse_fasta_for_cpg_sites()

            if not skip_cache:
                # Save the key & expensive objects to a cache
                self.save_embedding_cache()
        else:
            print(f"Loaded methylation embedding from cache: {self.cache_file}")

        ## Generate objects for efficient lookups
        # A dict of chromosomes -> index for quick lookups (e.g. "chr1" -> 0)
        self.chromosomes_dict: dict[str, int] = {
            ch: idx for idx, ch in enumerate(self.expected_chromosomes)
        }

        # How many CpG sites are there?
        self.total_cpg_sites = sum([len(v) for v in self.cpg_sites_dict.values()])
        if self.verbose:
            print(f"\tTotal CpG sites: {self.total_cpg_sites:,}")

        # Create a dictionary of chromosome -> CpG site -> index (embedding) for efficient lookup
        self.chr_to_cpg_to_embedding_dict = {
            ch: {cpg: idx for idx, cpg in enumerate(self.cpg_sites_dict[ch])}
            for ch in self.expected_chromosomes
        }

        # Count the number of CpGs per chromosome
        cpgs_per_chr: dict[str, int] = {
            k: len(v) for k, v in self.cpg_sites_dict.items()
        }

        # Add up the number of CpGs per chromosome, e.g. chr1, then chr1+chr2, then chr1+chr2+chr3, etc
        self.cpgs_per_chr_cumsum: np.ndarray = np.cumsum(
            [cpgs_per_chr[k] for k in self.expected_chromosomes]
        )

        if verbose:
            print("Loaded methylation embedding.")

    def save_embedding_cache(self) -> None:
        """Save the CpG site index to a compressed cache file.

        Serializes the CpG site positions and metadata to a gzipped JSON file.
        The cache file is named "{genome_name}.cache.json.gz" and contains:
        - genome_name: The genome identifier
        - fasta_source: Path to the original FASTA file
        - fasta_sha256: SHA-256 of the FASTA file bytes (for cache validation)
        - bam2tensor_version: Version that produced this cache
        - total_cpg_sites: Total CpG count across all included chromosomes
        - expected_chromosomes: List of included chromosomes
        - window_size: The window_size parameter (for compatibility checking)
        - cpg_sites_dict: Dictionary of chromosome -> list of CpG positions

        The cache can be loaded by future GenomeMethylationEmbedding instances
        to avoid re-parsing the FASTA file.

        Raises:
            AssertionError: If cpg_sites_dict is empty (nothing to cache).

        Note:
            Cache files use compression level 3 for a balance between
            file size and write speed. The cache is written to a temporary
            file and renamed into place, so an interrupted run (or several
            jobs starting at once in the same directory) can never leave a
            truncated cache behind.
        """

        assert len(self.cpg_sites_dict) > 0, "CpG sites dict is empty!"

        total_cpg_sites = sum(len(v) for v in self.cpg_sites_dict.values())
        cache_data = {
            "genome_name": self.genome_name,
            "fasta_source": self.fasta_source,
            "fasta_sha256": compute_fasta_sha256(self.fasta_source),
            "bam2tensor_version": __version__,
            "total_cpg_sites": total_cpg_sites,
            "expected_chromosomes": self.expected_chromosomes,
            "window_size": self.window_size,
            "cpg_sites_dict": self.cpg_sites_dict,
        }

        if self.verbose:
            print(f"\tSaving embedding to cache: {self.cache_file}")
        # Write-then-rename: os.replace() is atomic, so readers only ever see
        # a complete cache. The PID keeps concurrent writers (e.g. array jobs
        # sharing a working directory) from clobbering each other's temp file.
        tmp_file = f"{self.cache_file}.{os.getpid()}.tmp"
        try:
            with gzip.open(tmp_file, "wt", compresslevel=3, encoding="utf-8") as f:
                json.dump(cache_data, f)
            os.replace(tmp_file, self.cache_file)
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        if self.verbose:
            print("\tSaved embedding cache.")

    def load_embedding_cache(self) -> bool:
        """Load CpG site index from a previously saved cache file.

        Attempts to load the cache file "{genome_name}.cache.json.gz" and
        restore all CpG site data. If successful, this avoids the slow
        FASTA parsing step.

        Provenance is validated before the cached data is trusted: the
        cache must be readable, must have been written by the same
        version of bam2tensor, must reference a FASTA file with the same
        SHA-256 as the current ``fasta_source``, and must contain every
        expected chromosome. A cache that fails any of these checks is
        rejected with a ``FileNotFoundError`` so the caller falls through
        to a fresh FASTA parse and overwrites the stale cache on save.

        A cache that is valid but was built for a *different
        configuration* (chromosome list or window size) is not silently
        replaced: the same genome name is being used for two different
        column layouts, so a ``ValueError`` asks the user to disambiguate.

        Returns:
            True if the cache was successfully loaded.

        Raises:
            FileNotFoundError: If the cache file does not exist, or if
                the cache is stale or unusable (unreadable/truncated
                file, version mismatch, FASTA SHA-256 mismatch, or
                missing chromosomes).
            ValueError: If the cache was built for a different list of
                expected chromosomes or a different window size.

        Note:
            Only ``cpg_sites_dict`` is populated from the cache. The
            instance's own configuration (``genome_name``,
            ``fasta_source``, ``expected_chromosomes``) is left untouched.
        """

        if not os.path.exists(self.cache_file):
            raise FileNotFoundError("No cache of embedding found.")

        if self.verbose:
            print(f"\tReading embedding from cache: {self.cache_file}")

        # A truncated or corrupt cache (e.g. from a run interrupted while an
        # older version was writing it) must not be fatal: reject it and let
        # the caller regenerate.
        try:
            with gzip.open(self.cache_file, "rt", encoding="utf-8") as f:
                cache_data = json.load(f)
        except (OSError, EOFError, ValueError, zlib.error) as exc:
            raise FileNotFoundError(
                f"Unreadable cache {self.cache_file!r} "
                f"({type(exc).__name__}: {exc}). Regenerating."
            ) from exc
        if not isinstance(cache_data, dict):
            raise FileNotFoundError(
                f"Unreadable cache {self.cache_file!r} (not a JSON object). "
                "Regenerating."
            )

        # Validate cache provenance: stale caches predating v2.7 used a
        # case-sensitive CpG search that silently dropped roughly half
        # the CpG sites in soft-masked FASTAs (e.g. UCSC's hg38.fa.gz).
        cached_version = cache_data.get("bam2tensor_version")
        if cached_version != __version__:
            raise FileNotFoundError(
                f"Stale cache {self.cache_file!r}: written by bam2tensor "
                f"{cached_version!r}, current is {__version__!r}. "
                "Regenerating."
            )

        cached_fasta_sha256 = cache_data.get("fasta_sha256")
        current_fasta_sha256 = compute_fasta_sha256(self.fasta_source)
        if cached_fasta_sha256 != current_fasta_sha256:
            raise FileNotFoundError(
                f"Stale cache {self.cache_file!r}: FASTA SHA-256 mismatch "
                f"(cache={cached_fasta_sha256}, current={current_fasta_sha256}). "
                "Regenerating."
            )

        cached_chromosomes = cache_data.get("expected_chromosomes")
        cpg_sites_dict = cache_data.get("cpg_sites_dict")
        if not isinstance(cached_chromosomes, list) or not isinstance(
            cpg_sites_dict, dict
        ):
            raise FileNotFoundError(
                f"Unreadable cache {self.cache_file!r} (missing CpG site data). "
                "Regenerating."
            )

        # The cache is keyed only by genome name. A different chromosome list
        # (or window size) under the same name means a different column
        # layout, so refuse rather than silently overwrite either one.
        if cached_chromosomes != self.expected_chromosomes:
            raise ValueError(
                f"Expected chromosomes do not match the cache {self.cache_file!r}: "
                f"it was built for [{_format_names(cached_chromosomes)}] but "
                f"[{_format_names(self.expected_chromosomes)}] were requested. "
                "Use a different genome name for this chromosome list, skip "
                "the cache (--skip-cache), or delete the cache file."
            )
        cached_window_size = cache_data.get("window_size")
        if cached_window_size != self.window_size:
            raise ValueError(
                f"Window size does not match the cache {self.cache_file!r}: "
                f"it was built with {cached_window_size} but {self.window_size} "
                "was requested. Use a different genome name, skip the cache "
                "(--skip-cache), or delete the cache file."
            )

        # Releases up to 2.8 could save a cache that lacked an expected
        # chromosome (the FASTA did not contain it) before crashing. Reject
        # those so the fresh parse can report the problem properly.
        missing = [ch for ch in self.expected_chromosomes if ch not in cpg_sites_dict]
        if missing:
            raise FileNotFoundError(
                f"Stale cache {self.cache_file!r}: no CpG sites stored for "
                f"{_format_names(missing)}. Regenerating."
            )

        self.cpg_sites_dict = cpg_sites_dict

        if self.verbose:
            print(f"\tCached genome fasta source: {cache_data.get('fasta_source')}")

        return True

    def parse_fasta_for_cpg_sites(self) -> None:
        """Parse the reference FASTA to find all CpG sites.

        Scans each chromosome in the reference genome FASTA file to identify
        all CpG dinucleotide positions (where C is immediately followed by G).
        Results are stored in self.cpg_sites_dict.

        This operation is slow for large genomes (~10 minutes for GRCh38/hg38)
        but only needs to be done once per genome. Results should be cached
        using save_embedding_cache() for faster subsequent runs.

        The method populates self.cpg_sites_dict with the structure::

            {
                "chr1": [10525, 10542, 10563, ...],  # 1-based positions
                "chr2": [10182, 10193, 10218, ...],
                ...
            }

        Positions are 1-based to match standard genomic coordinate conventions
        (e.g., samtools faidx uses 1-based coordinates).

        Only chromosomes listed in self.expected_chromosomes are processed;
        other sequences in the FASTA are skipped with a message (if verbose).

        Raises:
            FileNotFoundError: If the FASTA file cannot be read.
            ValueError: If any expected chromosome is absent from the
                FASTA. The message lists the missing names and the
                sequences that are available, and flags ``chr`` prefix
                mismatches (UCSC vs. Ensembl naming).

        Note:
            For the human genome (GRCh38), expect approximately 28 million
            CpG sites across all chromosomes.
        """

        if self.verbose:
            print(f"\nParsing fasta file for CpG sites: {self.fasta_source}")

        # Check that we can read the fasta file
        if not os.access(self.fasta_source, os.R_OK):
            raise FileNotFoundError(
                "Cannot read fasta file: " + os.path.abspath(self.fasta_source)
            )

        # Every sequence name seen in the FASTA, for the missing-chromosome
        # diagnostic below.
        fasta_names: list[str] = []

        # Open the FASTA ourselves so the handle is closed deterministically,
        # including when parsing stops early on an error.
        with open(self.fasta_source) as fasta_handle:
            # Iterate over sequences
            for seqrecord in tqdm(
                SeqIO.parse(fasta_handle, "fasta"),  # type: ignore
                total=len(self.expected_chromosomes),
                disable=not self.verbose,
            ):
                fasta_names.append(seqrecord.id)
                if seqrecord.id not in self.expected_chromosomes:
                    if self.verbose:
                        tqdm.write(f"\tSkipping chromosome {seqrecord.id}")
                    continue
                # Upper-case the sequence so soft-masked FASTAs (UCSC's default
                # hg38.fa.gz uses lowercase for RepeatMasker/TRF regions) do not
                # silently drop CpGs in repeats — that is roughly half of all
                # CpGs in the human genome.
                sequence = seqrecord.seq.upper()

                # Find all CpG sites
                # The pos+1 is because we want to store the 1-based position, because .bed is wild and arguably 1-based maybe:
                # e.g. https://genome-blog.soe.ucsc.edu/blog/2016/12/12/the-ucsc-genome-browser-coordinate-counting-systems/
                # Regardless, $ samtools faidx GRCh38-DAC-U2AF1.fna chr1:0-0 throws an error, while
                # $ samtools faidx GRCh38-DAC-U2AF1.fna chr1:1-1 returns the correct base.

                cpg_indices = []
                search_str = "CG"
                pos = sequence.find(search_str)

                while pos != -1:
                    cpg_indices.append(pos + 1)
                    pos = sequence.find(search_str, pos + 1)

                self.cpg_sites_dict[seqrecord.id] = cpg_indices

        # Fail loudly, here, if the FASTA does not provide every expected
        # chromosome — otherwise the column layout cannot be built and the
        # failure surfaces later as an opaque KeyError.
        missing = [
            ch for ch in self.expected_chromosomes if ch not in self.cpg_sites_dict
        ]
        if missing:
            raise ValueError(
                _missing_chromosomes_message(missing, fasta_names, self.fasta_source)
            )

        if self.verbose:
            print(f"\tFound {len(self.cpg_sites_dict)} chromosomes in reference fasta.")

    def embedding_to_genomic_position(
        self, embedding: int | np.int64
    ) -> tuple[str, int]:
        """Convert a matrix column index to genomic coordinates.

        Given an embedding index (column number in the sparse matrix), returns
        the corresponding chromosome name and 1-based genomic position.

        This is the inverse of genomic_position_to_embedding().

        Args:
            embedding: The matrix column index (0-based). Must be in the range
                [0, total_cpg_sites).

        Returns:
            A tuple of (chromosome, position) where:
            - chromosome is the chromosome name (e.g., "chr1")
            - position is the 1-based genomic coordinate of the CpG site

        Raises:
            AssertionError: If embedding is out of range.

        Example:
            >>> # xdoctest: +SKIP
            >>> embedding = GenomeMethylationEmbedding(...)
            >>> chrom, pos = embedding.embedding_to_genomic_position(12345)
            >>> print(f"Column 12345 is {chrom}:{pos}")
            Column 12345 is chr1:987654
        """
        assert embedding >= 0 and embedding < self.total_cpg_sites

        # Find the index of the first element in cpgs_per_chr_cumsum that is greater than or equal to the embedding position
        chr_index = np.searchsorted(
            self.cpgs_per_chr_cumsum, embedding + 1, side="left"
        )

        # Now we know the chromosome, but we need to find the position within the chromosome
        # If this is the first chromosome, the position is just the embedding position
        if chr_index == 0:
            return (
                self.expected_chromosomes[chr_index],
                self.cpg_sites_dict[self.expected_chromosomes[chr_index]][embedding],
            )
        # Otherwise, subtract the length of the previous chromosomes from the embedding position
        embedding -= self.cpgs_per_chr_cumsum[chr_index - 1]  # type: ignore
        return (
            self.expected_chromosomes[chr_index],
            self.cpg_sites_dict[self.expected_chromosomes[chr_index]][embedding],
        )

    def genomic_position_to_embedding(self, chrom: str, pos: int) -> int:
        """Convert genomic coordinates to a matrix column index.

        Given a chromosome name and 1-based genomic position, returns the
        corresponding matrix column index (embedding position).

        This is the inverse of embedding_to_genomic_position().

        Args:
            chrom: The chromosome name (e.g., "chr1"). Must be one of the
                chromosomes in expected_chromosomes.
            pos: The 1-based genomic position of the CpG site. Must be a
                valid CpG position in the specified chromosome.

        Returns:
            The matrix column index (0-based) corresponding to this CpG site.

        Example:
            >>> # xdoctest: +SKIP
            >>> embedding = GenomeMethylationEmbedding(...)
            >>> col_idx = embedding.genomic_position_to_embedding("chr1", 10525)
            >>> print(f"chr1:10525 is at column {col_idx}")
            chr1:10525 is at column 0

        Note:
            This method assumes the position is a known CpG site. Calling with
            a position that is not a CpG site will raise a KeyError.
        """
        # Find the index of the chromosome
        chr_index = self.chromosomes_dict[chrom]
        # Find the index of the CpG site in the chromosome
        cpg_index = self.chr_to_cpg_to_embedding_dict[chrom][pos]
        # If this is the first chromosome, the embedding position is just the CpG index
        if chr_index == 0:
            return int(cpg_index)
        # Otherwise, add the length of the previous chromosomes to the CpG index
        return int(cpg_index + self.cpgs_per_chr_cumsum[chr_index - 1])
