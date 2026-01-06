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

import numpy as np

from tqdm import tqdm
from Bio import SeqIO


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
                assert (
                    expected_chromosomes == self.expected_chromosomes
                ), "Expected chromosomes do not match cached chromosomes!"
                assert (
                    window_size == self.window_size
                ), "Window size does not match cached window size!"
            except FileNotFoundError as e:
                if self.verbose:
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
        - expected_chromosomes: List of included chromosomes
        - window_size: The window_size parameter (for compatibility checking)
        - cpg_sites_dict: Dictionary of chromosome -> list of CpG positions

        The cache can be loaded by future GenomeMethylationEmbedding instances
        to avoid re-parsing the FASTA file.

        Raises:
            AssertionError: If cpg_sites_dict is empty (nothing to cache).

        Note:
            Cache files use compression level 3 for a balance between
            file size and write speed.
        """

        assert len(self.cpg_sites_dict) > 0, "CpG sites dict is empty!"

        cache_data = {
            "genome_name": self.genome_name,
            "fasta_source": self.fasta_source,
            "expected_chromosomes": self.expected_chromosomes,
            "window_size": self.window_size,
            "cpg_sites_dict": self.cpg_sites_dict,
        }

        if self.verbose:
            print(f"\tSaving embedding to cache: {self.cache_file}")
        with gzip.open(self.cache_file, "wt", compresslevel=3, encoding="utf-8") as f:
            json.dump(cache_data, f)
        if self.verbose:
            print("\tSaved embedding cache.")

    def load_embedding_cache(self) -> bool:
        """Load CpG site index from a previously saved cache file.

        Attempts to load the cache file "{genome_name}.cache.json.gz" and
        restore all CpG site data. If successful, this avoids the slow
        FASTA parsing step.

        Returns:
            True if the cache was successfully loaded.

        Raises:
            FileNotFoundError: If the cache file does not exist.

        Note:
            After loading, the caller should verify that expected_chromosomes
            and window_size match the current configuration, as this method
            overwrites those attributes with cached values.
        """

        if os.path.exists(self.cache_file):
            if self.verbose:
                print(f"\tReading embedding from cache: {self.cache_file}")

            # TODO: Add type hinting via TypedDicts?
            # e.g. https://stackoverflow.com/questions/51291722/define-a-jsonable-type-using-mypy-pep-526
            with gzip.open(self.cache_file, "rt") as f:
                self.cache_data = json.load(f)

            # Load the cached data
            self.genome_name = self.cache_data["genome_name"]
            self.fasta_source = self.cache_data["fasta_source"]
            self.expected_chromosomes = self.cache_data["expected_chromosomes"]
            self.window_size = self.cache_data["window_size"]
            self.cpg_sites_dict = self.cache_data["cpg_sites_dict"]

            if self.verbose:
                print(f"\tCached genome fasta source: {self.fasta_source}")
        else:
            raise FileNotFoundError("No cache of embedding found.")

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

        # Iterate over sequences
        for seqrecord in tqdm(
            SeqIO.parse(self.fasta_source, "fasta"),  # type: ignore
            total=len(self.expected_chromosomes),
            disable=not self.verbose,
        ):
            if seqrecord.id not in self.expected_chromosomes:
                if self.verbose:
                    tqdm.write(f"\tSkipping chromosome {seqrecord.id}")
                continue
            sequence = seqrecord.seq

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
