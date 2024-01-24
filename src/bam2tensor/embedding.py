"""Stores methylation embedding information for a given genome."""

import gzip
import json
import os
from typing import Union  # Remove when dropping Python 3.9

# Third party modules
import numpy as np

from tqdm import tqdm
from Bio import SeqIO


class GenomeMethylationEmbedding:
    """Stores methylation embedding information for a given genome.

    This class helps identify CpG sites in a given .fasta file and stores
    methylation embedding information for each CpG site for analyzing .bam files.

    Stored embedding information cached and stored in gzipped json files.
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
        """Initialize the methylation embedding.

        Args
        ----------
        genome_name : str
            The name of the genome.
        expected_chromosomes : list[str]
            The expected chromosomes in the genome.
        fasta_source : str, optional
            The source of the fasta file.
        skip_cache : bool, optional
            Skip loading from cache.
        verbose : bool, optional
            Verbose output.

        Returns
        --------
        None

        Raises
        -------
        ValueError
            If the expected chromosomes are empty.
        FileNotFoundError
            If the fasta file cannot be found.
        """
        self.genome_name = genome_name
        self.fasta_source = fasta_source
        self.expected_chromosomes = expected_chromosomes
        self.verbose = verbose
        self.window_size = window_size

        # Store the CpG sites in a dict per chromosome
        self.cpg_sites_dict: dict[str, list[int]] = {}

        # This is a dict of lists, where but each list contains a tuple of CpG ranges witin a window
        # Key: chromosome, e.g. "chr1"
        # Value: a list of tuples, e.g. [(0,35), (190,212), (1055,)]
        self.windowed_cpg_sites_dict: dict[str, list[tuple]] = {}

        # And a reverse dict of dicts where chrom->window_start->[cpgs]
        self.windowed_cpg_sites_dict_reverse: dict[str, dict[int, list]] = {}

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
            # Generate CpG sites if we don't have a cached embedding
            self.parse_fasta_for_cpg_sites()

            # Now generate windowed CpG sites for efficient querying of .bam files
            self.generate_windowed_cpg_sites()

            if not skip_cache:
                # Save the key & expensive objects to a cache
                self.save_embedding_cache()

        ## Generate objects for efficient lookups
        # A dict of chromosomes -> index for quick lookups (e.g. "chr1" -> 0)
        self.chromosomes_dict: dict[str, int] = {
            ch: idx for idx, ch in enumerate(self.expected_chromosomes)
        }

        # How many CpG sites are there?
        self.total_cpg_sites = sum([len(v) for v in self.cpg_sites_dict.values()])
        if self.verbose:
            print(f"\tTotal CpG sites: {self.total_cpg_sites:,}")
            print(
                f"\tTotal number of windows (at window_size = {self.window_size}): {len(self.windowed_cpg_sites_dict):,}"
            )

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

    def save_embedding_cache(self):
        """Save a cache of expensive objects as our methylation embedding."""

        assert len(self.cpg_sites_dict) > 0, "CpG sites dict is empty!"
        assert (
            len(self.windowed_cpg_sites_dict) > 0
        ), "Windowed CpG sites dict is empty!"

        cache_data = {
            "genome_name": self.genome_name,
            "fasta_source": self.fasta_source,
            "expected_chromosomes": self.expected_chromosomes,
            "window_size": self.window_size,
            "cpg_sites_dict": self.cpg_sites_dict,
            "windowed_cpg_sites_dict": self.windowed_cpg_sites_dict,
            "windowed_cpg_sites_dict_reverse": self.windowed_cpg_sites_dict_reverse,
        }

        if self.verbose:
            print(f"\tSaving embedding to cache: {self.cache_file}")
        with gzip.open(self.cache_file, "wt", compresslevel=3, encoding="utf-8") as f:
            json.dump(cache_data, f)
        if self.verbose:
            print("\tSaved embedding cache cache.")

    def load_embedding_cache(self) -> bool:
        """Load our cached embedding data from a prior run.

        Raises
        -------
        FileNotFoundError
            If the cached CpG site file cannot be found.
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
            self.windowed_cpg_sites_dict = self.cache_data["windowed_cpg_sites_dict"]

            # This is to convert the keys back to integers, since JSON only supports strings as keys
            # Note that we want this on the second level keys '1234', not the first level keys 'chr1'
            self.windowed_cpg_sites_dict_reverse = {
                chrom: {
                    int(cpg) if cpg.isdigit() else cpg: window
                    for cpg, window in v.items()
                }
                for chrom, v in self.cache_data[
                    "windowed_cpg_sites_dict_reverse"
                ].items()
            }

            if self.verbose:
                print(f"\tCached genome fasta source: {self.fasta_source}")
        else:
            raise FileNotFoundError("No cache of embedding found.")

        return True

    def parse_fasta_for_cpg_sites(self):
        """Generate a dict of *all* CpG sites across each chromosome in the reference genome.

        This can take a while (~10 minutes for GRCh38).

        This generates cpg_sites_dict: A dict of CpG sites for each chromosome in the reference genome.
        It's a dict of lists, where the key is the chromosome: e.g. "chr1"
        The value is a list of CpG sites: e.g. [0, 35, 190, 212, 1055, ...]

        We store this as a dict because it's easier to portably serialize to disk as JSON.

        Raises
        -------
        FileNotFoundError
            If the fasta file cannot be found.
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

    def generate_windowed_cpg_sites(self):
        """Generate a dict of CpG sites for each chromosome in the reference genome.

        This is a dict of lists, where but each list contains a tuple of CpG ranges witin a window
        The key is the chromosome: e.g. "chr1"
        The value is a list of CpG sites: e.g. [(0, 35), (190, 212), (1055, ...)]

        We store this as a dict because it's easier to portably serialize to disk as JSON.

        NOTE: The window size is tricky -- it's set usually to the read size, but reads can be larger than the window size,
        since they can map with deletions, etc. So we need to be careful here."""

        # Let's generate ranges (windows) of all CpGs within READ_SIZE of each other
        # This is to subsequently query our .bam/.sam files for reads containing CpGs
        # TODO: Shrink read size a little to account for trimming?
        assert (
            10 < self.window_size < 500
        ), "Read size is atypical, please double check (only validated for ~150 bp.)"

        if self.verbose:
            print(
                f"\n\tGenerating windowed CpG sites dict (window size = {self.window_size} bp.)\n"
            )

        # Loop over all chromosomes
        for chrom, cpg_sites in tqdm(
            self.cpg_sites_dict.items(), disable=not self.verbose
        ):
            self.windowed_cpg_sites_dict[chrom] = []
            self.windowed_cpg_sites_dict_reverse[chrom] = {}
            window_start = None
            window_end = None

            # Loop over all CpG sites
            cpg_sites_len = len(cpg_sites)
            temp_per_window_cpg_sites = []
            for i in range(cpg_sites_len):
                temp_per_window_cpg_sites.append(cpg_sites[i])

                if window_start is None:
                    window_start = cpg_sites[i]

                # If we're at the end of the chromosome or the next CpG site is too far away
                if (
                    i + 1 == cpg_sites_len
                    or (cpg_sites[i + 1] - cpg_sites[i]) > self.window_size
                ):
                    # We have a complete window
                    window_end = cpg_sites[i]
                    self.windowed_cpg_sites_dict[chrom].append(
                        (window_start, window_end)
                    )
                    self.windowed_cpg_sites_dict_reverse[chrom][
                        window_start
                    ] = temp_per_window_cpg_sites
                    temp_per_window_cpg_sites = []
                    window_start = None
                    window_end = None

        if self.verbose:
            print(
                f"Loaded {len(self.windowed_cpg_sites_dict)} chromosomes from window cache."
            )

    def embedding_to_genomic_position(
        self, embedding: Union[int, np.int64]
    ) -> tuple[str, int]:
        """Given an embedding position, return the chromosome and position.

        Args
        ----------
        embedding_pos : int
            The embedding position.

        Returns
        -------
        tuple:
            The chromosome and position, e.g. ("chr1", 12345)
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
        """Given a genomic position, return the embedding position.

        Args
        ----------
        chrom : str
            The chromosome, e.g. "chr1"
        pos : int
            The position, e.g. 12345

        Returns
        -------
        embedding_pos : int
            The numerical CpG embedding position, e.g. 3493
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
