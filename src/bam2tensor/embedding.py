"""Stores methylation embedding information for a given genome."""

import gzip
import json
import os

# Third party modules
import numpy as np

from tqdm import tqdm
from Bio import SeqIO


class GenomeMethylationEmbedding:
    """Stores methylation embedding information for a given genome.

    This class helps identify CpG sites in a given .fasta file and stores
    methylation embedding information for each CpG site for analyzing .bam files.

    Stored embedding information cached and stored in gzipped json files.

    Attributes:
        genome_name: The name of the genome.
        fasta_source: The source of the fasta file.
        expected_chromosomes: The expected chromosomes in the genome.
        methylation_embedding: The methylation embedding for the genome.

    """

    genome_name = None
    fasta_source = None
    expected_chromosomes = None
    embedding_loaded = False
    cpg_sites_dict = None

    def __init__(
        self,
        genome_name,
        expected_chromosomes,
        fasta_source: str = None,
        skip_cache: bool = False,
        verbose: bool = False,
    ):
        self.genome_name = genome_name
        self.fasta_source = fasta_source
        self.expected_chromosomes = expected_chromosomes
        self.verbose = verbose
        self.window_size = 150

        # Store the CpG sites in a dict per chromosome
        self.cpg_sites_dict: dict[str, list[int]] = {}

        self.cached_cpg_sites_json = self.genome_name + ".cpg_all_sites.json.gz"

        self.windowed_cpg_sites_cache = self.genome_name + ".cpg_windowed_sites.json.gz"
        # This is a dict of lists, where but each list contains a tuple of CpG ranges witin a window
        # Key: chromosome, e.g. "chr1"
        # Value: a list of tuples, e.g. [(0,35), (190,212), (1055,)]
        self.windowed_cpg_sites_dict = {}

        self.windowed_cpg_sites_reverse_cache = (
            self.genome_name + ".cpg_windowed_sites_reverse.json.gz"
        )
        # And a reverse dict of dicts where chrom->window_start->[cpgs]
        self.windowed_cpg_sites_dict_reverse = {}

        # TODO: Store the expected chromosomes in some cached object.
        # Check that the expected chromosomes are not empty
        if len(self.expected_chromosomes) == 0:
            raise ValueError("Expected chromosomes cannot be empty")

        # Try to load a cached methylation embedding if it exists
        if not skip_cache:
            try:
                self.load_cpg_site_cache()
            except FileNotFoundError as e:
                if verbose:
                    print("Could not load methylation embedding from cache: " + str(e))

        if len(self.cpg_sites_dict) == 0:
            self.parse_fasta_for_cpg_sites()

        if not skip_cache:
            self.save_cpg_site_cache()

        # How many CpG sites are there?
        total_cpg_sites = sum([len(v) for v in self.cpg_sites_dict.values()])
        if verbose:
            print(f"Total CpG sites: {total_cpg_sites:,}")

        assert total_cpg_sites > 28_000_000  # Validity check for hg38

        # TODO: Shove these and expected_chromosomes into an object that we can save and cache
        # Create a dictionary of chromosome -> CpG site -> index (embedding) for efficient lookup
        self.chr_to_cpg_to_embedding_dict = {
            ch: {cpg: idx for idx, cpg in enumerate(self.cpg_sites_dict[ch])}
            for ch in self.expected_chromosomes
        }

        # Count the number of CpGs per chromosome
        self.cpgs_per_chr: dict[str, int] = {
            k: len(v) for k, v in self.cpg_sites_dict.items()
        }

        # Add up the number of CpGs per chromosome, e.g. chr1, then chr1+chr2, then chr1+chr2+chr3, etc
        self.cpgs_per_chr_cumsum: np.ndarray = np.cumsum(
            [self.cpgs_per_chr[k] for k in self.expected_chromosomes]
        )

    def load_cpg_site_cache(self):
        """Load a cache of CpG sites from a previously parsed fasta."""

        if self.verbose:
            print(f"\nLoading all CpG sites for: {self.genome_name}")

        if os.path.exists(self.cached_cpg_sites_json):
            if self.verbose:
                print(
                    f"\tLoading all CpG sites from cache: {self.cached_cpg_sites_json}"
                )
            # TODO: Add type hinting via TypedDicts?
            # e.g. https://stackoverflow.com/questions/51291722/define-a-jsonable-type-using-mypy-pep-526
            with gzip.open(self.cached_cpg_sites_json, "rt") as f:
                cpg_sites_dict = json.load(f)

            return cpg_sites_dict  # type: ignore
        else:
            raise FileNotFoundError("\tNo cache of all CpG sites found.")

    def parse_fasta_for_cpg_sites(self):
        """Generate a dict of *all* CpG sites across each chromosome in the reference genome.

        This can take a while (~10 minutes for GRCh38).

        This generates cpg_sites_dict: A dict of CpG sites for each chromosome in the reference genome.
        It's a dict of lists, where the key is the chromosome: e.g. "chr1"
        The value is a list of CpG sites: e.g. [0, 35, 190, 212, 1055, ...]

        We store this as a dict because it's easier to portably serialize to disk as JSON.
        """

        if self.verbose:
            print(f"\nParsing fasta file for CpG sites: {self.fasta_source}")

        # Check that we can read the fasta file
        if not os.access(self.fasta_source, os.R_OK):
            raise FileNotFoundError("Cannot read fasta file: " + self.fasta_source)

        # Iterate over sequences
        for seqrecord in tqdm(
            SeqIO.parse(self.fasta_source, "fasta"),  # type: ignore
            total=len(self.expected_chromosomes),
            disable=not self.verbose,
        ):
            if seqrecord.id not in self.expected_chromosomes:
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

    def save_cpg_site_cache(self):
        """Save a cache of CpG sites from a previously parsed fasta."""

        assert self.cpg_sites_dict is not None

        if self.verbose:
            print(f"\tSaving all cpg sites to cache: {self.cached_cpg_sites_json}")
        with gzip.open(
            self.cached_cpg_sites_json, "wt", compresslevel=3, encoding="utf-8"
        ) as f:
            json.dump(self.cpg_sites_dict, f)

    def load_windowed_cpg_site_cache(self):
        """Load a cache of winowed CpG sites."""

        if os.path.exists(self.windowed_cpg_sites_cache) and os.path.exists(
            self.windowed_cpg_sites_reverse_cache
        ):
            if self.verbose:
                print(
                    f"\tLoading windowed CpG sites from caches:\n\t\t{self.windowed_cpg_sites_cache}\n\t\t{self.windowed_cpg_sites_reverse_cache}"
                )

            with gzip.open(self.windowed_cpg_sites_cache, "rt") as f:
                self.windowed_cpg_sites_dict = json.load(f)
            with gzip.open(self.windowed_cpg_sites_reverse_cache, "rt") as f:
                # This wild object_hook is to convert the keys back to integers, since JSON only supports strings as keys
                self.windowed_cpg_sites_dict_reverse = json.load(
                    f,
                    object_hook=lambda d: {
                        int(k) if k.isdigit() else k: v for k, v in d.items()
                    },
                )
        else:
            raise FileNotFoundError("\tNo cache of windowed CpG sites found.")

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

    def save_windowed_cpg_site_cache(self):
        """Save a cache of windowed CpG sites."""

        # Save these to .json caches
        if self.verbose:
            print(
                f"\tSaving windowed CpG sites to caches:\n\t\t{self.windowed_cpg_sites_cache}\n\t\t{self.windowed_cpg_sites_reverse_cache}"
            )

        with gzip.open(
            self.windowed_cpg_sites_cache, "wt", compresslevel=3, encoding="utf-8"
        ) as f:
            json.dump(self.windowed_cpg_sites_dict, f)
        with gzip.open(
            self.windowed_cpg_sites_reverse_cache,
            "wt",
            compresslevel=3,
            encoding="utf-8",
        ) as f:
            json.dump(self.windowed_cpg_sites_dict_reverse, f)

    def get_genomic_position(self, embedding: int) -> tuple[str, int]:
        """Given an embedding position, return the chromosome and position.

        Parameters
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

    def test_stub(self):
        """
        Holder stub for some test.

        TODO: Move these into to a formal test framework

        # TODO: Simplify the input framework (likely object orient the window / cpg dict?)
        """
        assert self.cpgs_per_chr_cumsum[-1] == self.total_cpg_sites
        assert self.embedding_to_genomic_position(
            self.total_cpg_sites, self.cpg_sites_dict, self.cpgs_per_chr_cumsum, 0
        ) == ("chr1", self.cpg_sites_dict["chr1"][0])
        assert self.embedding_to_genomic_position(
            self.total_cpg_sites, self.cpg_sites_dict, self.cpgs_per_chr_cumsum, 1
        ) == ("chr1", self.cpg_sites_dict["chr1"][1])
        # Edges
        assert self.embedding_to_genomic_position(
            self.total_cpg_sites,
            self.cpg_sites_dict,
            self.cpgs_per_chr_cumsum,
            self.cpgs_per_chr_cumsum[0],
        ) == ("chr2", self.cpg_sites_dict["chr2"][0])
        assert self.embedding_to_genomic_position(
            self.total_cpg_sites,
            self.cpg_sites_dict,
            self.cpgs_per_chr_cumsum,
            self.cpgs_per_chr_cumsum[-1] - 1,
        ) == ("chrY", self.cpg_sites_dict["chrY"][-1])

        ### Tests
        assert (
            self.genomic_position_to_embedding(
                self.chr_to_cpg_to_embedding_dict,
                self.cpgs_per_chr_cumsum,
                "chr1",
                self.cpg_sites_dict["chr1"][0],
            )
            == 0
        )
        assert (
            self.genomic_position_to_embedding(
                self.chr_to_cpg_to_embedding_dict,
                self.cpgs_per_chr_cumsum,
                "chr1",
                self.cpg_sites_dict["chr1"][1],
            )
            == 1
        )
        # Edges
        assert (
            self.genomic_position_to_embedding(
                self.chr_to_cpg_to_embedding_dict,
                self.cpgs_per_chr_cumsum,
                "chr2",
                self.cpg_sites_dict["chr2"][0],
            )
            == self.cpgs_per_chr_cumsum[0]
        )
        assert (
            self.genomic_position_to_embedding(
                self.chr_to_cpg_to_embedding_dict,
                self.cpgs_per_chr_cumsum,
                "chrY",
                self.cpg_sites_dict["chrY"][-1],
            )
            == self.cpgs_per_chr_cumsum[-1] - 1
        )
        ########