"""Core functions for bam2tensor.

"""

# Imports
import json
import os

# Third party modules
import numpy as np
import scipy.sparse
import pysam

from tqdm import tqdm
from Bio import SeqIO


def example_function(number1: int, number2: int) -> str:
    """Compare two integers.

    This is merely an example function can be deleted. It is used to show and test generating
    documentation from code, type hinting, testing, and testing examples
    in the code.


    Args:
        number1: The first number.
        number2: The second number, which will be compared to number1.

    Returns:
        A string describing which number is the greatest.

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.

        >>> example_function(1, 2)
        1 is less than 2

    """
    if number1 < number2:
        return f"{number1} is less than {number2}"

    return f"{number1} is greater than or equal to {number2}"


## Globals
HG38_CHROMOSOMES = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
MM39_CHROMOSOMES = ["chr" + str(i) for i in range(1, 20)] + ["chrX", "chrY"]
TEST_CHROMOSOMES = ["chr1"]

CHROMOSOMES = HG38_CHROMOSOMES

# dict: {chromosome: index}, e.g. {"chr1": 0, "chr2": 1, ...}
CHROMOSOMES_DICT = {ch: idx for idx, ch in enumerate(CHROMOSOMES)}


def get_cpg_sites_from_fasta(
    reference_fasta: str, verbose: bool = False, skip_cache: bool = False
) -> dict[str, list[int]]:
    """Generate a dict of *all* CpG sites across each chromosome in the reference genome.

    This is a dict of lists, where the key is the chromosome: e.g. "chr1"
    The value is a list of CpG sites: e.g. [0, 35, 190, 212, 1055, ...]

    We store this as a dict because it's easier to portably serialize to disk as JSON.

    Args:
        reference_fasta: Path to the reference genome fa file.
        verbose: Print verbose output.
        skip_cache: Ignore any cached files (slow!).

    Returns:
        cpg_sites_dict: A dict of CpG sites for each chromosome in the reference genome.
    """
    # We want a list of all CpG sites in the reference genome, including the chromosome and position
    # We'll use this to define our embedding size
    # Each chromosome is identified with a line starting with ">" followed by the chromosome name

    # Store the CpG sites in a dict per chromosome
    cpg_sites_dict: dict[str, list[int]] = {}

    # TODO: Store hash/metadata / reference file info, etc.?
    cached_cpg_sites_json = os.path.splitext(reference_fasta)[0] + ".cpg_all_sites.json"

    if verbose:
        print(f"\nLoading all CpG sites for: {reference_fasta}")

    if os.path.exists(cached_cpg_sites_json) and not skip_cache:
        if verbose:
            print(f"\tLoading all CpG sites from cache: {cached_cpg_sites_json}")
        # TODO: Add type hinting via TypedDicts?
        # e.g. https://stackoverflow.com/questions/51291722/define-a-jsonable-type-using-mypy-pep-526
        with open(cached_cpg_sites_json, "r", encoding="utf-8") as f:
            cpg_sites_dict = json.load(f)

        return cpg_sites_dict  # type: ignore

    if verbose:
        print(
            f"\tNo cache of all CpG sites found (or --skip-cache=True), generating from: {reference_fasta}"
        )

    # Iterate over sequences
    for seqrecord in tqdm(
        SeqIO.parse(reference_fasta, "fasta"),  # type: ignore
        total=len(CHROMOSOMES),
        disable=not verbose,
    ):
        if seqrecord.id not in CHROMOSOMES:
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

        cpg_sites_dict[seqrecord.id] = cpg_indices

    if verbose:
        print(f"\tSaving all cpg sites to cache: {cached_cpg_sites_json}")
    with open(cached_cpg_sites_json, "w", encoding="utf-8") as f:
        json.dump(cpg_sites_dict, f)

    return cpg_sites_dict


# TODO: Ponder this window size, as aligned reads might be larger by a bit... Is this useful?
def get_windowed_cpg_sites(
    reference_fasta: str,
    cpg_sites_dict: dict[str, list[int]],
    window_size: int,
    verbose: bool = False,
    skip_cache: bool = False,
) -> tuple:
    """Generate a dict of CpG sites for each chromosome in the reference genome.

    This is a dict of lists, where but each list contains a tuple of CpG ranges witin a window
    The key is the chromosome: e.g. "chr1"
    The value is a list of CpG sites: e.g. [(0, 35), (190, 212), (1055, ...)]

    We store this as a dict because it's easier to portably serialize to disk as JSON.

    NOTE: The window size is tricky -- it's set usually to the read size, but reads can be larger than the window size,
    since they can map with deletions, etc. So we need to be careful here.

    Args:
        reference_fasta (str): Path to the reference genome .fa file.
        window_size (int): Size of the window to use.
        cpg_sites_dict: A dict of CpG sites for each chromosome in the reference genome.
        verbose (bool): Print verbose output.
        skip_cache (bool): Ignore any cached files (slow!).

    Returns:
        windowed_cpg_sites_dict (dict): A dict of CpG sites for each chromosome in the reference genome.
        windowed_cpg_sites_dict_reverse (dict): A dict of each per-window CpG site. # TODO: Clarify this & rename var?
    """

    # Check if we have a cached version of our windowed cpg sites
    # TODO: Update caching to use a hash of the reference genome, window size, etc.
    windowed_cpg_sites_cache = (
        os.path.splitext(reference_fasta)[0] + ".cpg_windowed_sites.json"
    )
    windowed_cpg_sites_reverse_cache = (
        os.path.splitext(reference_fasta)[0] + ".cpg_windowed_sites_reverse.json"
    )

    if (
        os.path.exists(windowed_cpg_sites_cache)
        and os.path.exists(windowed_cpg_sites_reverse_cache)
        and not skip_cache
    ):
        if verbose:
            print(
                f"\tLoading windowed CpG sites from caches:\n\t\t{windowed_cpg_sites_cache}\n\t\t{windowed_cpg_sites_reverse_cache}"
            )
        with open(windowed_cpg_sites_cache, "r", encoding="utf-8") as f:
            windowed_cpg_sites_dict = json.load(f)
        with open(windowed_cpg_sites_reverse_cache, "r", encoding="utf-8") as f:
            # This wild object_hook is to convert the keys back to integers, since JSON only supports strings as keys
            windowed_cpg_sites_dict_reverse = json.load(
                f,
                object_hook=lambda d: {
                    int(k) if k.isdigit() else k: v for k, v in d.items()
                },
            )

        return windowed_cpg_sites_dict, windowed_cpg_sites_dict_reverse

    if verbose:
        print(
            f"\tNo cache found (or --skip-cache=True) at: {windowed_cpg_sites_cache} or {windowed_cpg_sites_reverse_cache}"
        )

    # Let's generate ranges (windows) of all CpGs within READ_SIZE of each other
    # This is to subsequently query our .bam/.sam files for reads containing CpGs
    # TODO: Shrink read size a little to account for trimming?
    assert (
        10 < window_size < 500
    ), "Read size is atypical, please double check (only validated for ~150 bp.)"

    if verbose:
        print(
            f"\n\tGenerating windowed CpG sites dict (window size = {window_size} bp.)\n"
        )

    # This is a dict of lists, where but each list contains a tuple of CpG ranges witin a window
    # Key: chromosome, e.g. "chr1"
    # Value: a list of tuples, e.g. [(0,35), (190,212), (1055,)]
    windowed_cpg_sites_dict = {}

    # And a reverse dict of dicts where chrom->window_start->[cpgs]
    windowed_cpg_sites_dict_reverse = {}

    # Loop over all chromosomes
    for chrom, cpg_sites in tqdm(cpg_sites_dict.items(), disable=not verbose):
        windowed_cpg_sites_dict[chrom] = []
        windowed_cpg_sites_dict_reverse[chrom] = {}
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
                or (cpg_sites[i + 1] - cpg_sites[i]) > window_size
            ):
                # We have a complete window
                window_end = cpg_sites[i]
                windowed_cpg_sites_dict[chrom].append((window_start, window_end))
                windowed_cpg_sites_dict_reverse[chrom][
                    window_start
                ] = temp_per_window_cpg_sites
                temp_per_window_cpg_sites = []
                window_start = None
                window_end = None

    # Save these to .json caches
    if verbose:
        print(
            f"\tSaving windowed CpG sites to caches:\n\t\t{windowed_cpg_sites_cache}\n\t\t{windowed_cpg_sites_reverse_cache}"
        )

    with open(windowed_cpg_sites_cache, "w", encoding="utf-8") as f:
        json.dump(windowed_cpg_sites_dict, f)
    with open(windowed_cpg_sites_reverse_cache, "w", encoding="utf-8") as f:
        json.dump(windowed_cpg_sites_dict_reverse, f)

    return windowed_cpg_sites_dict, windowed_cpg_sites_dict_reverse


# TODO: Object orient this input / simplify the input?
# TODO: Ingest chr_to_cpg_to_embedding_dict instead?
def embedding_to_genomic_position(
    total_cpg_sites: int,
    cpg_sites_dict: dict[str, list[int]],
    cpgs_per_chr_cumsum: np.array,  # type: ignore
    embedding_pos: int,
) -> tuple[str, int]:
    """
    Given an embedding position, return the chromosome and position.

    Parameters
    ----------
    embedding_pos : int
        The embedding position.

    Returns
    -------
    tuple:
        The chromosome and position, e.g. ("chr1", 12345)
    """
    assert embedding_pos >= 0 and embedding_pos < total_cpg_sites

    # Find the index of the first element in cpgs_per_chr_cumsum that is greater than or equal to the embedding position
    chr_index = np.searchsorted(cpgs_per_chr_cumsum, embedding_pos + 1, side="left")

    # Now we know the chromosome, but we need to find the position within the chromosome
    # If this is the first chromosome, the position is just the embedding position
    if chr_index == 0:
        return (
            CHROMOSOMES[chr_index],
            cpg_sites_dict[CHROMOSOMES[chr_index]][embedding_pos],
        )
    # Otherwise, subtract the length of the previous chromosomes from the embedding position
    embedding_pos -= cpgs_per_chr_cumsum[chr_index - 1]  # type: ignore
    return CHROMOSOMES[chr_index], cpg_sites_dict[CHROMOSOMES[chr_index]][embedding_pos]


# TODO: Object orient this input / simplify the input?
def genomic_position_to_embedding(
    chr_to_cpg_to_embedding_dict: dict,
    cpgs_per_chr_cumsum: np.array,  # type: ignore
    chrom: str,
    pos: int,
) -> int:
    """
    Given a genomic position, return the embedding position.

    Parameters
    ----------
    chr_to_cpg_to_embedding_dict : dict
    cpgs_per_chr_cumsum : np.array
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
    chr_index = CHROMOSOMES_DICT[chrom]
    # Find the index of the CpG site in the chromosome
    cpg_index = chr_to_cpg_to_embedding_dict[chrom][pos]
    # If this is the first chromosome, the embedding position is just the CpG index
    if chr_index == 0:
        return cpg_index
    # Otherwise, add the length of the previous chromosomes to the CpG index
    return cpg_index + cpgs_per_chr_cumsum[chr_index - 1]  # type: ignore


def extract_methylation_data_from_bam(
    input_bam: str,
    total_cpg_sites: int,
    chr_to_cpg_to_embedding_dict: dict,
    cpgs_per_chr_cumsum: np.array,  # type: ignore
    windowed_cpg_sites_dict: dict,
    windowed_cpg_sites_dict_reverse: dict,
    quality_limit: int = 0,
    verbose: bool = False,
    debug: bool = False,
) -> scipy.sparse.coo_matrix:
    """
    Extract methylation data from a .bam file.

    Args:

    """
    input_bam_object = pysam.AlignmentFile(
        input_bam, "rb", require_index=True, threads=1
    )

    if verbose:
        print(f"\tTotal reads: {input_bam_object.mapped:,}\n")
    if debug:
        debug_read_name_to_row_number = {}

    # Temporary storage for loading into a COO matrix
    # For a COO, we want three lists:
    # row: the read number (we'll store a read name -> ID dict perhaps?)
    # column: cpg #
    # data: methylation state (1 [methylated], 0 [unmethylated], and -1 [no data/snv/indel])
    read_number = 0

    coo_row = []  # Read number
    coo_col = []  # CpG number (embedding)
    coo_data = []  # Methylation state

    # This is slow, but we only run it once and store the results for later
    for chrom, windowed_cpgs in tqdm(
        windowed_cpg_sites_dict.items(), disable=not verbose
    ):
        for start_pos, stop_pos in windowed_cpgs:
            cpgs_within_window = windowed_cpg_sites_dict_reverse[chrom][start_pos]
            if debug:
                print(f"Window Position: {start_pos} - {stop_pos}")
                print(f"\tCovered CpGs: {cpgs_within_window}")
            # This returns an AlignmentSegment object from PySam

            # TODO: Think carefully here -- Can a read appear in more than one segment?
            # Maybe if the read is longer than the window? (Or significant indels / weird fusion alignment?)
            for aligned_segment in input_bam_object.fetch(
                contig=chrom, start=start_pos, end=stop_pos
            ):
                if aligned_segment.mapping_quality < quality_limit:
                    continue
                if aligned_segment.is_duplicate:
                    continue
                if aligned_segment.is_qcfail:
                    continue
                if aligned_segment.is_secondary:
                    continue

                if debug:
                    print(aligned_segment.query_name)
                    # Validity tests
                    assert not aligned_segment.is_unmapped
                    assert aligned_segment.is_supplementary is False
                    assert aligned_segment.reference_start <= stop_pos
                    assert aligned_segment.reference_end >= start_pos

                    # Ensure alignment methylation tags exist
                    assert aligned_segment.has_tag(
                        "MD"
                    )  # Location of mismatches (methylation)
                    assert aligned_segment.has_tag(
                        "YD"
                    )  # Bisulfite conversion strand label (f: OT/CTOT C->T or r: OB/CTOB G->A)

                    # Ensure each read is only seen once
                    assert (
                        aligned_segment.query_name not in debug_read_name_to_row_number
                    )
                    debug_read_name_to_row_number[
                        aligned_segment.query_name  # type: ignore
                        + ("_1" if aligned_segment.is_read1 else "_2")
                    ] = read_number

                # TODO: We ignore paired/unpaired read status for now. Should we treat paired reads / overlapping reads differently?

                ## Read tags and ensure we're on the correct bisulfite-converted strand
                bisulfite_parent_strand_is_reverse = None
                if aligned_segment.has_tag("YD"):  # Biscuit tag
                    yd_tag = aligned_segment.get_tag("YD")
                    if yd_tag == "f":  # Forward = C→T
                        # This read derives from OT/CTOT strand: C->T substitutions matter (C = methylated, T = unmethylated),
                        bisulfite_parent_strand_is_reverse = False
                    elif yd_tag == "r":  # Reverse = G→A
                        # This read derives from the OB/CTOB strand: G->A substitutions matter (G = methylated, A = unmethylated)
                        bisulfite_parent_strand_is_reverse = True
                elif aligned_segment.has_tag("XB"):  # gem3 / blueprint tag
                    xb_tag = aligned_segment.get_tag(
                        "XB"
                    )  # XB:C = Forward / Reference was CG
                    if xb_tag == "C":
                        bisulfite_parent_strand_is_reverse = False
                    elif xb_tag == "G":  # XB:G = Reverse / Reference was GA
                        bisulfite_parent_strand_is_reverse = True

                # We have paired-end reads; one half should (the "parent strand") has the methylation data.
                # The other half (the "daughter strand") was the complement created by PCR, which we don't care about.
                if bisulfite_parent_strand_is_reverse != aligned_segment.is_reverse:
                    # Skip if we're not on the bisulfite-converted parent strand.
                    if debug:
                        print("\tNot on methylated strand, ignoring.")
                    continue

                # get_aligned_pairs returns a list of tuples of (read_pos, ref_pos)
                # We filter this to only include the specific CpG sites from above
                this_segment_cpgs = [
                    e
                    for e in aligned_segment.get_aligned_pairs(matches_only=True)
                    if e[1] + 1 in cpgs_within_window
                ]

                # Ok we're on the same strand as the methylation (right?)
                # Let's compare the possible CpGs in this interval to the reference and note status
                #   A methylated C will be *unchanged* and read as C (pair G)
                #   An unmethylated C will be *changed* and read as T (pair A)
                for query_pos, ref_pos in this_segment_cpgs:
                    query_base = aligned_segment.query_sequence[query_pos]  # type: ignore
                    # query_base_raw = aligned_segment.get_forward_sequence()[query_pos] # raw off sequencer
                    # query_base_no_offset = aligned_segment.query_alignment_sequence[query_pos] # this needs to be offset by the soft clip

                    # Store the read # in our sparse array
                    coo_row.append(read_number)

                    # Store the CpG site in our sparse array
                    # TODO: Object orient these inputs? -- lots of bad inheritence style here
                    coo_col.append(
                        genomic_position_to_embedding(
                            chr_to_cpg_to_embedding_dict,
                            cpgs_per_chr_cumsum,
                            chrom,
                            ref_pos + 1,
                        )
                    )

                    if query_base == "C":
                        # Methylated
                        coo_data.append(1)
                        if debug:
                            print(
                                f"\t{query_pos} {ref_pos} C->{query_base} [Methylated]"
                            )
                    elif query_base == "T":
                        coo_data.append(0)
                        # Unmethylated
                        if debug:
                            print(
                                f"\t{query_pos} {ref_pos} C->{query_base} [Unmethylated]"
                            )
                    else:
                        coo_data.append(-1)  # or just 0?
                        if debug:
                            print(
                                f"\t{query_pos} {ref_pos} C->{query_base} [Unknown! SNV? Indel?]"
                            )

                read_number += 1

                if debug:
                    print("************************************************\n")

                # query_bp = aligned_segment.query_sequence[pileupread.query_position]
                # reference_bp = aligned_segment.get_reference_sequence()[aligned_segment.reference_start - pileupcolumn.reference_pos].upper()

    ## IIRC there's still a critical edge here, where sometimes we raise ValueError('row index exceeds matrix dimensions')

    if debug:
        print("Debug info for coo_matrix dimensions:")
        print(f"\tcoo_row: {len(coo_row):,}")
        print(f"\tcoo row max: {max(coo_row):,}")
        print(f"\tcoo_col: {len(coo_col):,}")
        print(f"\tcoo col max: {max(coo_col):,}")
        print(f"\tcoo_data: {len(coo_data):,}")
        print(f"\tlen(read_name_to_row_number): {len(debug_read_name_to_row_number):,}")
        print(f"\ttotal_cpg_sites: {total_cpg_sites:,}")

    # The size of the coo_matrix is:
    #   Number of rows = number of reads that pass our filters
    #   Number of columns = number of CpG sites

    return scipy.sparse.coo_matrix(
        (coo_data, (coo_row, coo_col)), shape=(read_number, total_cpg_sites)
    )

    # return scipy.sparse.coo_matrix((coo_data, (coo_row, coo_col)), shape=(len(read_name_to_row_number) + 1, total_cpg_sites))
