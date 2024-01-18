"""Core functions for bam2tensor."""

# Third party modules
import scipy.sparse
import pysam

from tqdm import tqdm
from bam2tensor.embedding import GenomeMethylationEmbedding


def extract_methylation_data_from_bam(
    input_bam: str,
    genome_methylation_embedding: GenomeMethylationEmbedding,
    quality_limit: int = 20,
    verbose: bool = False,
    debug: bool = False,
) -> scipy.sparse.coo_matrix:
    """
    Extract methylation data from a .bam file.

    Args
    -------
        input_bam: Path to the input .bam file.
        quality_limit: Minimum mapping quality to include.
        genome_methylation_embedding: A GenomeMethylationEmbedding object.
        verbose: Print verbose output.
        debug: Print debug output.

    Returns
    -------
        A scipy.sparse.coo_matrix of the methylation data.

    Raises
    -------
        FileNotFoundError: If the input .bam file is not found.
    """
    try:
        input_bam_object = pysam.AlignmentFile(  # type: ignore # pylint: disable=no-member
            input_bam, "rb", require_index=True, threads=1
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Index missing for bam file?: {input_bam}") from exc

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
        genome_methylation_embedding.windowed_cpg_sites_dict.items(),
        disable=not verbose,
    ):
        for start_pos, stop_pos in windowed_cpgs:
            cpgs_within_window = (
                genome_methylation_embedding.windowed_cpg_sites_dict_reverse[chrom][
                    start_pos
                ]
            )
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
                    print(f"Query: {aligned_segment.query_name}")
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
                    ), "Read seen twice!"
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
                        genome_methylation_embedding.genomic_position_to_embedding(
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
        print(f"\ttotal_cpg_sites: {genome_methylation_embedding.total_cpg_sites:,}")

    # Generate the sparse matrix
    sparse_matrix = scipy.sparse.coo_matrix(
        (coo_data, (coo_row, coo_col)),
        shape=(read_number, genome_methylation_embedding.total_cpg_sites),
    )

    # Validate a dimension of the coo_matrix, which should be:
    #   Number of rows = number of reads that pass our filters
    #   Number of columns = number of CpG sites
    assert sparse_matrix.shape[1] == genome_methylation_embedding.total_cpg_sites

    return sparse_matrix

    # return scipy.sparse.coo_matrix((coo_data, (coo_row, coo_col)), shape=(len(read_name_to_row_number) + 1, total_cpg_sites))
