"""Core methylation extraction functions for bam2tensor.

This module contains the main function for extracting methylation data from
BAM files: extract_methylation_data_from_bam(). This function reads aligned
bisulfite-sequencing reads and generates a sparse matrix representation of
methylation states at each CpG site.

The extraction process:
    1. Iterates through each chromosome in the genome embedding
    2. For each aligned read that passes quality filters:
       a. Identifies which CpG sites fall within the read's alignment
       b. Determines the bisulfite strand (forward or reverse)
       c. Extracts methylation state at each covered CpG
    3. Builds a sparse COO matrix with the results

Methylation State Encoding:
    - 1: Methylated (cytosine remained as C after bisulfite treatment)
    - 0: Unmethylated (cytosine converted to T by bisulfite treatment)
    - -1: No data (indel, SNV, or other non-C/T base at CpG position)

Supported Aligners:
    The module supports BAM files from aligners that provide strand information:
    - Biscuit: Uses the YD tag (f=forward, r=reverse)
    - gem3/Blueprint: Uses the XB tag (C=forward, G=reverse)

Example:
    >>> # xdoctest: +SKIP
    >>> from bam2tensor.embedding import GenomeMethylationEmbedding
    >>> from bam2tensor.functions import extract_methylation_data_from_bam
    >>>
    >>> # Create genome embedding
    >>> embedding = GenomeMethylationEmbedding(
    ...     genome_name="hg38",
    ...     expected_chromosomes=["chr1", "chr2"],
    ...     fasta_source="reference.fa",
    ... )
    >>>
    >>> # Extract methylation data
    >>> matrix = extract_methylation_data_from_bam(
    ...     input_bam="sample.bam",
    ...     genome_methylation_embedding=embedding,
    ...     quality_limit=20,
    ... )
    >>>
    >>> print(f"Extracted {matrix.shape[0]} reads, {matrix.nnz} data points")
"""

import scipy.sparse
import pysam
import bisect

from tqdm import tqdm
from bam2tensor.embedding import GenomeMethylationEmbedding


def extract_methylation_data_from_bam(
    input_bam: str,
    genome_methylation_embedding: GenomeMethylationEmbedding,
    quality_limit: int = 20,
    verbose: bool = False,
    debug: bool = False,
) -> scipy.sparse.coo_matrix:
    """Extract read-level CpG methylation data from a BAM file.

    Parses a bisulfite-sequencing BAM file and extracts methylation states
    at each CpG site covered by aligned reads. Returns a sparse matrix where
    each row represents a unique read and each column represents a CpG site.

    The function applies several filters to reads:
        - Mapping quality must be >= quality_limit (default 20)
        - Duplicate reads are excluded
        - QC-failed reads are excluded
        - Secondary and supplementary alignments are excluded
        - Only reads on the bisulfite-converted parent strand are processed

    For each qualifying read, the function:
        1. Uses binary search to find CpG sites within the read's alignment
        2. Checks the bisulfite strand tag (YD or XB) to determine strand
        3. Extracts the base at each CpG position from the read sequence
        4. Records methylation state: C=methylated(1), T=unmethylated(0), other=-1

    Args:
        input_bam: Path to the input BAM file. The file must be indexed
            (have a corresponding .bam.bai file).
        genome_methylation_embedding: A GenomeMethylationEmbedding object
            that defines the CpG site positions and provides coordinate
            conversion methods.
        quality_limit: Minimum mapping quality (MAPQ) threshold for reads.
            Reads with MAPQ below this value are skipped. Default is 20,
            which excludes reads mapping to multiple locations equally well.
        verbose: If True, display a progress bar and print the total read
            count. Useful for monitoring progress on large files.
        debug: If True, enable extensive validation and debug output.
            Prints per-read information and validates that each read is
            only processed once. Significantly slower.

    Returns:
        A scipy.sparse.coo_matrix with shape (n_reads, n_cpg_sites) where:
            - n_reads is the number of reads that passed filters and covered
              at least one CpG site
            - n_cpg_sites is genome_methylation_embedding.total_cpg_sites
            - Values are: 1 (methylated), 0 (unmethylated), -1 (no data)

        The matrix uses COO format for efficient construction. Convert to
        CSR (tocsr()) for row slicing or CSC (tocsc()) for column slicing.

    Raises:
        FileNotFoundError: If the BAM file index (.bam.bai) is missing.
            The BAM file must be indexed with `samtools index`.

    Example:
        >>> # xdoctest: +SKIP
        >>> from bam2tensor.embedding import GenomeMethylationEmbedding
        >>> from bam2tensor.functions import extract_methylation_data_from_bam
        >>> import scipy.sparse
        >>>
        >>> # Create embedding (or load from cache)
        >>> embedding = GenomeMethylationEmbedding(
        ...     genome_name="hg38",
        ...     expected_chromosomes=["chr1", "chr22"],
        ...     fasta_source="GRCh38.fa",
        ... )
        >>>
        >>> # Extract methylation data
        >>> matrix = extract_methylation_data_from_bam(
        ...     input_bam="sample.bam",
        ...     genome_methylation_embedding=embedding,
        ...     quality_limit=30,  # Stricter quality filter
        ...     verbose=True,
        ... )
        >>>
        >>> # Analyze results
        >>> print(f"Reads with CpG data: {matrix.shape[0]:,}")
        >>> print(f"Total CpG sites: {matrix.shape[1]:,}")
        >>> print(f"Data points: {matrix.nnz:,}")
        >>>
        >>> # Save to file
        >>> scipy.sparse.save_npz("sample.methylation.npz", matrix)

    Note:
        The function processes chromosomes in the order they appear in
        genome_methylation_embedding.cpg_sites_dict. Reads are numbered
        sequentially as they are encountered across all chromosomes.

    See Also:
        GenomeMethylationEmbedding: For creating the genome embedding.
        scipy.sparse.coo_matrix: Documentation on the output format.
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
    for chrom, cpg_sites in tqdm(
        genome_methylation_embedding.cpg_sites_dict.items(),
        disable=not verbose,
    ):
        # Ensure this chromosome exists in the BAM file
        try:
            iter_reads = input_bam_object.fetch(contig=chrom)
        except ValueError:
            # Chromosome not in BAM
            if verbose:
                tqdm.write(f"Chromosome {chrom} not found in BAM file, skipping.")
            continue

        for aligned_segment in iter_reads:
            if aligned_segment.mapping_quality < quality_limit:
                continue
            if aligned_segment.is_duplicate:
                continue
            if aligned_segment.is_qcfail:
                continue
            if aligned_segment.is_secondary:
                continue
            if aligned_segment.is_supplementary:
                continue

            # Use bisect to find CpGs covered by this read
            # aligned_segment.reference_start is 0-based inclusive
            # aligned_segment.reference_end is 0-based exclusive
            # cpg_sites are 1-based

            # We want CpGs where: read_start < cpg_pos <= read_end (in 1-based terms: read_start+1 <= cpg <= read_end)
            start_idx = bisect.bisect_left(
                cpg_sites, aligned_segment.reference_start + 1
            )
            end_idx = bisect.bisect_right(cpg_sites, aligned_segment.reference_end)

            # If no CpGs in this read, skip
            if start_idx >= end_idx:
                continue

            cpgs_within_read = cpg_sites[start_idx:end_idx]
            cpgs_within_read_set = set(cpgs_within_read)

            if debug:
                print(f"Query: {aligned_segment.query_name}")
                # Validity tests
                assert not aligned_segment.is_unmapped
                assert aligned_segment.is_supplementary is False

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
                if e[1] + 1 in cpgs_within_read_set
            ]

            # If no CpGs covered (after filtering for matches only), skip
            if not this_segment_cpgs:
                continue

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
                        print(f"\t{query_pos} {ref_pos} C->{query_base} [Methylated]")
                elif query_base == "T":
                    coo_data.append(0)
                    # Unmethylated
                    if debug:
                        print(f"\t{query_pos} {ref_pos} C->{query_base} [Unmethylated]")
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
