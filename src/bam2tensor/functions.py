"""Core methylation extraction functions for bam2tensor.

This module contains the main function for extracting methylation data from
BAM files: extract_methylation_data_from_bam(). This function reads aligned
bisulfite-sequencing (BS-seq) and enzymatic methylation sequencing (EM-seq)
reads and generates a sparse matrix representation of methylation states at
each CpG site.

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
    - Bismark: Uses the XM tag (Z=methylated CpG, z=unmethylated CpG)
    - Biscuit: Uses the YD tag (f=forward, r=reverse)
    - bwameth: Uses the YD tag (f=forward, r=reverse), identical to Biscuit
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
    >>> result = extract_methylation_data_from_bam(
    ...     input_bam="sample.bam",
    ...     genome_methylation_embedding=embedding,
    ...     quality_limit=20,
    ... )
    >>>
    >>> print(f"Extracted {result.matrix.shape[0]} reads, {result.matrix.nnz} data points")
"""

from typing import NamedTuple

import numpy as np
import scipy.sparse
import pysam
import bisect

from tqdm import tqdm
from bam2tensor.embedding import GenomeMethylationEmbedding


class ExtractionResult(NamedTuple):
    """Result of methylation extraction from a BAM file.

    Attributes:
        matrix: Sparse COO matrix of shape (n_reads, n_cpg_sites) with
            methylation states: 1 (methylated), 0 (unmethylated), -1
            (no data).
        tlen: 1-D numpy array of shape (n_reads,) containing the signed
            template length (TLEN from BAM) for each read. 0 for
            single-end reads or reads with unmapped mates.
    """

    matrix: scipy.sparse.coo_matrix
    tlen: np.ndarray


# BAM flag bits for reads to skip: duplicate (0x400), qcfail (0x200),
# secondary (0x100), supplementary (0x800).
_SKIP_FLAGS = 0x400 | 0x200 | 0x100 | 0x800


def count_non_cpg_retained_xm(xm_tag: str) -> int:
    """Count retained non-CpG cytosines in a Bismark XM methylation string.

    Bismark's ``XM`` tag encodes per-base methylation context. Uppercase
    letters indicate a cytosine that remained as ``C`` in the read
    (i.e., was *not* converted by bisulfite/EM-seq treatment). ``H``,
    ``X`` and ``U`` correspond to retained cytosines in CHH, CHG and
    unknown-context positions respectively. A high count of these on a
    single read is a strong signal of incomplete conversion.

    Args:
        xm_tag: The value of a read's Bismark ``XM`` tag.

    Returns:
        The count of ``H``, ``X`` and ``U`` characters in ``xm_tag``.

    Example:
        >>> count_non_cpg_retained_xm("..Z..hhh..HHH..z..")
        3
    """
    return xm_tag.count("H") + xm_tag.count("X") + xm_tag.count("U")


def count_non_cpg_retained_reference(
    aligned_segment: pysam.AlignedSegment,
    is_reverse_parent_strand: bool,
) -> int:
    """Count retained non-CpG bases validated against the reference.

    For a correctly bisulfite- or EM-seq-converted read, every
    non-CpG cytosine on the parent strand should have been converted.
    On the forward-parent strand that means every non-CpG ``C`` in the
    reference should appear as ``T`` in the read; on the reverse-parent
    strand every non-CpG ``G`` should appear as ``A``. Positions where
    the read still carries the unconverted base *and* the reference
    genuinely has a ``C``/``G`` (i.e., the mismatch is not a SNP) count
    as retained.

    This is a faithful port of the logic in
    ``nebiolabs/mark-nonconverted-reads``, re-using the read's existing
    ``MD`` tag (via :py:meth:`pysam.AlignedSegment.get_aligned_pairs`
    with ``with_seq=True``) instead of requiring a separate reference
    FASTA.

    Args:
        aligned_segment: A pysam aligned read. Must carry an ``MD``
            tag; BAMs produced by Bismark, Biscuit, bwameth and gem3
            all set this tag by default.
        is_reverse_parent_strand: ``True`` if the read derives from the
            reverse (OB/CTOB) bisulfite parent strand, ``False`` for
            the forward (OT/CTOT) strand.

    Returns:
        The number of reference-validated retained non-CpG
        cytosines (or guanines, for the reverse parent strand). Returns
        ``0`` when the read has no sequence or no ``MD`` tag is present.
    """
    if aligned_segment.query_sequence is None:
        return 0

    try:
        pairs = aligned_segment.get_aligned_pairs(matches_only=True, with_seq=True)
    except ValueError:
        # MD tag missing — cannot validate against reference.
        return 0

    # Map ref_pos → reference base (uppercase) for CpG-context lookup.
    # matches_only=True guarantees query_pos, ref_pos, ref_base are all set.
    ref_pos_to_base = {rpos: rb.upper() for _, rpos, rb in pairs}

    # On match, pysam returns ref_base uppercase (query matches ref).
    # On mismatch (SNP), it returns lowercase. We only care about matches
    # where ref is C/G — those are genuine retained, non-converted bases.
    target = "G" if is_reverse_parent_strand else "C"

    count = 0
    for _, rpos, ref_base in pairs:
        if ref_base != target:
            # Not a match, or ref is not C/G. This rejects SNPs (lowercase)
            # and converted positions (read has T/A, match has different base).
            continue
        # Exclude CpG context: on forward strand, next ref base == G;
        # on reverse strand, previous ref base == C.
        if is_reverse_parent_strand:
            if ref_pos_to_base.get(rpos - 1) == "C":
                continue
        else:
            if ref_pos_to_base.get(rpos + 1) == "G":
                continue
        count += 1

    return count


def is_em_overconversion_read(
    read_cpg_states: list[int],
    min_cpgs: int,
) -> bool:
    """Identify reads flagged as EM-seq fragment-level over-conversion.

    Loyfer et al. (bioRxiv 2026.03.24.713040) report that EM-seq
    produces a reproducible ~1–2.5% of multi-CpG fragments that appear
    fully unmethylated across every covered CpG, driven by failed TET
    protection and subsequent APOBEC hyper-conversion of an entire
    molecule. At constitutively methylated loci these reads are purely
    technical. Without a per-region methylation prior, the simplest
    correction consistent with their observation is: drop reads whose
    covered CpGs are all called unmethylated *and* cover at least
    ``min_cpgs`` sites (the paper's Fig. 1C regime where the artifact
    diverges clearly from WGBS).

    This heuristic also drops genuinely fully-unmethylated biological
    fragments, so callers should opt in only when the downstream
    application can tolerate that trade-off.

    Args:
        read_cpg_states: Per-CpG methylation state values for a single
            read, in column-order, using the bam2tensor encoding
            (``1``=methylated, ``0``=unmethylated, ``-1``=no data).
        min_cpgs: Minimum number of covered CpGs required to apply the
            filter. Reads with fewer covered CpGs are never flagged.

    Returns:
        ``True`` when the read has at least ``min_cpgs`` covered CpGs
        and every covered CpG is called unmethylated (value ``0``).
        ``-1`` (no-data) values do not count as unmethylated.

    Example:
        >>> is_em_overconversion_read([0, 0, 0], min_cpgs=3)
        True
        >>> is_em_overconversion_read([0, 0, 1], min_cpgs=3)
        False
        >>> is_em_overconversion_read([0, 0], min_cpgs=3)
        False
    """
    if len(read_cpg_states) < min_cpgs:
        return False
    return all(state == 0 for state in read_cpg_states)


def detect_aligner(input_bam: str, sample_size: int = 1000) -> str:
    """Detect the aligner used to produce a BAM file by checking read tags.

    Samples up to ``sample_size`` reads from the BAM file and checks for
    aligner-specific tags: ``XM`` (Bismark), ``YD`` (Biscuit/bwameth),
    or ``XB`` (gem3/Blueprint).

    Args:
        input_bam: Path to an indexed BAM file.
        sample_size: Maximum number of reads to examine. Only primary,
            non-duplicate, non-QC-failed reads are considered.

    Returns:
        A human-readable string identifying the detected aligner, such as
        ``"Bismark (XM tag)"`` or ``"Unknown"``.
    """
    try:
        bam = pysam.AlignmentFile(input_bam, "rb", require_index=True)  # type: ignore
    except (FileNotFoundError, ValueError):
        return "Unknown (could not open BAM)"

    checked = 0
    for read in bam.fetch():
        if read.flag & _SKIP_FLAGS:
            continue
        if read.has_tag("XM"):
            bam.close()
            return "Bismark (XM tag)"
        if read.has_tag("YD"):
            bam.close()
            return "Biscuit/bwameth (YD tag)"
        if read.has_tag("XB"):
            bam.close()
            return "gem3/Blueprint (XB tag)"
        checked += 1
        if checked >= sample_size:
            break

    bam.close()
    return "Unknown (no XM/YD/XB tags found)"


def check_chromosome_overlap(
    bam_references: tuple[str, ...] | list[str],
    embedding_chromosomes: list[str],
) -> None:
    """Check that BAM and embedding chromosome names overlap.

    Detects the common mistake where a BAM file uses one chromosome naming
    convention (e.g., ``1``, ``2``, ``3`` from Ensembl) while the reference
    embedding uses another (e.g., ``chr1``, ``chr2``, ``chr3`` from UCSC).
    If no overlap is found but adding or stripping a ``chr`` prefix would
    create overlap, raises a clear error with guidance.

    Args:
        bam_references: Tuple or list of chromosome/contig names from the
            BAM file header (e.g., from ``pysam.AlignmentFile.references``).
        embedding_chromosomes: List of chromosome names from the genome
            embedding (i.e., the expected chromosomes).

    Raises:
        ValueError: If there is no overlap between BAM and embedding
            chromosome names, with a diagnostic message indicating
            whether a ``chr`` prefix mismatch is the likely cause.
    """
    bam_set = set(bam_references)
    emb_set = set(embedding_chromosomes)

    overlap = bam_set & emb_set
    if overlap:
        return

    # Try adding "chr" prefix to BAM references
    bam_with_chr = {f"chr{name}" for name in bam_set}
    if bam_with_chr & emb_set:
        bam_example = next(iter(bam_set - emb_set))
        emb_example = f"chr{bam_example}"
        raise ValueError(
            f"Chromosome name mismatch: BAM file uses '{bam_example}' "
            f"but the reference embedding expects '{emb_example}'. "
            f"Your BAM file appears to use Ensembl-style names (no 'chr' prefix) "
            f"while the embedding uses UCSC-style names ('chr' prefix). "
            f"Use --expected-chromosomes with matching names "
            f"(e.g., --expected-chromosomes '1,2,3,...')."
        )

    # Try stripping "chr" prefix from BAM references
    bam_stripped = {name[3:] for name in bam_set if name.startswith("chr")}
    if bam_stripped & emb_set:
        bam_example = next(
            name for name in bam_set if name.startswith("chr") and name[3:] in emb_set
        )
        emb_example = bam_example[3:]
        raise ValueError(
            f"Chromosome name mismatch: BAM file uses '{bam_example}' "
            f"but the reference embedding expects '{emb_example}'. "
            f"Your BAM file uses UCSC-style names ('chr' prefix) "
            f"while the embedding uses Ensembl-style names (no 'chr' prefix). "
            f"Use --expected-chromosomes with matching names "
            f"(e.g., --expected-chromosomes 'chr1,chr2,chr3,...')."
        )

    # No overlap even with prefix changes -- completely different names
    bam_sample = sorted(bam_set)[:3]
    emb_sample = sorted(emb_set)[:3]
    raise ValueError(
        f"No overlapping chromosome names between BAM file and reference "
        f"embedding. BAM contains: {bam_sample}... "
        f"Embedding expects: {emb_sample}... "
        f"Ensure the reference FASTA matches the genome used for alignment "
        f"and that --expected-chromosomes is set correctly."
    )


def extract_methylation_data_from_bam(
    input_bam: str,
    genome_methylation_embedding: GenomeMethylationEmbedding,
    quality_limit: int = 20,
    filter_non_converted: bool = False,
    non_converted_threshold: int = 3,
    filter_em_overconversion: bool = False,
    em_overconversion_min_cpgs: int = 3,
    verbose: bool = False,
    debug: bool = False,
) -> ExtractionResult:
    """Extract read-level CpG methylation data from a BAM file.

    Parses a bisulfite-sequencing or EM-seq BAM file and extracts methylation
    states at each CpG site covered by aligned reads. Returns a sparse matrix
    where each row represents a unique read and each column represents a CpG
    site.

    The function applies several filters to reads:
        - Mapping quality must be >= quality_limit (default 20)
        - Duplicate reads are excluded
        - QC-failed reads are excluded
        - Secondary and supplementary alignments are excluded
        - For Biscuit/bwameth/gem3: only parent-strand reads are processed
        - For Bismark: all reads are processed (XM tag has pre-resolved calls)

    Two additional, opt-in per-read filters are available:
        - Non-converted reads (``filter_non_converted``): drops reads with
          too many retained non-CpG cytosines, the hallmark of incomplete
          bisulfite/EM-seq conversion. Ports the logic of
          ``nebiolabs/mark-nonconverted-reads``.
        - EM-seq fragment-level over-conversion
          (``filter_em_overconversion``): drops reads whose covered CpGs
          are all called unmethylated, a heuristic for the EM-seq
          artifact described by Loyfer et al.
          (bioRxiv 2026.03.24.713040).

    Two extraction paths are supported, detected automatically per-read:

    **Bismark path** (XM tag present):
        1. Uses binary search to find CpG sites within the read
        2. Reads the XM methylation call string directly
        3. Records: Z=methylated(1), z=unmethylated(0), other CpG=-1

    **Biscuit/bwameth/gem3 path** (YD or XB tag):
        1. Checks the strand tag to filter daughter-strand reads
        2. Uses binary search to find CpG sites within the read
        3. Extracts the base at each CpG position from the read sequence
        4. Records: C=methylated(1), T=unmethylated(0), other=-1

    Args:
        input_bam: Path to the input BAM file. The file must be indexed
            (have a corresponding .bam.bai file).
        genome_methylation_embedding: A GenomeMethylationEmbedding object
            that defines the CpG site positions and provides coordinate
            conversion methods.
        quality_limit: Minimum mapping quality (MAPQ) threshold for reads.
            Reads with MAPQ below this value are skipped. Default is 20,
            which excludes reads mapping to multiple locations equally well.
        filter_non_converted: If True, drop reads that carry at least
            ``non_converted_threshold`` retained non-CpG cytosines, a
            signature of incomplete bisulfite/EM-seq conversion. Default
            False.
        non_converted_threshold: Minimum count of retained non-CpG
            cytosines required for the non-converted filter to drop a
            read. Matches the NEB ``mark-nonconverted-reads`` default of
            3.
        filter_em_overconversion: If True, drop reads whose covered CpGs
            are all called unmethylated and cover at least
            ``em_overconversion_min_cpgs`` sites — the Loyfer et al.
            EM-seq fragment-level over-conversion heuristic. Default
            False.
        em_overconversion_min_cpgs: Minimum covered CpG count required
            before the over-conversion filter will drop a read. Matches
            the regime in Loyfer et al. Fig. 1C where the EM-seq
            artifact is clearly separable from WGBS.
        verbose: If True, display a progress bar and print the total read
            count. Useful for monitoring progress on large files.
        debug: If True, enable extensive validation and debug output.
            Prints per-read information and validates that each read is
            only processed once. Significantly slower.

    Returns:
        An ExtractionResult named tuple with two fields:

        - **matrix**: A scipy.sparse.coo_matrix with shape
          (n_reads, n_cpg_sites) where n_reads is the number of reads
          that passed filters and covered at least one CpG site,
          n_cpg_sites is genome_methylation_embedding.total_cpg_sites,
          and values are: 1 (methylated), 0 (unmethylated), -1 (no data).
          The matrix uses COO format for efficient construction; convert
          to CSR (tocsr()) for row slicing or CSC (tocsc()) for column
          slicing.
        - **tlen**: A 1-D numpy int32 array of shape (n_reads,) containing
          the signed template length (BAM TLEN field) for each read.
          Values are 0 for single-end reads or reads with unmapped mates.

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
        >>> result = extract_methylation_data_from_bam(
        ...     input_bam="sample.bam",
        ...     genome_methylation_embedding=embedding,
        ...     quality_limit=30,  # Stricter quality filter
        ...     verbose=True,
        ... )
        >>>
        >>> # Analyze results
        >>> print(f"Reads with CpG data: {result.matrix.shape[0]:,}")
        >>> print(f"Total CpG sites: {result.matrix.shape[1]:,}")
        >>> print(f"Data points: {result.matrix.nnz:,}")
        >>>
        >>> # Save to file
        >>> scipy.sparse.save_npz("sample.methylation.npz", result.matrix)

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
            input_bam, "rb", require_index=True, threads=4
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Index missing for bam file?: {input_bam}") from exc

    # Check for chromosome name mismatches (e.g. chr1 vs 1)
    check_chromosome_overlap(
        bam_references=input_bam_object.references,
        embedding_chromosomes=list(genome_methylation_embedding.cpg_sites_dict.keys()),
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
    tlen_list: list[int] = []  # Template length (TLEN) per read

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

            # Single bitwise check replaces 4 separate flag checks:
            # duplicate (0x400), qcfail (0x200), secondary (0x100), supplementary (0x800)
            if aligned_segment.flag & _SKIP_FLAGS:
                continue

            # Per-read buffers. We only flush these into the global
            # coo_* arrays once the read passes all filters (including
            # the post-CpG EM over-conversion filter).
            read_cpg_cols: list[int] = []
            read_cpg_data: list[int] = []

            # ============================================================
            # Bismark path: XM tag contains pre-resolved methylation calls.
            # No strand filtering needed — Bismark already resolved strand
            # and encoded calls as Z (methylated CpG) / z (unmethylated CpG).
            # Both reads in a pair carry valid XM tags.
            # ============================================================
            if aligned_segment.has_tag("XM"):
                xm_tag: str = aligned_segment.get_tag("XM")  # type: ignore[assignment]

                # Non-converted filter (Bismark): XM tag already encodes
                # retained non-CpG cytosines as H/X/U. Apply before any
                # CpG work so we bail as early as possible.
                if filter_non_converted:
                    if count_non_cpg_retained_xm(xm_tag) >= non_converted_threshold:
                        continue

                # Find CpG sites covered by this read
                start_idx = bisect.bisect_left(
                    cpg_sites, aligned_segment.reference_start + 1
                )
                end_idx = bisect.bisect_right(cpg_sites, aligned_segment.reference_end)
                if start_idx >= end_idx:
                    continue

                cpgs_within_read_set = set(cpg_sites[start_idx:end_idx])

                # Map read positions to reference positions, filter to CpG sites
                this_segment_cpgs = [
                    e
                    for e in aligned_segment.get_aligned_pairs(matches_only=True)
                    if e[1] + 1 in cpgs_within_read_set
                ]
                if not this_segment_cpgs:
                    continue

                if debug:
                    print(f"Query (Bismark): {aligned_segment.query_name}")

                for query_pos, ref_pos in this_segment_cpgs:
                    # Bounds check: XM tag should match query length, but be defensive
                    if query_pos >= len(xm_tag):
                        continue

                    xm_char = xm_tag[query_pos]
                    if xm_char == "Z":
                        read_cpg_data.append(1)  # Methylated CpG
                    elif xm_char == "z":
                        read_cpg_data.append(0)  # Unmethylated CpG
                    else:
                        # Non-CpG context at a CpG site (shouldn't happen
                        # normally, but possible with edge-case alignments)
                        read_cpg_data.append(-1)

                    read_cpg_cols.append(
                        genome_methylation_embedding.genomic_position_to_embedding(
                            chrom,
                            ref_pos + 1,
                        )
                    )

                    if debug:
                        print(f"\t{query_pos} {ref_pos} XM={xm_char}")

                if not read_cpg_data:
                    continue

                if filter_em_overconversion and is_em_overconversion_read(
                    read_cpg_data, em_overconversion_min_cpgs
                ):
                    if debug:
                        print("\tEM over-conversion filter: dropping read.")
                    continue

                if debug:
                    read_key = aligned_segment.query_name + (  # type: ignore
                        "_1" if aligned_segment.is_read1 else "_2"
                    )
                    assert (
                        read_key not in debug_read_name_to_row_number
                    ), "Read seen twice!"
                    debug_read_name_to_row_number[read_key] = read_number
                    print("************************************************\n")

                coo_row.extend([read_number] * len(read_cpg_cols))
                coo_col.extend(read_cpg_cols)
                coo_data.extend(read_cpg_data)
                tlen_list.append(aligned_segment.template_length)
                read_number += 1

                continue  # Skip the Biscuit/bwameth/gem3 path below

            # ============================================================
            # Biscuit / bwameth / gem3 path: use strand tags (YD / XB).
            # Check the strand tag early (before CpG lookup) since ~50% of
            # reads are on the daughter strand and can be skipped.
            # ============================================================
            bisulfite_parent_strand_is_reverse = None
            if aligned_segment.has_tag("YD"):  # Biscuit / bwameth tag
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

            # We have paired-end reads; one half (the "parent strand") has the methylation data.
            # The other half (the "daughter strand") was the complement created by PCR, which we don't care about.
            if bisulfite_parent_strand_is_reverse != aligned_segment.is_reverse:
                # Skip if we're not on the bisulfite-converted parent strand.
                if debug:
                    print("\tNot on methylated strand, ignoring.")
                continue

            # Non-converted filter (Biscuit/bwameth/gem3): count retained
            # non-CpG Cs (forward parent) or Gs (reverse parent) validated
            # against the reference via the MD tag. Applied after the
            # strand check so we don't waste work on daughter-strand reads.
            if filter_non_converted:
                if (
                    count_non_cpg_retained_reference(
                        aligned_segment,
                        bool(bisulfite_parent_strand_is_reverse),
                    )
                    >= non_converted_threshold
                ):
                    if debug:
                        print("\tNon-converted filter: dropping read.")
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
                assert aligned_segment.has_tag("YD") or aligned_segment.has_tag(
                    "XB"
                )  # Bisulfite strand tag (YD for Biscuit/bwameth, XB for gem3)

            # TODO: We ignore paired/unpaired read status for now. Should we treat paired reads / overlapping reads differently?

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

                read_cpg_cols.append(
                    genome_methylation_embedding.genomic_position_to_embedding(
                        chrom,
                        ref_pos + 1,
                    )
                )

                if query_base == "C":
                    # Methylated
                    read_cpg_data.append(1)
                    if debug:
                        print(f"\t{query_pos} {ref_pos} C->{query_base} [Methylated]")
                elif query_base == "T":
                    read_cpg_data.append(0)
                    # Unmethylated
                    if debug:
                        print(f"\t{query_pos} {ref_pos} C->{query_base} [Unmethylated]")
                else:
                    read_cpg_data.append(-1)
                    if debug:
                        print(
                            f"\t{query_pos} {ref_pos} C->{query_base} [Unknown! SNV? Indel?]"
                        )

            if filter_em_overconversion and is_em_overconversion_read(
                read_cpg_data, em_overconversion_min_cpgs
            ):
                if debug:
                    print("\tEM over-conversion filter: dropping read.")
                continue

            if debug:
                # Ensure each read is only seen once
                assert (
                    aligned_segment.query_name not in debug_read_name_to_row_number
                ), "Read seen twice!"
                debug_read_name_to_row_number[
                    aligned_segment.query_name  # type: ignore
                    + ("_1" if aligned_segment.is_read1 else "_2")
                ] = read_number

            coo_row.extend([read_number] * len(read_cpg_cols))
            coo_col.extend(read_cpg_cols)
            coo_data.extend(read_cpg_data)
            tlen_list.append(aligned_segment.template_length)
            read_number += 1

            if debug:
                print("************************************************\n")

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

    tlen_array = np.array(tlen_list, dtype=np.int32)

    return ExtractionResult(matrix=sparse_matrix, tlen=tlen_array)
