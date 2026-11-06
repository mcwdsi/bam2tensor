"""Core methylation extraction functions for bam2tensor.

This module contains the main function for extracting methylation data from
BAM files: extract_methylation_data_from_bam(). This function reads aligned
bisulfite-sequencing (BS-seq) and enzymatic methylation sequencing (EM-seq)
reads and generates a sparse matrix representation of methylation states at
each CpG site, with one row per sequenced DNA fragment.

The extraction process:
    1. Iterates through each chromosome in the genome embedding
    2. For each aligned read that passes the read filters:
       a. Determines which bisulfite strand the fragment derives from
       b. Identifies the CpG sites whose informative base the read covers
       c. Extracts the methylation state at each of those CpGs, keeping
          a call only where the read shows an intact CpG on a base of
          sufficient quality
    3. Combines the two mates of a read pair into a single fragment
    4. Builds a sparse COO matrix with one row per fragment

Methylation State Encoding:
    - 1: Methylated (cytosine remained as C after bisulfite treatment)
    - 0: Unmethylated (cytosine converted to T by bisulfite treatment)
    - -1: No usable call. The fragment covers the CpG, but the read does
      not show an intact CpG there (an indel, SNV or other unexpected
      base at either base of the CpG), the base carrying the call is
      below the minimum base quality, the CpG lies in the masked 3' end
      of the fragment (opt-in), or the two mates of a pair disagree.

Strands and the informative base:
    A fragment derives from either the top or the bottom strand of the
    original DNA, and both mates of a proper pair report on that same
    strand. BAM
    stores every read in top-strand orientation, so for a CpG whose C is at
    reference position p:

    - a top-strand fragment carries the call on the C, at p
      (C = methylated, T = unmethylated);
    - a bottom-strand fragment carries it on the G, at p + 1
      (G = methylated, A = unmethylated).

    Each read is therefore decoded at whichever of the two bases is
    informative for its fragment's strand, whatever the read's own alignment
    orientation.

Supported Aligners:
    The module supports BAM files from aligners that provide strand information:
    - Bismark: Uses the XM call string (Z=methylated CpG, z=unmethylated
      CpG) and the XG tag (CT=top strand, GA=bottom strand)
    - Biscuit: Uses the YD tag (f=top strand, r=bottom strand)
    - bwameth: Uses the YD tag (f=top strand, r=bottom strand), as Biscuit
    - gem3/Blueprint: Uses the XB tag (C=top strand, G=bottom strand)

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

from collections.abc import Container
from typing import Iterator, NamedTuple

import array
import collections
import multiprocessing as mp
import numpy as np
import scipy.sparse
import pysam
import bisect
import warnings

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from bam2tensor.embedding import GenomeMethylationEmbedding


class ExtractionResult(NamedTuple):
    """Result of methylation extraction from a BAM file.

    Attributes:
        matrix: Sparse COO matrix of shape (n_fragments, n_cpg_sites) with
            methylation states: 1 (methylated), 0 (unmethylated), -1
            (no data). Each row is one sequenced DNA fragment: the two
            mates of a read pair combined, or a single read.
        tlen: 1-D numpy array of shape (n_fragments,) containing the
            length of each fragment (the absolute BAM TLEN). 0 for
            single-end reads or reads with unmapped mates.
        stats: Counts describing what the extraction saw and discarded,
            for quality control (the CLI stores them in the output file's
            metadata). Keys:

            - ``reads_used``: reads that passed every read filter
              (mapping quality, flags, proper pair), whether or not they
              cover a CpG.
            - ``reads_dropped_improper_pair``: reads skipped because
              they are paired but not flagged as a proper pair.
            - ``reads_without_md``: reads the non-converted filter could
              not evaluate for lack of an ``MD`` tag.
            - ``fragments_from_two_mates``: fragments in which the calls
              of two mates were combined (counted before the
              fragment-level filters).
            - ``cpg_observations``: CpGs decoded, counted per read
              (before the two mates of a pair are combined).
            - ``cpg_not_intact``: observations stored as ``-1`` because
              the read does not show an intact CpG there.
            - ``cpg_reference_mismatch``: how many of those disagree with
              the reference in a way a wrong strand tag cannot explain
              (the count behind the reference check).
            - ``cpg_low_base_quality``: observations stored as ``-1``
              because the base carrying the call is below the minimum
              base quality.
            - ``cpg_masked_3prime``: calls stored as ``-1`` because they
              lie in the masked 3' end of their fragment (counted per
              fragment, after the mates are combined).
            - ``strand_checks`` and ``strand_contradictions``: the
              strand-tag check (CpGs examined, and how many contradict
              their read's strand tag).
    """

    matrix: scipy.sparse.coo_matrix
    tlen: np.ndarray
    stats: dict[str, int]


# BAM flag bits for reads to skip: duplicate (0x400), qcfail (0x200),
# secondary (0x100), supplementary (0x800), unmapped (0x4).
#
# Unmapped reads matter because an unmapped mate of a mapped read is "placed"
# at its mate's coordinates, so a region fetch returns it even though it has
# no alignment (no CIGAR, no reference end).
_SKIP_FLAGS = 0x400 | 0x200 | 0x100 | 0x800 | 0x4


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
    cpg_positions: Container[int] | None = None,
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

    This is a port of the logic in
    ``nebiolabs/mark-nonconverted-reads``, re-using the read's existing
    ``MD`` tag (via :py:meth:`pysam.AlignedSegment.get_aligned_pairs`
    with ``with_seq=True``) instead of requiring a separate reference
    FASTA.

    Which cytosines are in CpG context — and so exempt — is best taken
    from the CpG index via ``cpg_positions``. Without it, the context is
    inferred from the neighbouring reference base as reconstructed from
    the read, which cannot see a neighbour lying just outside the
    alignment: a methylated CpG cut by the end of the read is then
    counted as a retained non-CpG cytosine.

    Note:
        The count is only as good as the ``MD`` tag. bwa-meth writes
        ``MD`` relative to its bisulfite-converted reference. For the
        directional libraries bwa-meth is designed for, that gives the
        exact count, with one blind spot: a read ``C`` over a reference
        ``T`` (or ``G`` over ``A``) is a SNP that conversion makes
        invisible, so it is counted as retained. For a non-directional
        library aligned with bwa-meth the tag does not describe the
        read as stored, and retained cytosines are undercounted.

    Args:
        aligned_segment: A pysam aligned read. Must carry an ``MD``
            tag. bwa-meth writes one; gem3 does not for bisulfite
            alignments, so its reads cannot be evaluated unless the tag
            is added afterwards (for example with ``samtools calmd``).
        is_reverse_parent_strand: ``True`` if the read derives from the
            reverse (OB/CTOB) bisulfite parent strand, ``False`` for
            the forward (OT/CTOT) strand.
        cpg_positions: 0-based reference positions of the ``C`` of
            every CpG the read touches, including a CpG whose ``C`` or
            ``G`` lies just outside the alignment. If ``None``, CpG
            context is inferred from the read's own reconstruction of
            the reference.

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
        # Exclude CpG context.
        if cpg_positions is not None:
            # From the CpG index: exact, even when the other base of the CpG
            # lies outside the read. On the reverse strand the retained base
            # is the G, one position right of the CpG's C.
            if (rpos - 1 if is_reverse_parent_strand else rpos) in cpg_positions:
                continue
        elif is_reverse_parent_strand:
            # Reverse strand: previous reference base is C.
            if ref_pos_to_base.get(rpos - 1) == "C":
                continue
        else:
            # Forward strand: next reference base is G.
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
    CpG calls are all unmethylated *and* number at least ``min_cpgs``
    (the paper's Fig. 1C regime where the artifact diverges clearly
    from WGBS).

    CpGs without a usable call (``-1``) are left out of the decision
    altogether: they are neither evidence of methylation nor counted
    towards ``min_cpgs``. Otherwise a single low-quality base would
    shield an otherwise fully unmethylated fragment from the filter.

    This heuristic also drops genuinely fully-unmethylated biological
    fragments, so callers should opt in only when the downstream
    application can tolerate that trade-off.

    Args:
        read_cpg_states: Per-CpG methylation state values for a single
            read, in column-order, using the bam2tensor encoding
            (``1``=methylated, ``0``=unmethylated, ``-1``=no data).
        min_cpgs: Minimum number of CpG calls required to apply the
            filter. Reads with fewer calls are never flagged.

    Returns:
        ``True`` when the read has at least ``min_cpgs`` CpG calls and
        every one of them is unmethylated (value ``0``). ``-1``
        (no usable call) values are ignored.

    Example:
        >>> is_em_overconversion_read([0, 0, 0], min_cpgs=3)
        True
        >>> is_em_overconversion_read([0, 0, 1], min_cpgs=3)
        False
        >>> is_em_overconversion_read([0, 0], min_cpgs=3)
        False
        >>> is_em_overconversion_read([0, 0, -1], min_cpgs=3)
        False
        >>> is_em_overconversion_read([0, 0, -1, 0], min_cpgs=3)
        True
    """
    return read_cpg_states.count(0) >= min_cpgs and 1 not in read_cpg_states


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


def check_chromosome_lengths(
    bam_lengths: dict[str, int],
    reference_lengths: dict[str, int],
) -> None:
    """Check that the BAM was aligned to chromosomes of the reference's lengths.

    A BAM records the length of every sequence it was aligned to. If a
    chromosome has one length there and another in the reference FASTA the
    CpG index was built from, the two are different assemblies (hg19 and
    hg38, say) that merely share chromosome names. Positions do not
    correspond between them, so the reads would be decoded at positions
    that are not CpGs in the genome they were aligned to.

    Only chromosomes present under the same name on both sides are
    compared.

    Args:
        bam_lengths: Sequence name to length, from the BAM header.
        reference_lengths: Chromosome name to length, from the reference
            FASTA (``GenomeMethylationEmbedding.chromosome_lengths``).

    Raises:
        ValueError: If any shared chromosome differs in length.

    Example:
        >>> check_chromosome_lengths({"chr1": 1000}, {"chr1": 1000, "chr2": 500})
    """
    mismatched = [
        (name, bam_lengths[name], length)
        for name, length in reference_lengths.items()
        if name in bam_lengths and bam_lengths[name] != length
    ]
    if not mismatched:
        return
    shown = ", ".join(
        f"{name} ({bam_length:,} bp in the BAM, {length:,} bp in the reference)"
        for name, bam_length, length in mismatched[:3]
    )
    if len(mismatched) > 3:
        shown += f", ... ({len(mismatched)} chromosomes in total)"
    raise ValueError(
        "The BAM file was not aligned to the reference FASTA provided: "
        f"chromosome lengths differ for {shown}. Positions do not correspond "
        "between two different assemblies, so the methylation calls would be "
        "wrong. Use the reference the BAM was aligned to. (If the FASTA is a "
        "deliberately truncated copy of that reference, the check can be "
        "turned off with --allow-length-mismatch.)"
    )


class _ChromExtractionResult(NamedTuple):
    """Per-chromosome extraction output used by :func:`_extract_chromosome`.

    Observations are held in compact typed arrays rather than Python lists
    of ints: a whole-genome BAM yields on the order of a billion CpG
    observations, and a list of ints costs roughly nine times the memory
    of a 32-bit array (an 8-byte pointer plus a 28-byte int object per
    element), and far more time to pickle back from a worker process.

    Row indices are not stored. Fragments are emitted in order, so the row
    of every observation follows from ``cpgs_per_fragment``; the caller
    rebuilds the row array once, after concatenating chromosomes in the
    order they appear in ``GenomeMethylationEmbedding.cpg_sites_dict``.

    Attributes:
        cols: Global embedding column index of each CpG observation
            (already offset by the chromosome's position in the
            embedding).
        data: Methylation state of each CpG observation (1 / 0 / -1), int8.
        cpgs_per_fragment: Number of CpG observations in each emitted
            fragment (one entry per row), int32.
        tlens: Per-fragment template lengths (one entry per row), int32.
        missing: ``True`` if the chromosome was not present in the BAM
            file (the caller logs a verbose note in that case).
        reads_without_md: Number of reads the non-converted filter could
            not evaluate because they carry no ``MD`` tag. Always ``0``
            when that filter is off or the reads are Bismark-style.
        strand_checks: Number of CpGs, in reads decoded from their bases,
            at which the base that the read's strand tag says cannot be
            converted was examined.
        strand_contradictions: How many of those showed the other
            strand's conversion product, contradicting the tag.
        improper_pair_reads: Reads skipped because they are paired but
            not flagged as a proper pair.
        reads_used: Reads that passed every read filter.
        paired_fragments: Fragments in which the calls of two mates were
            combined.
        cpg_observations: CpGs decoded, counted per read (before mates
            are combined).
        cpgs_not_intact: Observations recorded as ``-1`` because the
            read does not show an intact CpG.
        cpgs_mismatched: How many of those disagree with the reference in
            a way a wrong strand tag cannot explain (every reason except
            the other strand's conversion product at the CpG's other
            base).
        cpgs_low_quality: Observations recorded as ``-1`` because the
            base carrying the call is below the minimum base quality.
        cpgs_masked: Calls recorded as ``-1`` because they lie in the
            masked 3' end of their fragment (counted per fragment, after
            the mates are combined).
    """

    cols: np.ndarray
    data: np.ndarray
    cpgs_per_fragment: np.ndarray
    tlens: np.ndarray
    missing: bool
    reads_without_md: int = 0
    strand_checks: int = 0
    strand_contradictions: int = 0
    improper_pair_reads: int = 0
    reads_used: int = 0
    paired_fragments: int = 0
    cpg_observations: int = 0
    cpgs_not_intact: int = 0
    cpgs_mismatched: int = 0
    cpgs_low_quality: int = 0
    cpgs_masked: int = 0


class _ExtractionConfig(NamedTuple):
    """Bundle of filter and decode settings shared across chromosomes.

    Passed once to each worker process (via the pool initializer in the
    parallel path) and inline in the sequential path. Keeps the
    per-chromosome helper's signature small and gives type-checked
    access to settings inside the worker.

    ``col_typecode`` is the :mod:`array` typecode used to buffer column
    indices: ``"i"`` (32-bit) whenever the embedding's column count fits,
    otherwise ``"q"`` (64-bit).
    """

    pysam_threads: int
    quality_limit: int
    filter_non_converted: bool
    non_converted_threshold: int
    filter_em_overconversion: bool
    em_overconversion_min_cpgs: int
    min_base_quality: int
    mask_3prime: int
    keep_improper_pairs: bool
    debug: bool
    col_typecode: str = "i"


# One unit of parallel work: (chromosome name, its sorted 1-based CpG
# positions, the matrix column of its first CpG).
_ChromTask = tuple[str, list[int], int]

# Worker-process state set by :func:`_worker_init`: the BAM path and the
# extraction settings, sent once per worker. The CpG positions are *not*
# part of it — each task carries only the positions for its own chromosome,
# so a worker never holds (or has to unpickle) the whole genome's index.
_WORKER_STATE: "tuple[str, _ExtractionConfig] | None" = None


def _worker_init(input_bam: str, config: _ExtractionConfig) -> None:
    """Initialize a parallel worker with shared extraction state."""
    global _WORKER_STATE
    _WORKER_STATE = (input_bam, config)


def _worker_extract_chromosome(task: _ChromTask) -> _ChromExtractionResult:
    """Worker entrypoint: extract methylation calls for one chromosome.

    Args:
        task: The chromosome name, its sorted 1-based CpG positions, and
            the matrix column of its first CpG.

    Returns:
        The per-chromosome extraction result.
    """
    assert _WORKER_STATE is not None
    input_bam, config = _WORKER_STATE
    chrom, cpg_sites, col_offset = task
    return _extract_chromosome(
        input_bam=input_bam,
        chrom=chrom,
        cpg_sites=cpg_sites,
        col_offset=col_offset,
        config=config,
    )


def _as_array(buffer: array.array) -> np.ndarray:
    """View a typed :mod:`array` buffer as a NumPy array without copying.

    Args:
        buffer: The buffer to expose.

    Returns:
        A NumPy array sharing ``buffer``'s memory, with the matching dtype.
    """
    return np.frombuffer(buffer, dtype=buffer.typecode)


# Strand-tag check (Biscuit / bwameth / gem3 reads, which are decoded from
# their bases on the strength of the aligner's strand tag).
#
# A read reports on one strand, so only one base of each CpG can show a
# conversion: the C for a top-strand fragment, the G for a bottom-strand one.
# The CpG's other base is never converted. Seeing that other base converted
# (an A at the G of a read tagged top strand, a T at the C of a read tagged
# bottom strand) contradicts the tag. In correctly tagged data that only
# happens through sequencing errors and CpG SNPs — a fraction of a percent of
# CpGs. Wrongly tagged reads show it at every unmethylated CpG, which is what
# happens to most complementary-strand pairs when a non-directional library is
# aligned by an aligner that assumes a directional one (bwa-meth).
#
# A BAM is flagged when more than this fraction of the CpGs checked contradict
# their read's tag...
_STRAND_CONTRADICTION_WARNING_RATE = 0.02
# ...provided at least this many CpGs were checked, so that a handful of
# errors in a small BAM cannot trigger the warning.
_STRAND_CHECK_MIN_CPGS = 1000

# How far downstream (in bp) a read's mate may start and still be combined
# with it into one fragment. Reads are processed in coordinate order and rows
# are written in the order fragments first appear, so a fragment waiting for
# its mate holds back every fragment behind it. Real inserts are a few hundred
# bases; this generous cap only bounds how long a far-flung (chimeric or
# mis-annotated) pair can make the queue grow. Mates further apart than this
# are reported as two separate fragments.
_MAX_MATE_DISTANCE = 100_000

# Two of the default filters act on what the BAM says about itself: the
# proper-pair flag and the base qualities. A BAM that does not carry them in
# the usual way (an aligner that never sets the flag; placeholder qualities)
# would have most of its data discarded without a word. A BAM is flagged when
# more than this fraction of its otherwise usable reads are dropped as
# improper pairs, or of its otherwise usable CpG calls for low base quality...
_FILTER_MAJORITY_WARNING_RATE = 0.5
# ...provided at least this many reads, or calls, were dropped.
_FILTER_MAJORITY_MIN_COUNT = 1000

# 3' end masking: how many fragment ends noted by a CpG-less mate (see
# ends_seen in _extract_chromosome) are kept before those whose other mate
# never arrived are discarded.
_MAX_ENDS_SEEN = 4096

# Key under which no fragment is ever filed while awaiting its mate (see
# _Fragment.key): the lookup key of a read that has no mate to be merged with.
_NO_MATE_KEY: tuple[str, int, int] = ("", -1, -1)

# Reference check. Wherever a read covers a CpG of the reference, the read is
# expected to show a CpG too: at the base carrying the call, the methylated
# or the unmethylated base; at the other base, the reference base. Sequencing
# errors, SNPs and indels break this at a percent or two of CpGs. A BAM
# decoded against a reference it was not aligned to breaks it at most of
# them, because the positions taken to be CpGs are arbitrary positions of the
# genome the reads were really aligned to.
#
# One way of not showing a CpG is left out of this count: the other strand's
# conversion product at the CpG's other base. That is the signature of a
# wrong strand tag, which the strand-tag check above reports.
#
# A BAM is flagged when more than this fraction of the CpGs decoded mismatch
# the reference...
_REFERENCE_MISMATCH_WARNING_RATE = 0.10
# ...provided at least this many CpGs were decoded.
_REFERENCE_MISMATCH_MIN_CPGS = 1000


def _three_prime_end(
    reference_start: int,
    reference_end: int,
    template_length: int,
    is_reverse: bool,
    mate_start: int | None,
    bottom_strand: bool,
) -> tuple[int, bool]:
    """Locate the 3' end of a fragment's original strand, as seen by one read.

    Library preparation repairs fragment ends before conversion: a recessed
    3' end is filled in by a polymerase, using unmethylated cytosines. Those
    cytosines are converted like any other unmethylated cytosine, so CpGs
    within the filled-in stretch read as unmethylated whatever the original
    molecule carried. The stretch is at the 3' end of the strand the
    fragment derives from: the fragment's right-hand end for a top-strand
    fragment, its left-hand end for a bottom-strand one.

    Sequencing starts at the two ends of the fragment, so that end is where
    one of the mates begins: the reverse-aligned mate for a top-strand
    fragment (its reference end), the forward-aligned mate for a
    bottom-strand fragment (its reference start). A read that is that mate
    knows the end exactly. So does the reverse-aligned read of a
    bottom-strand fragment, whose BAM record gives its mate's start. The
    forward-aligned read of a top-strand fragment cannot know where its mate
    ends, and estimates it from the template length; the estimate is
    replaced by the mate's own coordinate if the mate passes the read
    filters (see :func:`_extract_chromosome`). The template length is only a fallback
    because aligners disagree on it when the mates extend past each other:
    Bismark reports the outer span of the two alignments, bwa the distance
    between their starting ends.

    A read with no usable mate information (a single-end read, for
    instance) falls back on its own alignment end, the only fragment end it
    can vouch for.

    Args:
        reference_start: The read's 0-based reference start.
        reference_end: The read's 0-based reference end (exclusive).
        template_length: The read's BAM TLEN (signed; 0 if unknown).
        is_reverse: Whether the read is aligned to the reverse strand
            (BAM flag 0x10).
        mate_start: 0-based reference start of the read's mate, if the
            mate is mapped to the same chromosome and aligned forward;
            otherwise ``None``.
        bottom_strand: Whether the fragment derives from the bottom strand.

    Returns:
        The 0-based position of the fragment's 3' end — for a top-strand
        fragment the position just past its last base, for a bottom-strand
        fragment the position of its first base — and whether that is exact
        rather than estimated.

    Example:
        >>> # Top-strand pair spanning [100, 300): the reverse mate knows.
        >>> _three_prime_end(220, 300, -200, True, 100, False)
        (300, True)
        >>> _three_prime_end(100, 200, 200, False, None, False)
        (300, False)
        >>> # Bottom-strand pair spanning [100, 300): both mates know.
        >>> _three_prime_end(100, 200, 200, False, None, True)
        (100, True)
        >>> _three_prime_end(220, 300, -200, True, 100, True)
        (100, True)
        >>> # Single-end reads: their own end.
        >>> _three_prime_end(100, 200, 0, False, None, False)
        (200, False)
        >>> _three_prime_end(100, 200, 0, True, None, True)
        (100, False)
    """
    if bottom_strand:
        if not is_reverse:
            return reference_start, True
        if mate_start is not None:
            return mate_start, True
        if template_length < 0:
            return reference_end + template_length, False
        return reference_start, False
    if is_reverse:
        return reference_end, True
    if template_length > 0:
        return reference_start + template_length, False
    return reference_end, False


class _Fragment:
    """CpG calls for one DNA fragment, held until both mates have been seen.

    Attributes:
        name: Query name shared by the fragment's reads.
        is_read1: Whether the mate that opened this fragment is read 1.
        cols: Matrix columns of the CpGs called so far, ascending.
        data: Methylation state for each entry of ``cols``.
        tlen: Fragment length (absolute BAM TLEN); 0 if unknown.
        awaiting_mate: True while the second mate may still arrive.
        mate_start: 0-based reference start of the awaited mate.
        dropped: True if a filter removed the fragment. A dropped fragment
            stays queued while it awaits its mate, so that the mate is
            discarded with it rather than starting a fragment of its own.
        key: While the fragment awaits its mate, the key it is filed under:
            (query name, start of the first mate, start expected of the
            second).
        bottom_strand: Whether the fragment derives from the bottom strand
            (as tagged on the read that opened it).
        three_prime: 0-based position of the 3' end of the fragment's
            original strand (see :func:`_three_prime_end`). Only maintained
            when 3' end masking is on.
        three_prime_exact: False while ``three_prime`` is an estimate that
            the other mate's coordinates could still replace.
    """

    __slots__ = (
        "name",
        "is_read1",
        "cols",
        "data",
        "tlen",
        "awaiting_mate",
        "mate_start",
        "dropped",
        "key",
        "bottom_strand",
        "three_prime",
        "three_prime_exact",
    )

    def __init__(
        self, name: str, is_read1: bool, cols: list[int], data: list[int], tlen: int
    ) -> None:
        """Open a fragment with the calls of its first mate.

        Args:
            name: Query name shared by the fragment's reads.
            is_read1: Whether this first mate is read 1.
            cols: Matrix columns of the mate's CpGs, ascending.
            data: Methylation state for each entry of ``cols``.
            tlen: Fragment length (absolute BAM TLEN); 0 if unknown.
        """
        self.name = name
        self.is_read1 = is_read1
        self.cols = cols
        self.data = data
        self.tlen = tlen
        self.awaiting_mate = False
        self.mate_start = -1
        self.dropped = False
        self.key = _NO_MATE_KEY
        self.bottom_strand = False
        self.three_prime = 0
        self.three_prime_exact = True


def _bismark_is_bottom_strand(
    aligned_segment: pysam.AlignedSegment, xm_tag: str
) -> bool:
    """Tell whether a Bismark alignment derives from the bottom strand.

    Bismark records the strand of the genome a read (pair) aligned to in the
    ``XG`` tag: ``CT`` for the original top strand, ``GA`` for the original
    bottom strand. Both mates of a pair carry the same value. Bismark writes
    each methylation call on the base that carries it, so calls for a
    top-strand alignment sit on the reference ``C`` of a CpG and calls for a
    bottom-strand alignment sit on the ``G``.

    If ``XG`` is absent (a Bismark-style BAM from another tool), the strand
    is inferred from the read bases the calls sit on: ``C``/``T`` for the
    top strand, ``G``/``A`` for the bottom strand.

    Args:
        aligned_segment: A pysam aligned read carrying an ``XM`` tag.
        xm_tag: The value of that ``XM`` tag.

    Returns:
        ``True`` for a bottom-strand alignment, ``False`` for a top-strand
        alignment (also the fallback when nothing identifies the strand).
    """
    if aligned_segment.has_tag("XG"):
        return aligned_segment.get_tag("XG") == "GA"

    sequence = aligned_segment.query_sequence
    if sequence is not None:
        for base, call in zip(sequence, xm_tag):
            if call != ".":
                if base in "GA":
                    return True
                if base in "CT":
                    return False
    return False


def _mate_start_if_expected(
    aligned_segment: pysam.AlignedSegment, reference_start: int
) -> int | None:
    """Return where a read's mate starts, if the mate may still be seen.

    Reads are processed in coordinate order within one chromosome, so the
    mate can only still turn up if it is mapped to the same chromosome at or
    after this read's own start. Mates more than ``_MAX_MATE_DISTANCE``
    downstream are not waited for.

    Args:
        aligned_segment: The read whose mate is in question.
        reference_start: The read's own 0-based reference start.

    Returns:
        The mate's 0-based reference start, or ``None`` if the read has no
        mate that can be combined with it.
    """
    if (
        not aligned_segment.is_paired
        or aligned_segment.mate_is_unmapped
        or aligned_segment.next_reference_id != aligned_segment.reference_id
    ):
        return None
    mate_start = aligned_segment.next_reference_start
    if 0 <= mate_start - reference_start <= _MAX_MATE_DISTANCE:
        return mate_start
    return None


def _merge_mate_calls(
    cols: list[int],
    data: list[int],
    mate_cols: list[int],
    mate_data: list[int],
) -> tuple[list[int], list[int]]:
    """Combine the CpG calls of a fragment's two mates.

    Mates that overlap report some CpGs twice. Those are the same bases of
    the same molecule read twice, so they are recorded once:

    - if both mates make the same call, that call is kept;
    - if only one mate has a call (the other has ``-1``), that call is kept;
    - if the mates disagree (one methylated, one unmethylated), at least one
      is a sequencing error and there is no telling which, so the CpG is
      recorded as ``-1`` (no data).

    Args:
        cols: Matrix columns called by the first mate, ascending.
        data: Methylation state for each entry of ``cols``.
        mate_cols: Matrix columns called by the second mate, ascending.
        mate_data: Methylation state for each entry of ``mate_cols``.

    Returns:
        The merged columns, ascending, and the state for each.

    Example:
        >>> _merge_mate_calls([3, 4], [1, 0], [7, 8], [1, 1])
        ([3, 4, 7, 8], [1, 0, 1, 1])
        >>> _merge_mate_calls([3, 4, 5], [1, -1, 0], [4, 5, 6], [1, 1, 0])
        ([3, 4, 5, 6], [1, 1, -1, 0])
    """
    # Common case: the mates do not overlap at any CpG.
    if cols[-1] < mate_cols[0]:
        return cols + mate_cols, data + mate_data
    if mate_cols[-1] < cols[0]:
        return mate_cols + cols, mate_data + data

    merged = dict(zip(cols, data))
    for col, state in zip(mate_cols, mate_data):
        seen = merged.get(col)
        if seen is None or seen == -1:
            merged[col] = state
        elif state != -1 and state != seen:
            merged[col] = -1
    merged_cols = sorted(merged)
    return merged_cols, [merged[col] for col in merged_cols]


def _extract_chromosome(
    input_bam: str,
    chrom: str,
    cpg_sites: list[int],
    col_offset: int,
    config: _ExtractionConfig,
) -> _ChromExtractionResult:
    """Extract methylation calls from one chromosome of a BAM file.

    This is the per-chromosome unit of work used by both the sequential
    (``threads=1``) and parallel (``threads>1``) paths of
    :func:`extract_methylation_data_from_bam`. It opens its own pysam
    handle so it is safe to call from a worker subprocess.

    Reads are decoded one at a time and collected into fragments. A read
    opens a fragment the first time its query name is seen with CpG calls;
    if its mate is due later on the same chromosome the fragment waits for
    it, and the mate's calls are merged in when it arrives (see
    :func:`_merge_mate_calls`). The mate is recognised by its name, its
    mate number and its position: it must start where the first read's
    record says its mate starts, and point back at the first read's start.
    Fragments are written out in the order they were opened, each as soon
    as it and everything ahead of it is complete — a fragment stops waiting
    once its mate has been merged or the reads have moved past where the
    mate should have been.

    A call is decoded per read (strand, intact CpG, base quality). The 3'
    end mask is applied per fragment, as its row is written out, from the
    coordinates of the mates themselves (see :func:`_three_prime_end`).

    Args:
        input_bam: Path to the indexed BAM file.
        chrom: Chromosome name to process.
        cpg_sites: Sorted 1-based CpG positions on ``chrom``.
        col_offset: Matrix column of the first CpG on ``chrom``; the CpG
            at index ``i`` of ``cpg_sites`` is column ``col_offset + i``.
        config: Filter and decode settings; see :class:`_ExtractionConfig`.

    Returns:
        The per-chromosome extraction result; ``missing=True`` when
        ``chrom`` is not in the BAM.
    """
    quality_limit = config.quality_limit
    filter_non_converted = config.filter_non_converted
    non_converted_threshold = config.non_converted_threshold
    filter_em_overconversion = config.filter_em_overconversion
    em_overconversion_min_cpgs = config.em_overconversion_min_cpgs
    min_base_quality = config.min_base_quality
    mask_3prime = config.mask_3prime
    keep_improper_pairs = config.keep_improper_pairs
    debug = config.debug

    # Typed buffers (see _ChromExtractionResult): column index and state per
    # CpG observation, CpG count and TLEN per emitted fragment.
    coo_col = array.array(config.col_typecode)
    coo_data = array.array("b")
    cpgs_per_fragment = array.array("i")
    tlen_list = array.array("i")

    bam = pysam.AlignmentFile(  # type: ignore # pylint: disable=no-member
        input_bam, "rb", require_index=True, threads=config.pysam_threads
    )

    try:
        iter_reads = bam.fetch(contig=chrom)
    except ValueError:
        bam.close()
        return _ChromExtractionResult(
            cols=_as_array(coo_col),
            data=_as_array(coo_data),
            cpgs_per_fragment=_as_array(cpgs_per_fragment),
            tlens=_as_array(tlen_list),
            missing=True,
        )

    # Fragments whose rows have not been written yet, in the order they were
    # opened, and the subset still waiting for a second mate, keyed by
    # (query name, start of the first mate, start expected of the second).
    pending: collections.deque[_Fragment] = collections.deque()
    awaiting_mate: dict[tuple[str, int, int], _Fragment] = {}

    reads_without_md = 0
    strand_checks = 0
    strand_contradictions = 0
    improper_pair_reads = 0
    reads_used = 0
    paired_fragments = 0
    cpg_observations = 0
    cpgs_not_intact = 0
    cpgs_mismatched = 0
    cpgs_low_quality = 0
    cpgs_masked = 0

    # 3' end masking only: where a top-strand fragment ends, as given by a
    # reverse-aligned mate that covers no CpG itself (and so opens no
    # fragment) but comes before its forward-aligned mate in the BAM. Keyed
    # like awaiting_mate, for that mate to pick up.
    ends_seen: dict[tuple[str, int, int], int] = {}

    def emit(fragment: _Fragment) -> None:
        # Write a finished fragment's row, unless a filter removed it.
        nonlocal cpgs_masked
        if fragment.dropped:
            return
        if mask_3prime:
            # Discard calls within mask_3prime bases of the 3' end of the
            # fragment's original strand. Done here, once both mates have
            # been seen, so that the end is the one the mates themselves
            # give and both are masked alike. Columns ascend with position
            # (column col_offset + i is the CpG at cpg_sites[i], the 1-based
            # position of its C and the 0-based position of its G), so the
            # masked calls are the last few of a top-strand fragment and the
            # first few of a bottom-strand one.
            cols, data = fragment.cols, fragment.data
            if fragment.bottom_strand:
                # Calls sit on the G: masked if it starts fewer than
                # mask_3prime bases after the fragment's first base.
                limit = fragment.three_prime + mask_3prime
                for index, col in enumerate(cols):
                    if cpg_sites[col - col_offset] >= limit:
                        break
                    if data[index] != -1:
                        data[index] = -1
                        cpgs_masked += 1
            else:
                # Calls sit on the C: masked if it is among the last
                # mask_3prime bases of the fragment.
                limit = fragment.three_prime - mask_3prime
                for index in range(len(cols) - 1, -1, -1):
                    if cpg_sites[cols[index] - col_offset] - 1 < limit:
                        break
                    if data[index] != -1:
                        data[index] = -1
                        cpgs_masked += 1
        if filter_em_overconversion and is_em_overconversion_read(
            fragment.data, em_overconversion_min_cpgs
        ):
            if debug:
                print(f"\tEM over-conversion filter: dropping {fragment.name}.")
            return
        coo_col.extend(fragment.cols)
        coo_data.extend(fragment.data)
        cpgs_per_fragment.append(len(fragment.cols))
        tlen_list.append(fragment.tlen)

    for aligned_segment in iter_reads:
        if aligned_segment.mapping_quality < quality_limit:
            continue

        # Single bitwise check replaces separate flag checks: duplicate
        # (0x400), qcfail (0x200), secondary (0x100), supplementary (0x800),
        # unmapped (0x4).
        flag = aligned_segment.flag
        if flag & _SKIP_FLAGS:
            continue

        # A paired read (0x1) that the aligner did not mark as properly
        # paired (0x2): its mate is unmapped, on another chromosome, in the
        # wrong orientation or implausibly far away. Such pairs are enriched
        # for chimeric and mis-aligned molecules, so both mates are dropped
        # unless asked for. Single-end reads carry neither bit and are kept.
        if (flag & 0x3) == 0x1 and not keep_improper_pairs:
            improper_pair_reads += 1
            continue

        # A read flagged as mapped but carrying no alignment (e.g. CIGAR
        # "*") has no reference span, so there is nothing to extract.
        reference_end = aligned_segment.reference_end
        if reference_end is None:
            continue
        reference_start = aligned_segment.reference_start
        reads_used += 1

        # Write out, in order, every fragment that can no longer change. The
        # fragment at the head of the queue is finished once its mate has
        # been merged (or it never had one to wait for), or once the reads —
        # which arrive in coordinate order — have passed the position where
        # its mate was due without the mate turning up (it was filtered out,
        # or is not in the BAM).
        while pending:
            head = pending[0]
            if head.awaiting_mate and head.mate_start >= reference_start:
                break
            pending.popleft()
            if head.awaiting_mate and awaiting_mate.get(head.key) is head:
                del awaiting_mate[head.key]
            emit(head)

        name = aligned_segment.query_name
        assert name is not None
        is_read1 = aligned_segment.is_read1

        # Is a fragment already waiting for exactly this read? Its first mate
        # must have the same name, the other mate number, start where this
        # read says its mate starts, and expect its mate where this read
        # starts. Matching on the coordinates as well as the name keeps two
        # unrelated pairs that share a name from being combined.
        mate_key = _NO_MATE_KEY
        if (
            flag & 0x1
            and not flag & 0x8
            and aligned_segment.next_reference_id == aligned_segment.reference_id
        ):
            mate_key = (name, aligned_segment.next_reference_start, reference_start)
        partner = awaiting_mate.get(mate_key)
        if partner is not None and partner.is_read1 == is_read1:
            partner = None

        # For 3' end masking: a forward-aligned mate that mapped where this
        # read's record says, and — if this read completes a fragment whose
        # 3' end was only estimated — the end as this read knows it. This
        # comes before any reason to skip the read's calls, since a mate
        # that covers no CpG still shows where the fragment ends.
        if mask_3prime:
            forward_mate_start = (
                aligned_segment.next_reference_start
                if mate_key is not _NO_MATE_KEY and not flag & 0x20
                else None
            )
            if partner is not None and not partner.three_prime_exact:
                three_prime, exact = _three_prime_end(
                    reference_start,
                    reference_end,
                    aligned_segment.template_length,
                    bool(flag & 0x10),
                    forward_mate_start,
                    partner.bottom_strand,
                )
                if exact:
                    partner.three_prime = three_prime
                    partner.three_prime_exact = True

        # ============================================================
        # Which strand of the original DNA does this fragment derive
        # from? This depends on the aligner's strand tag — never on the
        # read's own alignment orientation.
        # ============================================================
        xm_tag: str | None = None
        non_converted = False
        if aligned_segment.has_tag("XM"):
            # Bismark: XM holds a pre-resolved call for every base.
            xm_tag = aligned_segment.get_tag("XM")  # type: ignore[assignment]
            assert xm_tag is not None
            bottom_strand = _bismark_is_bottom_strand(aligned_segment, xm_tag)

            # Non-converted filter (Bismark): XM already encodes retained
            # non-CpG cytosines as H/X/U.
            if filter_non_converted:
                non_converted = (
                    count_non_cpg_retained_xm(xm_tag) >= non_converted_threshold
                )
        else:
            if aligned_segment.has_tag("YD"):  # Biscuit / bwameth
                strand_tag = aligned_segment.get_tag("YD")
                if strand_tag == "f":  # OT/CTOT: C->T conversions
                    bottom_strand = False
                elif strand_tag == "r":  # OB/CTOB: G->A conversions
                    bottom_strand = True
                else:
                    continue
            elif aligned_segment.has_tag("XB"):  # gem3 / Blueprint
                strand_tag = aligned_segment.get_tag("XB")
                if strand_tag == "C":  # aligned to the C->T converted reference
                    bottom_strand = False
                elif strand_tag == "G":  # aligned to the G->A converted reference
                    bottom_strand = True
                else:
                    continue
            else:
                # No strand information: the read cannot be decoded.
                if debug:
                    print(f"\t{name}: no strand tag, ignoring.")
                continue

            # Non-converted filter (Biscuit/bwameth/gem3): count retained
            # non-CpG Cs (top strand) or Gs (bottom strand) validated
            # against the reference via the MD tag.
            if filter_non_converted:
                if not aligned_segment.has_tag("MD"):
                    # Without an MD tag the read cannot be compared against the
                    # reference, so the filter cannot evaluate it. Keep the read
                    # but count it, so the caller can warn instead of the filter
                    # silently doing nothing.
                    reads_without_md += 1
                else:
                    # The C of every CpG the read touches (0-based), counting
                    # a CpG whose C or G lies just outside the alignment, so
                    # that CpG context is exact right up to the read's ends.
                    nearby_cpgs = {
                        cpg_sites[i] - 1
                        for i in range(
                            bisect.bisect_left(cpg_sites, reference_start),
                            bisect.bisect_right(cpg_sites, reference_end),
                        )
                    }
                    non_converted = (
                        count_non_cpg_retained_reference(
                            aligned_segment, bottom_strand, nearby_cpgs
                        )
                        >= non_converted_threshold
                    )

        if non_converted:
            # Incomplete conversion is a property of the molecule, so the
            # whole fragment goes: a mate already queued is dropped, and a
            # mate still to come is discarded when it arrives.
            if debug:
                print(f"\tNon-converted filter: dropping {name}.")
            if partner is not None:
                del awaiting_mate[mate_key]
                partner.awaiting_mate = False
                partner.dropped = True
            else:
                mate_start = _mate_start_if_expected(aligned_segment, reference_start)
                if mate_start is not None:
                    own_key = (name, reference_start, mate_start)
                    if own_key not in awaiting_mate:
                        placeholder = _Fragment(name, is_read1, [], [], 0)
                        placeholder.dropped = True
                        placeholder.awaiting_mate = True
                        placeholder.mate_start = mate_start
                        placeholder.key = own_key
                        pending.append(placeholder)
                        awaiting_mate[own_key] = placeholder
            continue

        # ============================================================
        # Which CpGs does this read report on? Those whose informative
        # base lies within the alignment: the C (reference position p)
        # for a top-strand fragment, the G (p + 1) for a bottom-strand
        # fragment.
        #
        # aligned_segment.reference_start is 0-based inclusive,
        # reference_end is 0-based exclusive, cpg_sites are 1-based.
        # ============================================================
        informative_offset = 1 if bottom_strand else 0
        start_idx = bisect.bisect_left(
            cpg_sites, reference_start + 1 - informative_offset
        )
        end_idx = bisect.bisect_right(cpg_sites, reference_end - informative_offset)
        if start_idx >= end_idx:
            # No CpG, so no fragment to open. If this is the mate that marks
            # the 3' end of a top-strand fragment and the other mate is still
            # to come, leave the end where that mate will find it.
            if mask_3prime and partner is None and flag & 0x10 and not bottom_strand:
                mate_start = _mate_start_if_expected(aligned_segment, reference_start)
                if mate_start is not None:
                    if len(ends_seen) >= _MAX_ENDS_SEEN:
                        # Forget mates that never turned up: the reads have
                        # moved past where they were due.
                        ends_seen = {
                            key: end
                            for key, end in ends_seen.items()
                            if key[2] >= reference_start
                        }
                    ends_seen[(name, reference_start, mate_start)] = reference_end
            continue

        # Bismark calls can be read without the bases; everything else needs
        # them.
        sequence = aligned_segment.query_sequence
        if sequence is None:
            if xm_tag is None:
                continue
            bases = ""
        else:
            bases = sequence

        # 0-based reference positions of the two bases of each of those CpGs.
        # cpg_sites holds the 1-based position of each C, which is also the
        # 0-based position of its G.
        #
        # BAM SEQ is in top-strand orientation, so a top-strand fragment
        # shows C (methylated) or T (unmethylated) at the C of the CpG, and a
        # bottom-strand fragment shows G (methylated) or A (unmethylated) at
        # the G. The CpG's other base belongs to the strand the read does not
        # report on, so it must still read as the reference has it: a G after
        # the C for a top-strand read, a C before the G for a bottom-strand
        # read. The other strand's conversion product there (A at the G, T at
        # the C) contradicts the strand tag.
        g_positions = cpg_sites[start_idx:end_idx]
        c_positions = [position - 1 for position in g_positions]
        if bottom_strand:
            informative_positions, partner_positions = g_positions, c_positions
            methylated_base, unmethylated_base = "G", "A"
            partner_base, contradicting_base = "C", "T"
            partner_step = -1
        else:
            informative_positions, partner_positions = c_positions, g_positions
            methylated_base, unmethylated_base = "C", "T"
            partner_base, contradicting_base = "G", "A"
            partner_step = 1

        # Where each of those bases falls in the read (absent if deleted
        # from the read).
        wanted = set(informative_positions)
        if bases:
            wanted.update(partner_positions)
        query_position = {
            ref_pos: query_pos
            for query_pos, ref_pos in aligned_segment.get_aligned_pairs(
                matches_only=True
            )
            if ref_pos in wanted
        }

        qualities = aligned_segment.query_qualities if min_base_quality > 0 else None

        xm_length = len(xm_tag) if xm_tag is not None else 0
        read_data: list[int] = []
        for ref_pos, partner_ref_pos in zip(informative_positions, partner_positions):
            # The call itself, from the informative base.
            low_quality = False
            query_pos = query_position.get(ref_pos)
            if query_pos is None:
                state = -1  # deleted from the read
            else:
                if xm_tag is not None:
                    # Bismark: Z = methylated CpG, z = unmethylated CpG.
                    # Anything else at a CpG is a mismatch.
                    call = xm_tag[query_pos] if query_pos < xm_length else "."
                    state = 1 if call == "Z" else 0 if call == "z" else -1
                else:
                    base = bases[query_pos]
                    state = (
                        1
                        if base == methylated_base
                        else 0 if base == unmethylated_base else -1
                    )
                low_quality = (
                    qualities is not None and qualities[query_pos] < min_base_quality
                )

            # Unreadable against this reference: counts towards the
            # reference check (see _REFERENCE_MISMATCH_WARNING_RATE).
            mismatch = state == -1

            # The CpG's other base, where the read covers it. A call only
            # means something if this molecule has a CpG here, so the other
            # base must be present, adjacent, and the base the reference has.
            if bases and reference_start <= partner_ref_pos < reference_end:
                partner_pos = query_position.get(partner_ref_pos)
                if partner_pos is None:
                    state = -1  # deleted from the read
                    mismatch = True
                else:
                    other = bases[partner_pos]
                    # Strand-tag check, for reads decoded from their bases
                    # (see _STRAND_CONTRADICTION_WARNING_RATE).
                    if xm_tag is None and (
                        qualities is None or qualities[partner_pos] >= min_base_quality
                    ):
                        strand_checks += 1
                        if other == contradicting_base:
                            strand_contradictions += 1
                    if (
                        query_pos is not None
                        and partner_pos - query_pos != partner_step
                    ):
                        state = -1  # something inserted between the two bases
                        mismatch = True
                    elif other != partner_base:
                        state = -1
                        # The other strand's conversion product is what a
                        # wrong strand tag produces, and is reported by the
                        # strand-tag check; any other base is a mismatch.
                        if other != contradicting_base:
                            mismatch = True

            if mismatch:
                cpgs_mismatched += 1
            if state == -1:
                cpgs_not_intact += 1
            elif low_quality:
                state = -1
                cpgs_low_quality += 1
            read_data.append(state)

        # The CpG at index i of the sorted cpg_sites is column col_offset + i.
        read_cols = list(range(col_offset + start_idx, col_offset + end_idx))
        cpg_observations += end_idx - start_idx

        if debug:
            print(
                f"{name} ({'read1' if is_read1 else 'read2/single'}, "
                f"{'bottom' if bottom_strand else 'top'} strand, "
                f"{'XM' if xm_tag is not None else 'bases'}): "
                f"columns {read_cols[0]}-{read_cols[-1]}, calls {read_data}"
            )

        # ============================================================
        # Add the read's calls to its fragment.
        # ============================================================
        if partner is not None:
            # Second mate of a fragment that is already queued.
            del awaiting_mate[mate_key]
            partner.awaiting_mate = False
            if not partner.dropped:
                partner.cols, partner.data = _merge_mate_calls(
                    partner.cols, partner.data, read_cols, read_data
                )
                if partner.tlen == 0:
                    partner.tlen = abs(aligned_segment.template_length)
                paired_fragments += 1
            continue

        # First read of its fragment to contribute calls.
        fragment = _Fragment(
            name, is_read1, read_cols, read_data, abs(aligned_segment.template_length)
        )
        if mask_3prime:
            fragment.bottom_strand = bottom_strand
            fragment.three_prime, fragment.three_prime_exact = _three_prime_end(
                reference_start,
                reference_end,
                aligned_segment.template_length,
                bool(flag & 0x10),
                forward_mate_start,
                bottom_strand,
            )
            if not fragment.three_prime_exact:
                # The mate may already have gone by without opening a
                # fragment (see ends_seen).
                seen_end = ends_seen.pop(mate_key, None)
                if seen_end is not None:
                    fragment.three_prime = seen_end
                    fragment.three_prime_exact = True
        pending.append(fragment)
        mate_start = _mate_start_if_expected(aligned_segment, reference_start)
        if mate_start is not None:
            own_key = (name, reference_start, mate_start)
            if own_key not in awaiting_mate:
                fragment.awaiting_mate = True
                fragment.mate_start = mate_start
                fragment.key = own_key
                awaiting_mate[own_key] = fragment

    # End of the chromosome: nothing more can arrive.
    for fragment in pending:
        emit(fragment)

    bam.close()
    return _ChromExtractionResult(
        cols=_as_array(coo_col),
        data=_as_array(coo_data),
        cpgs_per_fragment=_as_array(cpgs_per_fragment),
        tlens=_as_array(tlen_list),
        missing=False,
        reads_without_md=reads_without_md,
        strand_checks=strand_checks,
        strand_contradictions=strand_contradictions,
        improper_pair_reads=improper_pair_reads,
        reads_used=reads_used,
        paired_fragments=paired_fragments,
        cpg_observations=cpg_observations,
        cpgs_not_intact=cpgs_not_intact,
        cpgs_mismatched=cpgs_mismatched,
        cpgs_low_quality=cpgs_low_quality,
        cpgs_masked=cpgs_masked,
    )


def extract_methylation_data_from_bam(
    input_bam: str,
    genome_methylation_embedding: GenomeMethylationEmbedding,
    quality_limit: int = 20,
    filter_non_converted: bool = False,
    non_converted_threshold: int = 3,
    filter_em_overconversion: bool = False,
    em_overconversion_min_cpgs: int = 3,
    threads: int = 1,
    verbose: bool = False,
    debug: bool = False,
    min_base_quality: int = 20,
    mask_3prime: int = 0,
    keep_improper_pairs: bool = False,
    allow_length_mismatch: bool = False,
) -> ExtractionResult:
    """Extract per-fragment CpG methylation data from a BAM file.

    Parses a bisulfite-sequencing or EM-seq BAM file and extracts methylation
    states at each CpG site covered by aligned reads. Returns a sparse matrix
    where each row represents one sequenced DNA fragment and each column
    represents a CpG site.

    **Rows are fragments.** The two mates of a read pair come from the same
    molecule, so their calls are combined into a single row; a single-end
    read, or a read whose mate is unusable, is a row on its own. Where the
    mates overlap, a CpG they both cover is recorded once: the shared call
    if they agree, the available call if only one mate has one, and ``-1``
    if they disagree. Mates are combined when both pass the read filters
    and map to the same chromosome within 100 kb of each other; otherwise
    each is reported as its own fragment.

    The function applies several filters to reads:
        - Mapping quality must be >= quality_limit (default 20)
        - Duplicate reads are excluded
        - QC-failed reads are excluded
        - Secondary and supplementary alignments are excluded
        - Unmapped reads are excluded (including unmapped mates placed at
          their mapped mate's position)
        - Paired reads that are not flagged as a proper pair are excluded
          (unless ``keep_improper_pairs``); single-end reads are kept
        - Reads without a recognised strand tag are excluded

    A call is kept only where it can be trusted; otherwise the CpG is
    recorded as ``-1`` (the fragment covers it, but there is no usable
    call):
        - **Intact CpG.** The read must show a CpG where the reference
          has one: the methylated or unmethylated base at the base that
          carries the call, and the reference base at the CpG's other
          base (where the read covers it), with nothing inserted between
          them. A molecule carrying a CpG-destroying variant (CpG to
          TpG/CpA is the commonest human SNP) has no CpG to be
          methylated, and on one strand such a variant is
          indistinguishable from an unmethylated CpG unless the other
          base is checked.
        - **Base quality.** The base carrying the call must have a Phred
          quality of at least ``min_base_quality`` (default 20; reads
          without base qualities are not filtered).
        - **3' end masking** (opt-in, ``mask_3prime``). End repair fills
          recessed 3' ends with unmethylated cytosines before
          conversion, so CpGs near the 3' end of a fragment's original
          strand read as unmethylated whatever the molecule carried.
          ``mask_3prime=N`` discards calls within N bases of that end.

    Two additional, opt-in filters are available:
        - Non-converted reads (``filter_non_converted``): drops fragments
          in which a read has too many retained non-CpG cytosines, the
          hallmark of incomplete bisulfite/EM-seq conversion. Ports the
          per-read logic of ``nebiolabs/mark-nonconverted-reads``; because
          incomplete conversion affects the whole molecule, a read that
          fails takes its mate with it. For Biscuit/bwameth/gem3 reads
          this needs the ``MD`` tag; reads without one cannot be
          evaluated, are kept, and trigger a ``RuntimeWarning`` reporting
          how many were affected.
        - EM-seq fragment-level over-conversion
          (``filter_em_overconversion``): drops fragments whose CpG
          calls (across both mates) are all unmethylated, a heuristic
          for the EM-seq artifact described by Loyfer et al. (bioRxiv
          2026.03.24.713040). CpGs without a usable call are left out
          of that decision.

    Every read is decoded at the base that is informative for the strand
    its fragment derives from: the ``C`` of a CpG for a top-strand
    fragment, the ``G`` for a bottom-strand fragment (BAM stores reads in
    top-strand orientation). This is independent of the read's own
    alignment orientation, so both mates of a pair are used. Two tag
    dialects are supported, detected automatically per read:

    **Bismark** (XM tag present):
        1. Takes the fragment's strand from the XG tag (CT = top, GA =
           bottom)
        2. Uses binary search to find the CpGs whose informative base the
           read covers
        3. Reads the XM call string there: Z=methylated(1),
           z=unmethylated(0), anything else=-1

    **Biscuit/bwameth/gem3** (YD or XB tag):
        1. Takes the fragment's strand from the tag (YD f / XB C = top,
           YD r / XB G = bottom)
        2. Uses binary search to find the CpGs whose informative base the
           read covers
        3. Reads the base there: top strand C=methylated(1),
           T=unmethylated(0); bottom strand G=methylated(1),
           A=unmethylated(0); anything else=-1

    In both dialects the intact-CpG, base-quality and 3' end rules above
    are then applied from the read's own bases and qualities, so the two
    give identical output for the same reads.

    **Strand-tag check.** Decoding from bases relies on the strand tag
    being right. It is checked against the reads themselves: the base of
    a CpG that the tag says cannot be converted (the G for a top-strand
    read, the C for a bottom-strand read) is examined wherever a read
    covers it with sufficient base quality. If more than 2% of at least
    1,000 such CpGs show the other strand's conversion product, a
    ``RuntimeWarning`` reports it. That pattern means many reads are
    mis-tagged — typically a non-directional library aligned by an
    aligner that assumes a directional one. In a mis-tagged read every
    unmethylated CpG fails the intact-CpG rule and is recorded as
    ``-1``, so only methylated calls survive and methylation is
    overstated. With ``verbose=True`` the counts are printed for every
    BAM.

    **Reference check.** The BAM must have been aligned to the reference
    the CpG index was built from. Chromosomes whose length in the BAM
    header differs from the reference raise a ``ValueError`` (unless
    ``allow_length_mismatch``). Independently, if more than 10% of at
    least 1,000 decoded CpGs mismatch the reference in their read, a
    ``RuntimeWarning`` reports it: real data stays within a few percent,
    while a BAM decoded against the wrong reference fails at most CpGs.
    (CpGs whose only fault is the other strand's conversion product at
    the CpG's other base are left to the strand-tag check.)

    **Filters that depend on the BAM's own annotations.** The proper-pair
    and base-quality filters trust the flag and the qualities in the
    BAM. If more than half of the otherwise usable reads are dropped as
    improper pairs, or more than half of the otherwise usable calls for
    low base quality (at least 1,000 in either case), a
    ``RuntimeWarning`` says so and names the option that turns the
    filter off.

    Args:
        input_bam: Path to the input BAM file. The file must be indexed
            (have a corresponding .bam.bai file).
        genome_methylation_embedding: A GenomeMethylationEmbedding object
            that defines the CpG site positions and provides coordinate
            conversion methods.
        quality_limit: Minimum mapping quality (MAPQ) threshold for reads.
            Reads with MAPQ below this value are skipped. Default is 20,
            which excludes reads mapping to multiple locations equally well.
        filter_non_converted: If True, drop fragments in which a read
            carries at least ``non_converted_threshold`` retained non-CpG
            cytosines, a signature of incomplete bisulfite/EM-seq
            conversion. Default False.
        non_converted_threshold: Minimum count of retained non-CpG
            cytosines required for the non-converted filter to drop a
            read. Matches the NEB ``mark-nonconverted-reads`` default of
            3.
        filter_em_overconversion: If True, drop fragments whose CpG
            calls are all unmethylated and number at least
            ``em_overconversion_min_cpgs`` — the Loyfer et al. EM-seq
            fragment-level over-conversion heuristic. Default False.
        em_overconversion_min_cpgs: Minimum number of CpG calls required
            before the over-conversion filter will drop a fragment. Matches
            the regime in Loyfer et al. Fig. 1C where the EM-seq
            artifact is clearly separable from WGBS.
        threads: Number of worker processes to use, one chromosome per
            worker. Default 1 (in-process, no IPC overhead). Values
            above 1 fan out per-chromosome extraction across worker
            processes; output is bitwise-identical to ``threads=1``
            because per-chromosome results are concatenated in
            ``cpg_sites_dict`` order, whatever order the workers finish
            in. Each worker is sent only the CpG positions of the
            chromosome it is processing, not the whole embedding.
            Recommended: leave at 1 for small BAMs (worker
            startup dominates); set to the number of physical cores
            for whole-genome WGBS BAMs. Workers are started with the
            ``spawn`` method, so a script that passes ``threads > 1``
            must do so from under ``if __name__ == "__main__":``
            (standard :mod:`multiprocessing` practice).
        verbose: If True, display a progress bar and print the total read
            count. Useful for monitoring progress on large files.
        debug: If True, print the calls decoded from every read and
            each filter decision. Significantly slower.
        min_base_quality: Minimum Phred base quality of the base that
            carries a CpG call. Calls on lower-quality bases are
            recorded as ``-1``. Default 20; 0 disables the filter. Reads
            stored without base qualities are never filtered.
        mask_3prime: Number of bases at the 3' end of each fragment's
            original strand in which calls are recorded as ``-1``, to
            remove the end-repair fill-in artifact. The end is the
            fragment's right-hand end for a top-strand fragment and its
            left-hand end for a bottom-strand one, taken from the
            alignment of the mate that starts there; a read whose mate
            is not available estimates it from the template length, and
            a single-end read uses its own end. The distance is measured
            on the aligned fragment, so bases already trimmed from the
            reads before alignment are not counted. Default 0 (off).
        keep_improper_pairs: If True, keep paired reads that are not
            flagged as a proper pair. Default False.
        allow_length_mismatch: If True, do not require the chromosome
            lengths in the BAM header to match the reference. Only for
            a reference that is a deliberately truncated copy of the one
            the BAM was aligned to. Default False.

    Returns:
        An ExtractionResult named tuple with three fields:

        - **matrix**: A scipy.sparse.coo_matrix with shape
          (n_fragments, n_cpg_sites) where n_fragments is the number of
          fragments that passed filters and covered at least one CpG
          site, n_cpg_sites is
          genome_methylation_embedding.total_cpg_sites, and values are:
          1 (methylated), 0 (unmethylated), -1 (no data). Within a row,
          CpGs are in ascending column order. Values are stored as int64.
          The matrix uses COO format for efficient construction; convert
          to CSR (tocsr()) for row slicing or CSC (tocsc()) for column
          slicing.
        - **tlen**: A 1-D numpy int32 array of shape (n_fragments,)
          containing the length of each fragment (the absolute value of
          the BAM TLEN field). Values are 0 for single-end reads or reads
          with unmapped mates.
        - **stats**: A dict of counts describing what was decoded and
          what was discarded; see :class:`ExtractionResult`.

    Raises:
        FileNotFoundError: If the BAM file index (.bam.bai) is missing.
            The BAM file must be indexed with `samtools index`.
        ValueError: If ``threads`` is less than 1, or
            ``min_base_quality`` or ``mask_3prime`` is negative.

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
        >>> print(f"Fragments with CpG data: {result.matrix.shape[0]:,}")
        >>> print(f"Total CpG sites: {result.matrix.shape[1]:,}")
        >>> print(f"Data points: {result.matrix.nnz:,}")
        >>>
        >>> # Save to file
        >>> scipy.sparse.save_npz("sample.methylation.npz", result.matrix)

    Note:
        Rows are ordered by chromosome, in the order of the embedding's
        ``expected_chromosomes`` (the same order as the columns), and
        within a chromosome by where each fragment's first CpG-covering
        read appears in the (coordinate-sorted) BAM.

        A ``ValueError`` also propagates from the checks on the BAM
        header: no chromosome name in common with the reference, or a
        chromosome whose length differs from the reference.

    See Also:
        GenomeMethylationEmbedding: For creating the genome embedding.
        scipy.sparse.coo_matrix: Documentation on the output format.
    """
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    if min_base_quality < 0:
        raise ValueError(f"min_base_quality must be >= 0, got {min_base_quality}")
    if mask_3prime < 0:
        raise ValueError(f"mask_3prime must be >= 0, got {mask_3prime}")

    # Open once on the parent side to surface a clean error if the index
    # is missing, to validate chromosome-name overlap, and to read the
    # mapped-read count for verbose output. Workers open their own
    # handles. Sequential mode passes pysam_threads=4 (the historical
    # setting) to its handle; parallel mode uses 1 per worker since each
    # worker is its own process. The parent handle only touches metadata
    # so threads=1 here is sufficient.
    try:
        input_bam_object = pysam.AlignmentFile(  # type: ignore # pylint: disable=no-member
            input_bam, "rb", require_index=True, threads=1
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Index missing for bam file?: {input_bam}") from exc

    try:
        check_chromosome_overlap(
            bam_references=input_bam_object.references,
            embedding_chromosomes=list(
                genome_methylation_embedding.cpg_sites_dict.keys()
            ),
        )
        if not allow_length_mismatch:
            check_chromosome_lengths(
                bam_lengths=dict(
                    zip(input_bam_object.references, input_bam_object.lengths)
                ),
                reference_lengths=genome_methylation_embedding.chromosome_lengths,
            )
        if verbose:
            print(f"\tTotal reads: {input_bam_object.mapped:,}\n")
    finally:
        input_bam_object.close()

    n_cols = genome_methylation_embedding.total_cpg_sites

    config = _ExtractionConfig(
        pysam_threads=4 if threads == 1 else 1,
        quality_limit=quality_limit,
        filter_non_converted=filter_non_converted,
        non_converted_threshold=non_converted_threshold,
        filter_em_overconversion=filter_em_overconversion,
        em_overconversion_min_cpgs=em_overconversion_min_cpgs,
        min_base_quality=min_base_quality,
        mask_3prime=mask_3prime,
        keep_improper_pairs=keep_improper_pairs,
        debug=debug,
        col_typecode="i" if n_cols <= np.iinfo(np.int32).max else "q",
    )

    # One task per chromosome, in cpg_sites_dict order: the row order of the
    # output follows this order in both the sequential and parallel paths.
    chrom_tasks: list[_ChromTask] = [
        (chrom, cpg_sites, genome_methylation_embedding.chromosome_offsets[chrom])
        for chrom, cpg_sites in genome_methylation_embedding.cpg_sites_dict.items()
    ]

    def _iter_chrom_results() -> Iterator[tuple[str, _ChromExtractionResult]]:
        if threads == 1:
            for chrom, cpg_sites, col_offset in chrom_tasks:
                yield chrom, _extract_chromosome(
                    input_bam=input_bam,
                    chrom=chrom,
                    cpg_sites=cpg_sites,
                    col_offset=col_offset,
                    config=config,
                )
            return
        # Spawn is the safe start method on macOS (fork + pysam can hang
        # because pysam holds C-level locks across the fork). Spawn is
        # also the default on macOS for Python 3.8+, but be explicit.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=threads,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(input_bam, config),
        ) as executor:
            # map() preserves submission order — required so chromosomes are
            # concatenated in the same order as the sequential path. The
            # executor pickles tasks a few at a time as workers free up, so
            # only the chromosomes in flight are ever serialised at once.
            yield from zip(
                (task[0] for task in chrom_tasks),
                executor.map(_worker_extract_chromosome, chrom_tasks),
            )

    # Per-chromosome chunks, concatenated once at the end.
    col_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    cpgs_per_fragment_chunks: list[np.ndarray] = []
    tlen_chunks: list[np.ndarray] = []
    stats = {
        "reads_used": 0,
        "reads_dropped_improper_pair": 0,
        "reads_without_md": 0,
        "fragments_from_two_mates": 0,
        "cpg_observations": 0,
        "cpg_not_intact": 0,
        "cpg_reference_mismatch": 0,
        "cpg_low_base_quality": 0,
        "cpg_masked_3prime": 0,
        "strand_checks": 0,
        "strand_contradictions": 0,
    }

    for chrom, chrom_result in tqdm(
        _iter_chrom_results(), total=len(chrom_tasks), disable=not verbose
    ):
        if chrom_result.missing:
            if verbose:
                tqdm.write(f"Chromosome {chrom} not found in BAM file, skipping.")
            continue
        stats["reads_used"] += chrom_result.reads_used
        stats["reads_dropped_improper_pair"] += chrom_result.improper_pair_reads
        stats["fragments_from_two_mates"] += chrom_result.paired_fragments
        stats["reads_without_md"] += chrom_result.reads_without_md
        stats["cpg_observations"] += chrom_result.cpg_observations
        stats["cpg_not_intact"] += chrom_result.cpgs_not_intact
        stats["cpg_reference_mismatch"] += chrom_result.cpgs_mismatched
        stats["cpg_low_base_quality"] += chrom_result.cpgs_low_quality
        stats["cpg_masked_3prime"] += chrom_result.cpgs_masked
        stats["strand_checks"] += chrom_result.strand_checks
        stats["strand_contradictions"] += chrom_result.strand_contradictions
        if len(chrom_result.cpgs_per_fragment) == 0:
            continue
        col_chunks.append(chrom_result.cols)
        data_chunks.append(chrom_result.data)
        cpgs_per_fragment_chunks.append(chrom_result.cpgs_per_fragment)
        tlen_chunks.append(chrom_result.tlens)

    reads_without_md = stats["reads_without_md"]
    strand_checks = stats["strand_checks"]
    strand_contradictions = stats["strand_contradictions"]
    cpg_observations = stats["cpg_observations"]
    cpg_not_intact = stats["cpg_not_intact"]
    cpg_mismatch = stats["cpg_reference_mismatch"]

    if verbose:
        if stats["reads_dropped_improper_pair"]:
            print(
                f"\tImproper pairs: {stats['reads_dropped_improper_pair']:,} "
                "read(s) dropped."
            )
        if cpg_observations:
            print(
                f"\tCpG observations: {cpg_observations:,} decoded; "
                f"{cpg_not_intact:,} ({cpg_not_intact / cpg_observations:.2%}) "
                "not an intact CpG in the read, "
                f"{stats['cpg_low_base_quality']:,} below base quality "
                f"{min_base_quality}; {stats['cpg_masked_3prime']:,} call(s) "
                "in the masked 3' end of their fragment."
            )

    improper = stats["reads_dropped_improper_pair"]
    if (
        improper >= _FILTER_MAJORITY_MIN_COUNT
        and improper > _FILTER_MAJORITY_WARNING_RATE * (improper + stats["reads_used"])
    ):
        warnings.warn(
            f"{improper:,} of {improper + stats['reads_used']:,} otherwise usable "
            f"reads ({improper / (improper + stats['reads_used']):.1%}) in "
            f"{input_bam} are paired but not flagged as a proper pair, and were "
            "dropped. Ordinary libraries lose a few percent this way. If the "
            "aligner or a later tool does not set the proper-pair flag on this "
            "data, keep these reads with keep_improper_pairs "
            "(--keep-improper-pairs).",
            RuntimeWarning,
            stacklevel=2,
        )

    low_quality = stats["cpg_low_base_quality"]
    readable = cpg_observations - cpg_not_intact
    if (
        low_quality >= _FILTER_MAJORITY_MIN_COUNT
        and low_quality > _FILTER_MAJORITY_WARNING_RATE * readable
    ):
        warnings.warn(
            f"{low_quality:,} of {readable:,} CpG calls "
            f"({low_quality / readable:.1%}) in {input_bam} sit on bases below "
            f"the minimum base quality of {min_base_quality}, and were recorded "
            "as no data. Ordinary data loses a few percent this way. If this "
            "BAM carries placeholder base qualities, turn the filter off with "
            "min_base_quality=0 (--min-base-quality 0).",
            RuntimeWarning,
            stacklevel=2,
        )

    if reads_without_md:
        warnings.warn(
            f"{reads_without_md:,} read(s) in {input_bam} have no MD tag, so the "
            "non-converted filter could not evaluate them and they were kept. "
            "Add MD tags (e.g. with `samtools calmd`) to apply the filter to "
            "these reads.",
            RuntimeWarning,
            stacklevel=2,
        )

    if (
        cpg_observations >= _REFERENCE_MISMATCH_MIN_CPGS
        and cpg_mismatch > _REFERENCE_MISMATCH_WARNING_RATE * cpg_observations
    ):
        warnings.warn(
            f"{cpg_mismatch:,} of {cpg_observations:,} CpGs "
            f"({cpg_mismatch / cpg_observations:.1%}) covered by reads in "
            f"{input_bam} do not match the reference: the read shows something "
            "other than a CpG there. Sequencing errors and genetic variants "
            "account for a percent or two. A rate this high suggests the BAM "
            "was not aligned to the reference FASTA provided (the positions "
            "taken for CpGs are then not CpGs in the reads), in which case "
            "this output is meaningless; check that the reference is the one "
            "used for alignment.",
            RuntimeWarning,
            stacklevel=2,
        )

    if strand_checks:
        contradiction_rate = strand_contradictions / strand_checks
        if verbose:
            print(
                f"\tStrand-tag check: {strand_contradictions:,} of "
                f"{strand_checks:,} CpGs ({contradiction_rate:.3%}) contradict "
                "their read's strand tag."
            )
        if (
            strand_checks >= _STRAND_CHECK_MIN_CPGS
            and contradiction_rate > _STRAND_CONTRADICTION_WARNING_RATE
        ):
            warnings.warn(
                f"{strand_contradictions:,} of {strand_checks:,} CpGs "
                f"({contradiction_rate:.1%}) in {input_bam} show a conversion "
                "on the strand opposite to their read's strand tag. Correctly "
                "tagged reads do this only through sequencing errors and SNPs "
                "(well under 1%). A rate this high means the strand tags do "
                "not describe many of the reads, as happens when a "
                "non-directional library is aligned with an aligner that "
                "assumes a directional one (such as bwa-meth). In those reads "
                "only methylated CpGs are called (unmethylated ones are "
                "recorded as no data), so methylation is overstated and this "
                "output should not be trusted; re-align with a tool that "
                "models non-directional libraries (for example Bismark with "
                "--non_directional). A BAM decoded against a reference it was "
                "not aligned to also produces this pattern.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _concatenate(chunks: list[np.ndarray], dtype: type) -> np.ndarray:
        # Concatenate straight into the final dtype, then drop the chunks so
        # their memory is released before the next array is built.
        if not chunks:
            return np.empty(0, dtype=dtype)
        combined: np.ndarray = np.concatenate(chunks, dtype=dtype)
        chunks.clear()
        return combined

    cpgs_per_fragment = _concatenate(cpgs_per_fragment_chunks, np.int32)
    n_fragments = len(cpgs_per_fragment)

    # Build the index arrays directly in the dtype scipy will choose for a
    # matrix of this shape, so it adopts them instead of converting copies.
    index_dtype = scipy.sparse.get_index_dtype(maxval=max(n_fragments, n_cols))
    # Fragments were emitted in order, so each fragment's row index simply
    # repeats once per CpG it covers.
    coo_row = np.repeat(np.arange(n_fragments, dtype=index_dtype), cpgs_per_fragment)
    coo_col = _concatenate(col_chunks, index_dtype)
    # int64 values keep the saved .npz identical to earlier releases, which
    # built the matrix from Python ints.
    coo_data = _concatenate(data_chunks, np.int64)
    tlen_array = _concatenate(tlen_chunks, np.int32)

    if debug:
        print("Debug info for coo_matrix dimensions:")
        print(f"\tcoo_row: {len(coo_row):,}")
        if len(coo_row):
            print(f"\tcoo row max: {int(coo_row.max()):,}")
        print(f"\tcoo_col: {len(coo_col):,}")
        if len(coo_col):
            print(f"\tcoo col max: {int(coo_col.max()):,}")
        print(f"\tcoo_data: {len(coo_data):,}")
        print(f"\ttotal_cpg_sites: {n_cols:,}")

    sparse_matrix = scipy.sparse.coo_matrix(
        (coo_data, (coo_row, coo_col)),
        shape=(n_fragments, n_cols),
    )

    assert sparse_matrix.shape[1] == n_cols

    return ExtractionResult(matrix=sparse_matrix, tlen=tlen_array, stats=stats)
