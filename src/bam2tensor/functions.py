"""Core methylation extraction functions for bam2tensor.

This module contains the main function for extracting methylation data from
BAM files: extract_methylation_data_from_bam(). This function reads aligned
bisulfite-sequencing (BS-seq) and enzymatic methylation sequencing (EM-seq)
reads and generates a sparse matrix representation of methylation states at
each CpG site, with one row per sequenced DNA fragment.

The extraction process:
    1. Iterates through each chromosome in the genome embedding
    2. For each aligned read that passes quality filters:
       a. Determines which bisulfite strand the fragment derives from
       b. Identifies the CpG sites whose informative base the read covers
       c. Extracts the methylation state at each of those CpGs
    3. Combines the two mates of a read pair into a single fragment
    4. Builds a sparse COO matrix with one row per fragment

Methylation State Encoding:
    - 1: Methylated (cytosine remained as C after bisulfite treatment)
    - 0: Unmethylated (cytosine converted to T by bisulfite treatment)
    - -1: No data (indel, SNV, other non-C/T base at the CpG, or the two
      mates of a pair disagreeing)

Strands and the informative base:
    A fragment derives from either the top or the bottom strand of the
    original DNA, and both mates of a pair report on that same strand. BAM
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
    """

    matrix: scipy.sparse.coo_matrix
    tlen: np.ndarray


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
    """

    cols: np.ndarray
    data: np.ndarray
    cpgs_per_fragment: np.ndarray
    tlens: np.ndarray
    missing: bool
    reads_without_md: int = 0


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


# How far downstream (in bp) a read's mate may start and still be combined
# with it into one fragment. Reads are processed in coordinate order and rows
# are written in the order fragments first appear, so a fragment waiting for
# its mate holds back every fragment behind it. Real inserts are a few hundred
# bases; this generous cap only bounds how long a far-flung (chimeric or
# mis-annotated) pair can make the queue grow. Mates further apart than this
# are reported as two separate fragments.
_MAX_MATE_DISTANCE = 100_000


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
    :func:`_merge_mate_calls`). Fragments are written out in the order they
    were opened, each as soon as it and everything ahead of it is complete
    — a fragment stops waiting once its mate has been merged or the reads
    have moved past where the mate should have been.

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
    # opened, and the subset still waiting for a second mate (by query name).
    pending: collections.deque[_Fragment] = collections.deque()
    awaiting_mate: dict[str, _Fragment] = {}

    reads_without_md = 0

    def emit(fragment: _Fragment) -> None:
        # Write a finished fragment's row, unless a filter removed it.
        if fragment.dropped:
            return
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
        if aligned_segment.flag & _SKIP_FLAGS:
            continue

        # A read flagged as mapped but carrying no alignment (e.g. CIGAR
        # "*") has no reference span, so there is nothing to extract.
        reference_end = aligned_segment.reference_end
        if reference_end is None:
            continue
        reference_start = aligned_segment.reference_start

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
            if head.awaiting_mate and awaiting_mate.get(head.name) is head:
                del awaiting_mate[head.name]
            emit(head)

        # ============================================================
        # Which strand of the original DNA does this fragment derive
        # from? Both mates of a pair report on the same strand, so this
        # depends on the aligner's strand tag — never on the read's own
        # alignment orientation.
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
                    print(f"\t{aligned_segment.query_name}: no strand tag, ignoring.")
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
                    non_converted = (
                        count_non_cpg_retained_reference(aligned_segment, bottom_strand)
                        >= non_converted_threshold
                    )

        if non_converted:
            # Incomplete conversion is a property of the molecule, so the
            # whole fragment goes: a mate already queued is dropped, and a
            # mate still to come is discarded when it arrives.
            if debug:
                print(f"\tNon-converted filter: dropping {aligned_segment.query_name}.")
            name = aligned_segment.query_name
            assert name is not None
            partner = awaiting_mate.get(name)
            if partner is not None and partner.is_read1 != aligned_segment.is_read1:
                del awaiting_mate[name]
                partner.awaiting_mate = False
                partner.dropped = True
            elif partner is None:
                mate_start = _mate_start_if_expected(aligned_segment, reference_start)
                if mate_start is not None:
                    placeholder = _Fragment(name, aligned_segment.is_read1, [], [], 0)
                    placeholder.dropped = True
                    placeholder.awaiting_mate = True
                    placeholder.mate_start = mate_start
                    pending.append(placeholder)
                    awaiting_mate[name] = placeholder
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
            continue

        # 0-based reference position of each of those informative bases, and
        # where each falls in the read (absent if deleted from the read).
        informative_positions = [
            cpg_sites[i] - 1 + informative_offset for i in range(start_idx, end_idx)
        ]
        wanted = set(informative_positions)
        query_position = {
            ref_pos: query_pos
            for query_pos, ref_pos in aligned_segment.get_aligned_pairs(
                matches_only=True
            )
            if ref_pos in wanted
        }

        read_data: list[int] = []
        if xm_tag is not None:
            # Bismark: Z = methylated CpG, z = unmethylated CpG. Anything
            # else at a CpG (a mismatch, or a base deleted from the read)
            # is no data.
            xm_length = len(xm_tag)
            for ref_pos in informative_positions:
                query_pos = query_position.get(ref_pos)
                if query_pos is None or query_pos >= xm_length:
                    read_data.append(-1)
                else:
                    call = xm_tag[query_pos]
                    read_data.append(1 if call == "Z" else 0 if call == "z" else -1)
        else:
            # Biscuit / bwameth / gem3: read the base itself. BAM SEQ is in
            # top-strand orientation, so a top-strand fragment shows C
            # (methylated) or T (unmethylated) at the C of the CpG, and a
            # bottom-strand fragment shows G (methylated) or A
            # (unmethylated) at the G.
            sequence = aligned_segment.query_sequence
            if sequence is None:
                continue
            if bottom_strand:
                methylated_base, unmethylated_base = "G", "A"
            else:
                methylated_base, unmethylated_base = "C", "T"
            for ref_pos in informative_positions:
                query_pos = query_position.get(ref_pos)
                if query_pos is None:
                    read_data.append(-1)  # deleted from the read
                else:
                    base = sequence[query_pos]
                    read_data.append(
                        1
                        if base == methylated_base
                        else 0 if base == unmethylated_base else -1
                    )

        # The CpG at index i of the sorted cpg_sites is column col_offset + i.
        read_cols = list(range(col_offset + start_idx, col_offset + end_idx))

        name = aligned_segment.query_name
        assert name is not None
        is_read1 = aligned_segment.is_read1

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
        partner = awaiting_mate.get(name)
        if partner is not None and partner.is_read1 != is_read1:
            # Second mate of a fragment that is already queued.
            del awaiting_mate[name]
            partner.awaiting_mate = False
            if not partner.dropped:
                partner.cols, partner.data = _merge_mate_calls(
                    partner.cols, partner.data, read_cols, read_data
                )
                if partner.tlen == 0:
                    partner.tlen = abs(aligned_segment.template_length)
            continue

        # First read of its fragment to contribute calls.
        fragment = _Fragment(
            name, is_read1, read_cols, read_data, abs(aligned_segment.template_length)
        )
        pending.append(fragment)
        if partner is None:
            mate_start = _mate_start_if_expected(aligned_segment, reference_start)
            if mate_start is not None:
                fragment.awaiting_mate = True
                fragment.mate_start = mate_start
                awaiting_mate[name] = fragment

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
        - Reads without a recognised strand tag are excluded

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
          (``filter_em_overconversion``): drops fragments whose covered
          CpGs (across both mates) are all called unmethylated, a
          heuristic for the EM-seq artifact described by Loyfer et al.
          (bioRxiv 2026.03.24.713040).

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
        filter_em_overconversion: If True, drop fragments whose covered
            CpGs are all called unmethylated and cover at least
            ``em_overconversion_min_cpgs`` sites — the Loyfer et al.
            EM-seq fragment-level over-conversion heuristic. Default
            False.
        em_overconversion_min_cpgs: Minimum covered CpG count required
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

    Returns:
        An ExtractionResult named tuple with two fields:

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

    Raises:
        FileNotFoundError: If the BAM file index (.bam.bai) is missing.
            The BAM file must be indexed with `samtools index`.
        ValueError: If ``threads`` is less than 1.

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
        Rows are ordered by chromosome, in the order chromosomes appear in
        genome_methylation_embedding.cpg_sites_dict, and within a
        chromosome by where each fragment's first CpG-covering read
        appears in the (coordinate-sorted) BAM.

    See Also:
        GenomeMethylationEmbedding: For creating the genome embedding.
        scipy.sparse.coo_matrix: Documentation on the output format.
    """
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")

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

    check_chromosome_overlap(
        bam_references=input_bam_object.references,
        embedding_chromosomes=list(genome_methylation_embedding.cpg_sites_dict.keys()),
    )
    if verbose:
        print(f"\tTotal reads: {input_bam_object.mapped:,}\n")
    input_bam_object.close()

    n_cols = genome_methylation_embedding.total_cpg_sites

    config = _ExtractionConfig(
        pysam_threads=4 if threads == 1 else 1,
        quality_limit=quality_limit,
        filter_non_converted=filter_non_converted,
        non_converted_threshold=non_converted_threshold,
        filter_em_overconversion=filter_em_overconversion,
        em_overconversion_min_cpgs=em_overconversion_min_cpgs,
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
    reads_without_md = 0

    for chrom, chrom_result in tqdm(
        _iter_chrom_results(), total=len(chrom_tasks), disable=not verbose
    ):
        if chrom_result.missing:
            if verbose:
                tqdm.write(f"Chromosome {chrom} not found in BAM file, skipping.")
            continue
        reads_without_md += chrom_result.reads_without_md
        if len(chrom_result.cpgs_per_fragment) == 0:
            continue
        col_chunks.append(chrom_result.cols)
        data_chunks.append(chrom_result.data)
        cpgs_per_fragment_chunks.append(chrom_result.cpgs_per_fragment)
        tlen_chunks.append(chrom_result.tlens)

    if reads_without_md:
        warnings.warn(
            f"{reads_without_md:,} read(s) in {input_bam} have no MD tag, so the "
            "non-converted filter could not evaluate them and they were kept. "
            "Add MD tags (e.g. with `samtools calmd`) to apply the filter to "
            "these reads.",
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

    return ExtractionResult(matrix=sparse_matrix, tlen=tlen_array)
