"""Deterministic bisulfite-read simulator for integration tests.

The bundled ``test.bam`` is real aligner output in hg38 coordinates, so almost
none of its reads overlap the 700 kb ``test_fasta.fa``: extraction against it
yields a single row, which exercises very little. This module fills the gap by
*simulating* paired-end bisulfite reads directly from ``test_fasta.fa``:

* every fragment is assigned a strand of origin and a methylation state per
  CpG, and is bisulfite-converted accordingly;
* both mates are written to an indexed BAM in the tag dialect of a chosen
  aligner (Biscuit/bwameth ``YD``, or Bismark ``XM``/``XR``/``XG``);
* the ground truth for every read is returned alongside the BAM.

Tests can therefore check the extractor's output value-for-value against what
was simulated, instead of against numbers copied from a previous run.

Both mates of a pair are copies of the same converted molecule, so they agree
wherever they overlap — except where a simulated sequencing error says
otherwise. Errors are drawn independently for each mate and come in three
kinds: a base that is neither the methylated nor the unmethylated one (the CpG
reads as "no data"), a miscall that flips methylated to unmethylated or back
(the mate reports the wrong state, which only shows where the other mate
covers the same CpG and disagrees), and a misread of the CpG's *other* base
(the read no longer shows a CpG there, so the call is "no data").

Independently of the bases, a few percent of all bases are given a low base
quality, and a few percent of fragments are flagged as improper pairs, so that
the base-quality and proper-pair filters have something to act on. A read's
``cpg_calls`` describe its bases only; ``SimulatedRead.qualities``,
``improper`` and the fragment's ends are what the tests apply those filters
(and the 3' end mask) from.

Conventions (all coordinates 0-based unless stated otherwise):

* A fragment from the original **top** strand (OT) yields read 1 aligned
  forward and read 2 aligned reverse; a fragment from the original **bottom**
  strand (OB) yields read 1 reverse and read 2 forward.
* BAM stores every read in reference (top-strand) orientation. A top-strand
  fragment therefore shows ``C`` (methylated) or ``T`` (unmethylated) at the
  ``C`` of a CpG. A bottom-strand fragment shows ``G`` (methylated) or ``A``
  (unmethylated) at the ``G`` of the CpG, one base to the right. This holds
  for both mates of the pair.
* Bismark writes its ``Z``/``z`` call on the base that carries the
  information: the ``C`` for top-strand (``XG:Z:CT``) alignments and the ``G``
  for bottom-strand (``XG:Z:GA``) alignments.
"""

from __future__ import annotations

import bisect
import os
import random
from dataclasses import dataclass

import pysam
from Bio import SeqIO

READ_LENGTH = 100
MAX_FRAGMENT_LENGTH = 3 * READ_LENGTH

# Per-fragment / per-site event rates. Chosen so that a few hundred fragments
# contain every case the extractor distinguishes.
_P_METHYLATED = 0.7
_P_FULLY_UNMETHYLATED_FRAGMENT = 0.10
_P_NON_CONVERTED_FRAGMENT = 0.06
_P_DUPLICATE = 0.05
_P_LOW_MAPQ = 0.10
_P_LOW_MAPQ_ONE_MATE = 0.05
_P_SNV_AT_CPG = 0.03
_P_MISCALL_AT_CPG = 0.03
# Chance that the *other* base of a CpG — the one this strand never converts —
# is misread as the other strand's conversion product (G read as A on the top
# strand, C read as T on the bottom strand). The read then shows no CpG there,
# so the call is "no data"; it also gives correctly tagged reads the small
# rate of "contradictions" that real sequencing errors and SNPs produce, for
# the strand-tag check to tolerate.
_P_ERROR_AT_PARTNER_BASE = 0.005
# Chance that any one base of a read has a low base quality, or a middling
# one, rather than HIGH_BASE_QUALITY; and chance that a fragment is flagged as
# an improper pair. All are drawn from a second random stream, so they leave
# the bases of the simulated reads untouched.
_P_LOW_BASE_QUALITY = 0.03
_P_MEDIUM_BASE_QUALITY = 0.03
_P_IMPROPER_PAIR = 0.05
LOW_BASE_QUALITY = 10  # below the extractor's default limit of 20
MEDIUM_BASE_QUALITY = 25  # above it, below a stricter limit of 30
HIGH_BASE_QUALITY = 40

# A fragment is only simulated as non-converted if every one of its reads
# would then retain at least this many non-CpG cytosines. That keeps each
# read's status clear-cut: comfortably above the filter's default threshold
# of 3, with no borderline reads whose fate would hinge on a single base.
_MIN_RETAINED_IF_NON_CONVERTED = 8


@dataclass
class SimulatedRead:
    """One simulated read and the ground truth for it.

    Attributes:
        name: Query name (shared by both mates of a fragment).
        chrom: Reference name.
        start: 0-based leftmost reference position.
        sequence: Read sequence as stored in the BAM (reference orientation).
        is_read1: True for the first mate.
        is_reverse: True if the read aligns to the reverse strand (flag 0x10).
        bottom_strand: True if the fragment derives from the original bottom
            (OB) strand, False for the original top (OT) strand.
        mapq: Mapping quality.
        duplicate: True if the read is flagged as a PCR duplicate.
        non_converted: True if the fragment escaped bisulfite conversion (its
            non-CpG cytosines are retained).
        tlen: Signed template length (BAM TLEN).
        cpg_calls: ``(position, state)`` for every CpG whose informative
            base lies within the read, in reference order: the ``C`` of the
            CpG for a top-strand fragment, the ``G`` for a bottom-strand
            fragment. ``position`` is the 1-based position of the CpG's ``C``
            either way; ``state`` is what the read shows: ``1`` (methylated),
            ``0`` (unmethylated) or ``-1`` (a sequencing error left a base
            that is neither, or changed the CpG's other base so that the
            read shows no CpG). Base quality is not taken into account.
        retained_non_cpg: Number of unconverted non-CpG cytosines in the read
            (guanines, for the bottom strand). ``0`` unless ``non_converted``.
        mistagged: True if the read was written with the wrong strand tag
            (see ``mistagged_fraction``). Such a read is decoded on the wrong
            base, so ``cpg_calls`` does not describe the extractor's output
            for it.
        qualities: Phred base quality of every base of ``sequence``.
        improper: True if the pair is not flagged as a proper pair (BAM flag
            0x2 unset on both mates).
        fragment_start: 0-based start of the fragment the read comes from.
        fragment_end: 0-based end (exclusive) of that fragment.
    """

    name: str
    chrom: str
    start: int
    sequence: str
    is_read1: bool
    is_reverse: bool
    bottom_strand: bool
    mapq: int
    duplicate: bool
    non_converted: bool
    tlen: int
    cpg_calls: list[tuple[int, int]]
    retained_non_cpg: int
    mistagged: bool = False
    qualities: tuple[int, ...] = ()
    improper: bool = False
    fragment_start: int = 0
    fragment_end: int = 0


@dataclass
class SimulatedDataset:
    """A simulated BAM and the truth for every read in it.

    Attributes:
        bam_path: Path to the coordinate-sorted, indexed BAM.
        aligner: Tag dialect the BAM was written in.
        chromosomes: Reference names, in BAM header order.
        reads: Every read, in the order it appears in the BAM.
        paired: Whether the reads are paired-end.
    """

    bam_path: str
    aligner: str
    chromosomes: list[str]
    reads: list[SimulatedRead]
    paired: bool = True


def _draw_base_quality(rng: random.Random) -> int:
    """Draw one base quality: mostly high, occasionally middling or low."""
    draw = rng.random()
    if draw < _P_LOW_BASE_QUALITY:
        return LOW_BASE_QUALITY
    if draw < _P_LOW_BASE_QUALITY + _P_MEDIUM_BASE_QUALITY:
        return MEDIUM_BASE_QUALITY
    return HIGH_BASE_QUALITY


def _cpg_positions(reference: str) -> list[int]:
    """0-based positions of the ``C`` of every CpG in ``reference``."""
    positions = []
    pos = reference.find("CG")
    while pos != -1:
        positions.append(pos)
        pos = reference.find("CG", pos + 1)
    return positions


def _non_cpg_cytosines(
    reference: str, start: int, end: int, bottom_strand: bool
) -> int:
    """Count non-CpG cytosines of a read's bisulfite strand within [start, end).

    These are the bases bisulfite conversion acts on regardless of
    methylation: reference ``C`` not followed by ``G`` for the top strand,
    reference ``G`` not preceded by ``C`` for the bottom strand.
    """
    if bottom_strand:
        return sum(
            1
            for pos in range(start, end)
            if reference[pos] == "G" and reference[pos - 1] != "C"
        )
    return sum(
        1
        for pos in range(start, end)
        if reference[pos] == "C" and reference[pos + 1] != "G"
    )


def _md_tag(reference: str, read: str) -> str:
    """MD tag for an ungapped alignment of ``read`` against ``reference``."""
    parts = []
    run = 0
    for ref_base, read_base in zip(reference, read):
        if ref_base == read_base:
            run += 1
        else:
            parts.append(f"{run}{ref_base}")
            run = 0
    parts.append(str(run))
    return "".join(parts)


def _bismark_xm(reference: str, start: int, read: str, bottom_strand: bool) -> str:
    """Bismark methylation-call string for an ungapped read.

    Top-strand alignments are called on reference ``C`` (read ``C`` = retained,
    ``T`` = converted) with context taken from the two bases downstream.
    Bottom-strand alignments are called on reference ``G`` (read ``G`` =
    retained, ``A`` = converted) with context taken from the two bases
    upstream — i.e. downstream on the bottom strand.
    """
    calls = []
    for offset, read_base in enumerate(read):
        pos = start + offset
        if bottom_strand:
            target, retained, converted = "G", "G", "A"
            near = reference[pos - 1] if pos >= 1 else "N"
            far = reference[pos - 2] if pos >= 2 else "N"
            partner = "C"
        else:
            target, retained, converted = "C", "C", "T"
            near = reference[pos + 1] if pos + 1 < len(reference) else "N"
            far = reference[pos + 2] if pos + 2 < len(reference) else "N"
            partner = "G"

        if reference[pos] != target or read_base not in (retained, converted):
            calls.append(".")
            continue
        if near == partner:
            letter = "z"  # CpG
        elif near == "N":
            letter = "u"  # unknown context
        elif far == partner:
            letter = "x"  # CHG
        elif far == "N":
            letter = "u"
        else:
            letter = "h"  # CHH
        calls.append(letter.upper() if read_base == retained else letter)
    return "".join(calls)


def simulate_bam(
    out_dir: str,
    fasta_path: str,
    aligner: str,
    chromosomes: list[str],
    fragments_per_chromosome: int = 150,
    seed: int = 0,
    strands: tuple[str, ...] = ("top", "bottom"),
    paired: bool = True,
    bismark_xg_tag: bool = True,
    mistagged_fraction: float = 0.0,
) -> SimulatedDataset:
    """Simulate bisulfite reads and write them as an indexed BAM.

    Args:
        out_dir: Existing directory to write ``simulated.<aligner>.bam`` into.
        fasta_path: Reference FASTA to draw fragments from.
        aligner: ``"biscuit"`` (``YD`` strand tag, also what bwameth writes) or
            ``"bismark"`` (``XM`` call string plus ``XR``/``XG``).
        chromosomes: Reference names to simulate, in BAM header order.
        fragments_per_chromosome: Number of read pairs per chromosome.
        seed: Seed for the random number generator; the same arguments always
            produce the same reads.
        strands: Strands of origin to draw fragments from: any of ``"top"``
            and ``"bottom"``.
        paired: If True (default) write both mates of each fragment. If
            False write single-end reads: only read 1, unpaired.
        bismark_xg_tag: Bismark dialect only. If False, omit the ``XG``
            genome-strand tag, as a Bismark-style BAM from another tool might.
        mistagged_fraction: Biscuit dialect only. Fraction of fragments whose
            reads are written with the opposite ``YD`` strand tag, as bwa-meth
            does to most complementary-strand pairs of a non-directional
            library. The bases are unchanged; only the tag is wrong.

    Returns:
        The path of the BAM together with the truth for every read.

    Raises:
        ValueError: If ``aligner`` or ``strands`` is not recognised, or if
            ``mistagged_fraction`` is used with the Bismark dialect.
    """
    if aligner not in ("biscuit", "bismark"):
        raise ValueError(f"Unknown aligner dialect: {aligner!r}")
    if mistagged_fraction and aligner != "biscuit":
        raise ValueError("mistagged_fraction applies to the biscuit dialect only")
    if not strands or any(strand not in ("top", "bottom") for strand in strands):
        raise ValueError(f"Invalid strands: {strands!r}")

    references = {
        record.id: str(record.seq).upper()
        for record in SeqIO.parse(fasta_path, "fasta")
        if record.id in chromosomes
    }
    rng = random.Random(seed)
    # Base qualities and pair flags come from their own stream: see
    # _P_LOW_BASE_QUALITY.
    extra_rng = random.Random(f"{seed}-qualities-and-flags")
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": name, "LN": len(references[name])} for name in chromosomes],
        }
    )

    # (sort key, read, mate start) — sorted into coordinate order before writing.
    records: list[tuple[tuple[int, int], SimulatedRead, int]] = []

    for chrom_index, chrom in enumerate(chromosomes):
        reference = references[chrom]
        cpgs = _cpg_positions(reference)

        for fragment_index in range(fragments_per_chromosome):
            fragment_length = rng.randrange(READ_LENGTH, MAX_FRAGMENT_LENGTH)
            while True:
                fragment_start = rng.randrange(2, len(reference) - fragment_length - 2)
                fragment_end = fragment_start + fragment_length
                if "N" not in reference[fragment_start - 2 : fragment_end + 2]:
                    break

            bottom_strand = rng.choice(strands) == "bottom"
            mapq = rng.randrange(0, 20) if rng.random() < _P_LOW_MAPQ else 60
            # Occasionally only one mate maps poorly: the other must then
            # stand alone as its fragment.
            low_mapq_mate = (
                rng.randrange(2) if rng.random() < _P_LOW_MAPQ_ONE_MATE else None
            )
            duplicate = rng.random() < _P_DUPLICATE
            mistagged = mistagged_fraction > 0 and rng.random() < mistagged_fraction
            non_converted = rng.random() < _P_NON_CONVERTED_FRAGMENT
            fully_unmethylated = rng.random() < _P_FULLY_UNMETHYLATED_FRAGMENT
            improper = paired and extra_rng.random() < _P_IMPROPER_PAIR

            # Methylation state of every CpG touching the fragment, keyed by
            # the position of its C. (Start one base early: a bottom-strand
            # read can begin on the G of a CpG whose C lies just outside.)
            first = bisect.bisect_left(cpgs, fragment_start - 1)
            last = bisect.bisect_left(cpgs, fragment_end)
            methylated = {
                cpg: (not fully_unmethylated) and rng.random() < _P_METHYLATED
                for cpg in cpgs[first:last]
            }

            # Left mate is forward, right mate is reverse. Which of them is
            # read 1 depends on the strand the fragment came from.
            mate_starts = (fragment_start, fragment_end - READ_LENGTH)

            if non_converted and any(
                _non_cpg_cytosines(reference, start, start + READ_LENGTH, bottom_strand)
                < _MIN_RETAINED_IF_NON_CONVERTED
                for start in mate_starts
            ):
                non_converted = False
            for mate_index, start in enumerate(mate_starts):
                end = start + READ_LENGTH
                is_left = mate_index == 0
                is_reverse = not is_left
                is_read1 = is_left != bottom_strand
                if not paired and not is_read1:
                    continue

                # Bisulfite conversion, expressed in reference orientation.
                bases = []
                retained_non_cpg = 0
                for pos in range(start, end):
                    base = reference[pos]
                    if not bottom_strand and base == "C":
                        if reference[pos + 1] == "G":
                            base = "C" if methylated[pos] else "T"
                        elif non_converted:
                            retained_non_cpg += 1
                        else:
                            base = "T"
                    elif bottom_strand and base == "G":
                        if reference[pos - 1] == "C":
                            base = "G" if methylated[pos - 1] else "A"
                        elif non_converted:
                            retained_non_cpg += 1
                        else:
                            base = "A"
                    bases.append(base)

                # What the read shows at each CpG whose informative base it
                # covers — the C for a top-strand fragment, the G (one base
                # to the right) for a bottom-strand fragment — after
                # sequencing errors, which are drawn per mate.
                informative_offset = 1 if bottom_strand else 0
                if bottom_strand:
                    methylated_base, unmethylated_base, other_base = "G", "A", "C"
                else:
                    methylated_base, unmethylated_base, other_base = "C", "T", "A"
                cpg_calls = []
                first_cpg = bisect.bisect_left(cpgs, start - informative_offset)
                last_cpg = bisect.bisect_left(cpgs, end - informative_offset)
                for cpg in cpgs[first_cpg:last_cpg]:
                    index = cpg + informative_offset - start
                    error = rng.random()
                    if error < _P_SNV_AT_CPG:
                        bases[index] = other_base
                        state = -1
                    elif error < _P_SNV_AT_CPG + _P_MISCALL_AT_CPG:
                        state = 0 if methylated[cpg] else 1
                        bases[index] = methylated_base if state else unmethylated_base
                    else:
                        state = 1 if methylated[cpg] else 0
                    cpg_calls.append((cpg + 1, state))

                    # Occasionally misread the CpG's other base (see
                    # _P_ERROR_AT_PARTNER_BASE), where the read covers it. The
                    # read then shows no CpG here, whatever the call was.
                    partner = cpg if bottom_strand else cpg + 1
                    if start <= partner < end and (
                        rng.random() < _P_ERROR_AT_PARTNER_BASE
                    ):
                        bases[partner - start] = "T" if bottom_strand else "A"
                        cpg_calls[-1] = (cpg + 1, -1)

                read = SimulatedRead(
                    name=f"{chrom}_fragment{fragment_index:04d}",
                    chrom=chrom,
                    start=start,
                    sequence="".join(bases),
                    is_read1=is_read1,
                    is_reverse=is_reverse,
                    bottom_strand=bottom_strand,
                    mapq=(
                        rng.randrange(0, 20) if low_mapq_mate == mate_index else mapq
                    ),
                    duplicate=duplicate,
                    non_converted=non_converted,
                    tlen=(
                        0
                        if not paired
                        else fragment_length if is_left else -fragment_length
                    ),
                    cpg_calls=cpg_calls,
                    retained_non_cpg=retained_non_cpg,
                    mistagged=mistagged,
                    qualities=tuple(
                        _draw_base_quality(extra_rng) for _ in range(READ_LENGTH)
                    ),
                    improper=improper,
                    fragment_start=fragment_start,
                    fragment_end=fragment_end,
                )
                mate_start = mate_starts[1 - mate_index]
                records.append(((chrom_index, start), read, mate_start))

    records.sort(key=lambda record: record[0])  # stable: ties keep insertion order

    bam_path = os.path.join(out_dir, f"simulated.{aligner}.bam")
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
        for (chrom_index, _), read, mate_start in records:
            reference = references[read.chrom]
            segment = pysam.AlignedSegment(header)
            segment.query_name = read.name
            segment.query_sequence = read.sequence
            segment.query_qualities = pysam.qualitystring_to_array(
                "".join(chr(quality + 33) for quality in read.qualities)
            )
            if paired:
                segment.flag = (
                    0x1  # paired
                    | (0 if read.improper else 0x2)  # proper pair
                    | (0x10 if read.is_reverse else 0x20)  # self / mate reverse
                    | (0x40 if read.is_read1 else 0x80)
                    | (0x400 if read.duplicate else 0)
                )
            else:
                segment.flag = (0x10 if read.is_reverse else 0) | (
                    0x400 if read.duplicate else 0
                )
            segment.reference_id = chrom_index
            segment.reference_start = read.start
            segment.mapping_quality = read.mapq
            segment.cigartuples = [(0, READ_LENGTH)]
            if paired:
                segment.next_reference_id = chrom_index
                segment.next_reference_start = mate_start
            segment.template_length = read.tlen

            reference_span = reference[read.start : read.start + READ_LENGTH]
            segment.set_tag(
                "NM", sum(a != b for a, b in zip(reference_span, read.sequence))
            )
            segment.set_tag("MD", _md_tag(reference_span, read.sequence))
            if aligner == "biscuit":
                tagged_bottom = read.bottom_strand != read.mistagged
                segment.set_tag("YD", "r" if tagged_bottom else "f")
            else:
                segment.set_tag(
                    "XM",
                    _bismark_xm(
                        reference, read.start, read.sequence, read.bottom_strand
                    ),
                )
                segment.set_tag("XR", "CT" if read.is_read1 else "GA")
                if bismark_xg_tag:
                    segment.set_tag("XG", "GA" if read.bottom_strand else "CT")
            bam.write(segment)
    pysam.index(bam_path)

    return SimulatedDataset(
        bam_path=bam_path,
        aligner=aligner,
        chromosomes=list(chromosomes),
        reads=[read for _, read, _ in records],
        paired=paired,
    )
