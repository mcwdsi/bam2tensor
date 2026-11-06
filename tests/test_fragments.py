"""Fragment assembly: how reads become rows.

Each row of the output is one sequenced DNA fragment. These tests build small
BAMs by hand to pin down each rule: which mates are combined, how their calls
are merged where they overlap, what happens when a mate is missing or
filtered, the order rows come out in, and how the opt-in filters apply to a
fragment rather than a read.

``tests/test_simulated.py`` checks the same behaviour in bulk against
simulated truth; here every case is spelled out.
"""

import re
import warnings

import numpy as np
import pysam
import pytest

from bam2tensor import embedding, functions
from bam2tensor.functions import (
    _bismark_is_bottom_strand,
    _merge_mate_calls,
)

# 154 bp reference with four CpGs. 0-based positions of each CpG's C and G:
REFERENCE = (
    "N" * 9 + "CG" + "N" * 9 + "CG" + "N" * 79 + "CG" + "N" * 9 + "CG" + "N" * 40
)
CPG_C = [9, 20, 101, 112]
CPG_G = [10, 21, 102, 113]
assert [REFERENCE[c : c + 2] for c in CPG_C] == ["CG"] * 4
assert len(REFERENCE) == 154

PAIRED, PROPER, REVERSE, MATE_REVERSE, READ1, READ2 = 0x1, 0x2, 0x10, 0x20, 0x40, 0x80


class Reads:
    """Collects hand-built reads and writes them as an indexed BAM."""

    def __init__(self, tmp_path, chromosomes=("chr1",)):
        self.tmp_path = tmp_path
        self.chromosomes = list(chromosomes)
        self.header = pysam.AlignmentHeader.from_dict(
            {
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [{"SN": name, "LN": len(REFERENCE)} for name in self.chromosomes],
            }
        )
        self.segments = []

    def read(
        self,
        name,
        start,
        end,
        flag=0,
        strand="f",
        bases=None,
        xm=None,
        xg=None,
        mapq=60,
        mate_start=None,
        mate_chrom=None,
        tlen=0,
        chrom="chr1",
        cigar=None,
        qualities=None,
    ):
        """Add one read covering reference [start, end).

        ``bases`` maps 0-based reference positions to read bases. ``xm``
        likewise maps reference positions to Bismark call letters, turning
        the read into a Bismark-style one; otherwise it carries a ``YD`` tag
        with the given ``strand``. Positions are reference positions, so
        with a custom ``cigar`` they refer to an ungapped layout of the query
        starting at ``start``.

        Bases not given are N, with one exception: the *other* base of each
        CpG (the G for a top-strand read, the C for a bottom-strand read),
        which reads as the reference has it, so that the read shows an
        intact CpG wherever a test gives it a call. ``qualities`` maps
        reference positions to Phred base qualities (40 elsewhere).
        """
        length = end - start
        sequence = ["N"] * length
        if xm is None:
            bottom_strand = strand == "r"
        elif xg is not None:
            bottom_strand = xg == "GA"
        else:
            # As the extractor infers it: from the bases the calls sit on.
            bottom_strand = any(
                (bases or {}).get(position) in ("G", "A") for position in xm
            )
        for c, g in zip(CPG_C, CPG_G):
            other = c if bottom_strand else g
            if start <= other < end:
                sequence[other - start] = REFERENCE[other]
        for position, base in (bases or {}).items():
            sequence[position - start] = base
        segment = pysam.AlignedSegment(self.header)
        segment.query_name = name
        segment.query_sequence = "".join(sequence)
        base_qualities = [40] * length
        for position, quality in (qualities or {}).items():
            base_qualities[position - start] = quality
        segment.query_qualities = pysam.qualitystring_to_array(
            "".join(chr(quality + 33) for quality in base_qualities)
        )
        segment.flag = flag
        segment.reference_id = self.chromosomes.index(chrom)
        segment.reference_start = start
        segment.mapping_quality = mapq
        segment.cigartuples = cigar or [(0, length)]
        if mate_start is not None:
            segment.next_reference_id = self.chromosomes.index(mate_chrom or chrom)
            segment.next_reference_start = mate_start
        segment.template_length = tlen
        if xm is None:
            segment.set_tag("YD", strand)
        else:
            calls = ["."] * length
            for position, letter in xm.items():
                calls[position - start] = letter
            segment.set_tag("XM", "".join(calls))
            if xg is not None:
                segment.set_tag("XG", xg)
        self.segments.append(segment)
        return segment

    def pair(
        self,
        name,
        left,
        right,
        strand="f",
        left_bases=None,
        right_bases=None,
        left_kwargs: dict | None = None,
        right_kwargs: dict | None = None,
    ):
        """Add a proper pair: ``left`` and ``right`` are (start, end) spans.

        Follows a directional library: for a top-strand fragment
        (``strand="f"``) the left, forward mate is read 1; for a bottom-strand
        fragment (``strand="r"``) the right, reverse mate is read 1.
        """
        left_is_read1 = strand == "f"
        fragment_length = right[1] - left[0]
        left_settings: dict = dict(
            flag=PAIRED | PROPER | MATE_REVERSE | (READ1 if left_is_read1 else READ2),
            strand=strand,
            bases=left_bases,
            mate_start=right[0],
            tlen=fragment_length,
        )
        left_settings.update(left_kwargs or {})
        self.read(name, left[0], left[1], **left_settings)

        right_settings: dict = dict(
            flag=PAIRED | PROPER | REVERSE | (READ2 if left_is_read1 else READ1),
            strand=strand,
            bases=right_bases,
            mate_start=left[0],
            tlen=-fragment_length,
        )
        right_settings.update(right_kwargs or {})
        self.read(name, right[0], right[1], **right_settings)

    def extract(self, **kwargs):
        """Write the BAM, run the extractor, and return its result."""
        fasta_path = self.tmp_path / "ref.fa"
        with open(fasta_path, "w") as fasta:
            for name in self.chromosomes:
                fasta.write(f">{name}\n{REFERENCE}\n")
        emb = embedding.GenomeMethylationEmbedding(
            "fragments",
            expected_chromosomes=self.chromosomes,
            fasta_source=str(fasta_path),
            skip_cache=True,
        )
        bam_path = str(self.tmp_path / "reads.bam")
        # Coordinate order; Python's sort is stable, so ties keep the order
        # in which the reads were added.
        ordered = sorted(
            self.segments, key=lambda s: (s.reference_id, s.reference_start)
        )
        with pysam.AlignmentFile(bam_path, "wb", header=self.header) as bam:
            for segment in ordered:
                bam.write(segment)
        pysam.index(bam_path)
        return functions.extract_methylation_data_from_bam(
            input_bam=bam_path, genome_methylation_embedding=emb, **kwargs
        )


def rows_of(result):
    """The matrix as a list of ``{column: state}`` dicts, one per row."""
    rows: list[dict[int, int]] = [{} for _ in range(result.matrix.shape[0])]
    for row, col, state in zip(
        result.matrix.row.tolist(),
        result.matrix.col.tolist(),
        result.matrix.data.tolist(),
    ):
        assert col not in rows[row], "a CpG must not be stored twice in a row"
        rows[row][col] = state
    return rows


@pytest.fixture
def reads(tmp_path):
    """A fresh read collection on a single chromosome."""
    return Reads(tmp_path)


# ---------------------------------------------------------------------------
# Mates are combined
# ---------------------------------------------------------------------------


def test_non_overlapping_mates_become_one_row(reads) -> None:
    """Read 1 covers the first two CpGs, read 2 the last two: one row of four."""
    reads.pair(
        "frag",
        left=(0, 60),
        right=(94, 154),
        left_bases={9: "C", 20: "T"},
        right_bases={101: "T", 112: "C"},
    )
    result = reads.extract()

    assert rows_of(result) == [{0: 1, 1: 0, 2: 0, 3: 1}]
    assert result.tlen.tolist() == [154]


@pytest.mark.parametrize("strand", ["f", "r"])
def test_both_mates_of_either_strand_are_used(reads, strand) -> None:
    """Top- and bottom-strand pairs alike contribute both mates.

    For the bottom strand the informative base is the G of each CpG.
    """
    if strand == "f":
        left_bases, right_bases = {9: "C", 20: "C"}, {101: "T", 112: "T"}
    else:
        left_bases, right_bases = {10: "G", 21: "G"}, {102: "A", 113: "A"}
    reads.pair(
        "frag",
        left=(0, 60),
        right=(94, 154),
        strand=strand,
        left_bases=left_bases,
        right_bases=right_bases,
    )
    assert rows_of(reads.extract()) == [{0: 1, 1: 1, 2: 0, 3: 0}]


OVERLAP_CASES = [
    # (read 1 base, read 2 base, expected state) at a CpG both mates cover
    pytest.param("C", "C", 1, id="both-methylated"),
    pytest.param("T", "T", 0, id="both-unmethylated"),
    pytest.param("C", "T", -1, id="disagree"),
    pytest.param("T", "C", -1, id="disagree-reversed"),
    pytest.param("C", "A", 1, id="only-first-has-a-call"),
    pytest.param("G", "T", 0, id="only-second-has-a-call"),
    pytest.param("A", "G", -1, id="neither-has-a-call"),
]


@pytest.mark.parametrize("first, second, expected", OVERLAP_CASES)
def test_overlapping_mates_record_each_cpg_once(reads, first, second, expected) -> None:
    """A CpG covered by both mates is one entry, merged from the two calls.

    Agreement keeps the call; a lone call is kept; a methylated/unmethylated
    disagreement is a sequencing error in one mate or the other, so it is
    recorded as no data.
    """
    # Both mates cover CpG 2 (C at 101); read 1 alone covers CpG 0 and 1,
    # read 2 alone covers CpG 3.
    reads.pair(
        "frag",
        left=(0, 110),
        right=(95, 154),
        left_bases={9: "C", 20: "T", 101: first},
        right_bases={101: second, 112: "C"},
    )
    result = reads.extract()

    assert rows_of(result) == [{0: 1, 1: 0, 2: expected, 3: 1}]
    assert result.matrix.nnz == 4


def test_fully_overlapping_mates(reads) -> None:
    """Mates covering exactly the same bases (a short fragment) merge cleanly."""
    bases = {9: "C", 20: "T", 101: "C", 112: "T"}
    reads.pair(
        "frag", left=(0, 154), right=(0, 154), left_bases=bases, right_bases=bases
    )
    result = reads.extract()

    assert rows_of(result) == [{0: 1, 1: 0, 2: 1, 3: 0}]
    assert result.tlen.tolist() == [154]


def test_mate_roles_do_not_matter(tmp_path) -> None:
    """The row is the same whichever mate is read 1 and whichever comes first."""
    outcomes = []
    for index, strand in enumerate(["f", "r"]):
        collection = Reads(tmp_path / str(index))
        (tmp_path / str(index)).mkdir()
        # Same calls expressed on the informative base of each strand.
        shift = 0 if strand == "f" else 1
        methylated, unmethylated = ("C", "T") if strand == "f" else ("G", "A")
        collection.pair(
            "frag",
            left=(0, 110),
            right=(95, 154),
            strand=strand,
            left_bases={9 + shift: methylated, 101 + shift: unmethylated},
            right_bases={101 + shift: unmethylated, 112 + shift: methylated},
        )
        outcomes.append(rows_of(collection.extract()))
    assert outcomes[0] == outcomes[1] == [{0: 1, 1: -1, 2: 0, 3: 1}]


def test_mate_without_cpgs_adds_nothing(reads) -> None:
    """A mate that covers no CpG leaves the fragment as its partner found it."""
    reads.pair(
        "frag",
        left=(0, 60),
        right=(30, 90),  # no CpG between 30 and 90
        left_bases={9: "C", 20: "T"},
    )
    result = reads.extract()
    assert rows_of(result) == [{0: 1, 1: 0}]
    assert result.tlen.tolist() == [90]


def test_fragment_opened_by_its_second_mate(reads) -> None:
    """If only the right-hand mate covers CpGs, it alone forms the row."""
    reads.pair(
        "frag",
        left=(30, 90),  # no CpG
        right=(94, 154),
        right_bases={101: "C", 112: "T"},
    )
    result = reads.extract()
    assert rows_of(result) == [{2: 1, 3: 0}]
    # The right-hand mate's TLEN is negative in the BAM; the row reports the
    # fragment length.
    assert result.tlen.tolist() == [124]


# ---------------------------------------------------------------------------
# Mates that are not combined
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_mate_kwargs",
    [
        pytest.param({"mapq": 3}, id="low-mapq"),
        pytest.param({"cigar": None, "mapq": 0}, id="mapq-zero"),
    ],
)
def test_filtered_mate_leaves_the_other_as_the_fragment(reads, bad_mate_kwargs) -> None:
    """When one mate fails the read filters, the other is the whole fragment."""
    reads.pair(
        "frag",
        left=(0, 60),
        right=(94, 154),
        left_bases={9: "C", 20: "T"},
        right_bases={101: "C", 112: "C"},
        right_kwargs=bad_mate_kwargs,
    )
    assert rows_of(reads.extract()) == [{0: 1, 1: 0}]


@pytest.mark.parametrize("flag_bit", [0x400, 0x200], ids=["duplicate", "qc-fail"])
def test_flagged_mate_leaves_the_other_as_the_fragment(reads, flag_bit) -> None:
    """A duplicate or QC-failed mate is skipped without taking its partner."""
    reads.pair(
        "frag",
        left=(0, 60),
        right=(94, 154),
        left_bases={9: "C", 20: "T"},
        right_bases={101: "C", 112: "C"},
    )
    reads.segments[0].flag |= flag_bit  # the left mate
    assert rows_of(reads.extract()) == [{2: 1, 3: 1}]


def test_mate_missing_from_the_bam(reads) -> None:
    """A read whose mate is announced but absent still yields its own row.

    The fragment waits for the mate only until the reads have moved past the
    position where the mate should have started.
    """
    reads.read(
        "orphan",
        0,
        60,
        flag=PAIRED | PROPER | MATE_REVERSE | READ1,
        bases={9: "C", 20: "T"},
        mate_start=94,
        tlen=154,
    )
    reads.read("later", 95, 154, bases={101: "T", 112: "T"})
    result = reads.extract()

    assert rows_of(result) == [{0: 1, 1: 0}, {2: 0, 3: 0}]
    assert result.tlen.tolist() == [154, 0]


def test_mate_missing_from_the_bam_at_end_of_chromosome(reads) -> None:
    """Fragments still waiting when the chromosome ends are written out."""
    reads.read(
        "orphan",
        0,
        60,
        flag=PAIRED | PROPER | MATE_REVERSE | READ1,
        bases={9: "C", 20: "T"},
        mate_start=94,
        tlen=154,
    )
    assert rows_of(reads.extract()) == [{0: 1, 1: 0}]


def test_mates_on_different_chromosomes_are_separate_fragments(tmp_path) -> None:
    """A mate mapped to another chromosome is not combined.

    Such a pair is never a proper pair, so it is only seen at all when
    improper pairs are kept.
    """
    reads = Reads(tmp_path, chromosomes=("chr1", "chr2"))
    reads.read(
        "split",
        0,
        60,
        flag=PAIRED | MATE_REVERSE | READ1,
        bases={9: "C", 20: "C"},
        mate_start=94,
        mate_chrom="chr2",
    )
    reads.read(
        "split",
        94,
        154,
        flag=PAIRED | REVERSE | READ2,
        bases={101: "T", 112: "T"},
        mate_start=0,
        mate_chrom="chr1",
        chrom="chr2",
    )
    assert rows_of(reads.extract()) == []

    result = reads.extract(keep_improper_pairs=True)
    # chr2's CpGs are columns 4-7.
    assert rows_of(result) == [{0: 1, 1: 1}, {6: 0, 7: 0}]
    for threads in (2,):
        assert rows_of(
            reads.extract(keep_improper_pairs=True, threads=threads)
        ) == rows_of(result)


def test_mates_too_far_apart_are_separate_fragments(reads, monkeypatch) -> None:
    """Beyond the mate-distance cap each mate is reported on its own."""
    monkeypatch.setattr(functions, "_MAX_MATE_DISTANCE", 50)
    reads.pair(
        "far",
        left=(0, 60),
        right=(94, 154),  # starts 94 bp downstream: past the (patched) cap
        left_bases={9: "C", 20: "T"},
        right_bases={101: "C", 112: "T"},
    )
    assert rows_of(reads.extract()) == [{0: 1, 1: 0}, {2: 1, 3: 0}]

    monkeypatch.setattr(functions, "_MAX_MATE_DISTANCE", 94)  # exactly at the cap
    assert rows_of(reads.extract()) == [{0: 1, 1: 0, 2: 1, 3: 0}]


def test_real_mate_distance_cap_is_generous() -> None:
    """The cap is far above any real insert size."""
    assert functions._MAX_MATE_DISTANCE >= 10_000


def test_unpaired_reads_sharing_a_name_stay_separate(reads) -> None:
    """Single-end reads are never combined, even with identical names."""
    reads.read("same", 0, 60, bases={9: "C", 20: "T"})
    reads.read("same", 94, 154, flag=REVERSE, bases={101: "C", 112: "T"})
    assert rows_of(reads.extract()) == [{0: 1, 1: 0}, {2: 1, 3: 0}]


def test_two_first_mates_with_one_name_are_not_merged(reads) -> None:
    """Only a read 1 and a read 2 form a pair; two read 1s do not."""
    for start, end, bases in (
        (0, 60, {9: "C", 20: "C"}),
        (94, 154, {101: "T", 112: "T"}),
    ):
        reads.read(
            "odd",
            start,
            end,
            flag=PAIRED | PROPER | READ1,
            bases=bases,
            mate_start=94,
            tlen=154,
        )
    assert rows_of(reads.extract()) == [{0: 1, 1: 1}, {2: 0, 3: 0}]


# ---------------------------------------------------------------------------
# Row order
# ---------------------------------------------------------------------------


def test_rows_are_ordered_by_first_appearance(reads) -> None:
    """Rows follow the order fragments start in, not the order they finish in.

    Fragment A starts first but its second mate comes last; B and the
    single-end read C are complete before A is. The rows are still A, B, C.
    """
    reads.pair(
        "A", left=(0, 15), right=(105, 154), left_bases={9: "C"}, right_bases={112: "C"}
    )
    reads.pair(
        "B",
        left=(15, 40),
        right=(90, 110),
        left_bases={20: "T"},
        right_bases={101: "T"},
    )
    reads.read("C", 15, 60, bases={20: "C"})
    result = reads.extract()

    assert rows_of(result) == [{0: 1, 3: 1}, {1: 0, 2: 0}, {1: 1}]
    assert result.tlen.tolist() == [154, 95, 0]


def test_parallel_and_serial_agree_on_hand_built_pairs(tmp_path) -> None:
    """The per-chromosome workers assemble fragments exactly as the serial path does."""
    reads = Reads(tmp_path, chromosomes=("chr1", "chr2"))
    for chrom in ("chr1", "chr2"):
        reads.read(
            "A",
            0,
            110,
            flag=PAIRED | PROPER | MATE_REVERSE | READ1,
            bases={9: "C", 20: "T", 101: "C"},
            mate_start=95,
            tlen=154,
            chrom=chrom,
        )
        reads.read("single", 15, 60, bases={20: "T"}, chrom=chrom)
        reads.read(
            "A",
            95,
            154,
            flag=PAIRED | PROPER | REVERSE | READ2,
            bases={101: "T", 112: "T"},
            mate_start=0,
            tlen=-154,
            chrom=chrom,
        )
    serial = reads.extract()
    parallel = reads.extract(threads=2)

    assert rows_of(serial) == [
        {0: 1, 1: 0, 2: -1, 3: 0},
        {1: 0},
        {4: 1, 5: 0, 6: -1, 7: 0},
        {5: 0},
    ]
    for attribute in ("row", "col", "data"):
        assert np.array_equal(
            getattr(serial.matrix, attribute), getattr(parallel.matrix, attribute)
        )
    assert np.array_equal(serial.tlen, parallel.tlen)


# ---------------------------------------------------------------------------
# Which CpGs a read reports on
# ---------------------------------------------------------------------------


def test_top_strand_read_edges(reads) -> None:
    """A top-strand read reports a CpG only if it covers the C."""
    reads.read("ends_on_c", 0, 10, bases={9: "C"})  # last base is the C
    reads.read("starts_on_g", 21, 60, bases={21: "G"})  # first base is the G
    assert rows_of(reads.extract()) == [{0: 1}]


def test_bottom_strand_read_edges(reads) -> None:
    """A bottom-strand read reports a CpG only if it covers the G."""
    reads.read("ends_on_c", 0, 10, strand="r", bases={9: "C"})  # G not covered
    reads.read("starts_on_g", 21, 60, strand="r", bases={21: "G"})  # C not covered
    assert rows_of(reads.extract()) == [{1: 1}]


@pytest.mark.parametrize(
    "strand, deleted_position, bases",
    [
        pytest.param("f", 20, {9: "C"}, id="top-strand-c-deleted"),
        pytest.param("r", 21, {10: "G"}, id="bottom-strand-g-deleted"),
    ],
)
def test_deleted_informative_base_is_no_data(
    reads, strand, deleted_position, bases
) -> None:
    """A CpG whose informative base is deleted from the read is recorded as -1."""
    # 60 reference bases with one deleted: [start, deleted) M, 1 D, rest M.
    before = deleted_position
    after = 60 - deleted_position - 1
    length = before + after
    segment = reads.read(
        "deletion",
        0,
        length,
        strand=strand,
        cigar=[(0, before), (2, 1), (0, after)],
    )
    # Place the informative bases by query position (reference position for
    # everything left of the deletion).
    sequence = list(segment.query_sequence)
    for position, base in bases.items():
        sequence[position] = base
    segment.query_sequence = "".join(sequence)
    segment.cigartuples = [(0, before), (2, 1), (0, after)]
    assert segment.reference_end == 60

    assert rows_of(reads.extract()) == [{0: 1, 1: -1}]


# ---------------------------------------------------------------------------
# Bismark: calls sit on the informative base
# ---------------------------------------------------------------------------


def test_bismark_pair_of_each_strand(reads) -> None:
    """Bismark writes top-strand calls on the C and bottom-strand calls on the G."""
    # Top-strand pair (XG=CT): flags 99 / 147.
    reads.read(
        "top",
        0,
        60,
        flag=PAIRED | PROPER | MATE_REVERSE | READ1,
        xm={9: "Z", 20: "z"},
        xg="CT",
        mate_start=94,
        tlen=154,
    )
    reads.read(
        "top",
        94,
        154,
        flag=PAIRED | PROPER | REVERSE | READ2,
        xm={101: "z", 112: "Z"},
        xg="CT",
        mate_start=0,
        tlen=-154,
    )
    # Bottom-strand pair (XG=GA): flags 163 / 83, calls on the G positions.
    reads.read(
        "bottom",
        1,
        60,
        flag=PAIRED | PROPER | MATE_REVERSE | READ2,
        xm={10: "z", 21: "Z"},
        xg="GA",
        mate_start=94,
        tlen=153,
    )
    reads.read(
        "bottom",
        94,
        154,
        flag=PAIRED | PROPER | REVERSE | READ1,
        xm={102: "Z", 113: "z"},
        xg="GA",
        mate_start=1,
        tlen=-153,
    )
    result = reads.extract()

    assert rows_of(result) == [{0: 1, 1: 0, 2: 0, 3: 1}, {0: 0, 1: 1, 2: 1, 3: 0}]
    assert result.tlen.tolist() == [154, 153]


def test_bismark_bottom_strand_calls_are_not_read_at_the_c(reads) -> None:
    """Regression: bottom-strand Bismark calls used to decode as -1.

    The call string was read at the C of every CpG. For a bottom-strand
    alignment the C position holds '.', so half of all Bismark calls were
    recorded as no data.
    """
    reads.read(
        "bottom",
        0,
        154,
        flag=REVERSE,
        xm={10: "Z", 21: "z", 102: "Z", 113: "z"},
        xg="GA",
    )
    assert rows_of(reads.extract()) == [{0: 1, 1: 0, 2: 1, 3: 0}]


def test_bismark_non_call_letter_at_cpg_is_no_data(reads) -> None:
    """Anything other than Z / z at the informative base is no data."""
    reads.read("odd", 0, 154, xm={9: "Z", 20: ".", 101: "h", 112: "z"}, xg="CT")
    assert rows_of(reads.extract()) == [{0: 1, 1: -1, 2: -1, 3: 0}]


@pytest.mark.parametrize(
    "bases, xm, expected",
    [
        # Calls sit on C/T: top strand.
        pytest.param({9: "C", 20: "T"}, {9: "Z", 20: "z"}, {0: 1, 1: 0}, id="top"),
        # Calls sit on G/A: bottom strand.
        pytest.param({10: "G", 21: "A"}, {10: "Z", 21: "z"}, {0: 1, 1: 0}, id="bottom"),
    ],
)
def test_bismark_strand_inferred_without_xg(reads, bases, xm, expected) -> None:
    """Without an XG tag the strand comes from the bases the calls sit on."""
    reads.read("no_xg", 0, 60, bases=bases, xm=xm)
    assert rows_of(reads.extract()) == [expected]


def test_bismark_is_bottom_strand_helper(reads) -> None:
    """XG decides when present; otherwise the bases under the calls do."""
    cases: list[tuple[dict, bool]] = [
        (dict(xm={9: "Z"}, xg="CT"), False),
        (dict(xm={10: "Z"}, xg="GA"), True),
        # XG wins over what the bases would suggest.
        (dict(xm={10: "Z"}, bases={10: "G"}, xg="CT"), False),
        (dict(xm={9: "z"}, bases={9: "T"}), False),
        (dict(xm={10: "z"}, bases={10: "A"}), True),
        # Nothing to go on (calls on N, or no calls at all): top strand.
        (dict(xm={9: "Z"}), False),
        (dict(xm={}), False),
    ]
    for kwargs, expected in cases:
        segment = reads.read("helper", 0, 60, **kwargs)
        assert _bismark_is_bottom_strand(segment, segment.get_tag("XM")) is expected


# ---------------------------------------------------------------------------
# Opt-in filters apply to the fragment
# ---------------------------------------------------------------------------


def _non_converted_pair(reads, name, failing_mate, start=0):
    """A Bismark pair in which one mate has three retained non-CpG cytosines."""
    retained = {
        "left": {30 + start: "H", 31 + start: "H", 32 + start: "X"},
        "right": {},
    }
    if failing_mate == "right":
        retained = {"left": {}, "right": {130: "H", 131: "H", 132: "X"}}
    reads.read(
        name,
        start,
        60,
        flag=PAIRED | PROPER | MATE_REVERSE | READ1,
        xm={9: "Z", 20: "Z", **retained["left"]},
        xg="CT",
        mate_start=94,
        tlen=154 - start,
    )
    reads.read(
        name,
        94,
        154,
        flag=PAIRED | PROPER | REVERSE | READ2,
        xm={101: "Z", 112: "Z", **retained["right"]},
        xg="CT",
        mate_start=start,
        tlen=-(154 - start),
    )


@pytest.mark.parametrize("failing_mate", ["left", "right"])
def test_non_converted_mate_removes_the_whole_fragment(reads, failing_mate) -> None:
    """Incomplete conversion is a property of the molecule: both mates go.

    It makes no difference whether the failing mate is seen first (its
    partner must be discarded when it arrives) or second (the partner is
    already waiting and must be withdrawn).
    """
    _non_converted_pair(reads, "bad", failing_mate)
    reads.read("good", 2, 154, xm={9: "z", 20: "z", 101: "z", 112: "z"}, xg="CT")

    kept = rows_of(reads.extract(filter_non_converted=True))
    assert kept == [{0: 0, 1: 0, 2: 0, 3: 0}]

    # With the filter off, the pair is an ordinary fragment.
    assert rows_of(reads.extract()) == [
        {0: 1, 1: 1, 2: 1, 3: 1},
        {0: 0, 1: 0, 2: 0, 3: 0},
    ]


def test_non_converted_filter_threshold_is_per_read(reads) -> None:
    """Two retained cytosines in each mate do not add up to the threshold of 3."""
    reads.read(
        "pair",
        0,
        60,
        flag=PAIRED | PROPER | MATE_REVERSE | READ1,
        xm={9: "Z", 20: "Z", 30: "H", 31: "H"},
        xg="CT",
        mate_start=94,
        tlen=154,
    )
    reads.read(
        "pair",
        94,
        154,
        flag=PAIRED | PROPER | REVERSE | READ2,
        xm={101: "Z", 112: "Z", 130: "H", 131: "H"},
        xg="CT",
        mate_start=0,
        tlen=-154,
    )
    assert rows_of(reads.extract(filter_non_converted=True)) == [
        {0: 1, 1: 1, 2: 1, 3: 1}
    ]
    # Lowering the threshold to 2 flags each mate.
    assert (
        rows_of(reads.extract(filter_non_converted=True, non_converted_threshold=2))
        == []
    )


def test_non_converted_mate_without_cpgs_still_removes_its_partner(reads) -> None:
    """A failing mate takes its partner even if it covers no CpG itself."""
    reads.read(
        "pair",
        30,
        90,
        flag=PAIRED | PROPER | MATE_REVERSE | READ1,
        xm={40: "H", 41: "H", 42: "H"},
        xg="CT",
        mate_start=94,
        tlen=124,
    )
    reads.read(
        "pair",
        94,
        154,
        flag=PAIRED | PROPER | REVERSE | READ2,
        xm={101: "Z", 112: "Z"},
        xg="CT",
        mate_start=30,
        tlen=-124,
    )
    assert rows_of(reads.extract(filter_non_converted=True)) == []
    assert rows_of(reads.extract()) == [{2: 1, 3: 1}]


def test_em_filter_sees_the_merged_fragment(reads) -> None:
    """Over-conversion is judged on the fragment's CpGs, not each mate's.

    Each mate covers two unmethylated CpGs — below the minimum of three on
    its own — but the fragment covers four.
    """
    reads.pair(
        "all_unmethylated",
        left=(0, 60),
        right=(94, 154),
        left_bases={9: "T", 20: "T"},
        right_bases={101: "T", 112: "T"},
    )
    reads.pair(
        "one_methylated",
        left=(1, 60),
        right=(94, 153),
        left_bases={9: "T", 20: "T"},
        right_bases={101: "T", 112: "C"},
    )
    filtered = reads.extract(
        filter_em_overconversion=True, em_overconversion_min_cpgs=3
    )
    assert rows_of(filtered) == [{0: 0, 1: 0, 2: 0, 3: 1}]

    assert len(rows_of(reads.extract())) == 2
    # Five CpGs required: the four-CpG fragment is no longer eligible.
    lenient = reads.extract(filter_em_overconversion=True, em_overconversion_min_cpgs=5)
    assert len(rows_of(lenient)) == 2


def test_em_filter_does_not_count_a_disagreement_as_unmethylated(reads) -> None:
    """A CpG the mates disagree on is no data: it is not an unmethylated call.

    Two unmethylated calls and a disagreement are not the three unmethylated
    calls the filter asks for.
    """
    reads.pair(
        "disagree",
        left=(0, 110),
        right=(95, 110),
        left_bases={9: "T", 20: "T", 101: "T"},
        right_bases={101: "C"},
    )
    result = reads.extract(filter_em_overconversion=True, em_overconversion_min_cpgs=3)
    assert rows_of(result) == [{0: 0, 1: 0, 2: -1}]


def test_em_filter_ignores_cpgs_without_a_call(reads) -> None:
    """A CpG with no usable call neither counts towards the minimum nor shields.

    Otherwise one low-quality base would protect an otherwise fully
    unmethylated fragment from the filter.
    """
    # Three unmethylated calls and one CpG on a low-quality base.
    reads.read(
        "three_calls",
        0,
        154,
        bases={9: "T", 20: "T", 101: "T", 112: "T"},
        qualities={101: 5},
    )
    # Two unmethylated calls and two CpGs without a call: too few calls.
    reads.read(
        "two_calls",
        1,
        154,
        bases={9: "T", 20: "T", 101: "T", 112: "T"},
        qualities={101: 5, 112: 5},
    )
    # Three unmethylated calls and a methylated one: not over-converted.
    reads.read("mixed", 2, 154, bases={9: "T", 20: "T", 101: "T", 112: "C"})

    assert rows_of(reads.extract()) == [
        {0: 0, 1: 0, 2: -1, 3: 0},
        {0: 0, 1: 0, 2: -1, 3: -1},
        {0: 0, 1: 0, 2: 0, 3: 1},
    ]
    filtered = reads.extract(
        filter_em_overconversion=True, em_overconversion_min_cpgs=3
    )
    assert rows_of(filtered) == [
        {0: 0, 1: 0, 2: -1, 3: -1},
        {0: 0, 1: 0, 2: 0, 3: 1},
    ]


# ---------------------------------------------------------------------------
# Fragment length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tlen", [300, -300])
def test_tlen_is_the_absolute_template_length(reads, tlen) -> None:
    """The BAM's TLEN sign depends on which mate is leftmost; the row drops it."""
    reads.read(
        "mate",
        0,
        60,
        flag=PAIRED | PROPER | READ1,
        bases={9: "C"},
        mate_start=None,
        tlen=tlen,
    )
    result = reads.extract()
    assert result.tlen.tolist() == [300]
    assert result.tlen.dtype == np.int32


def test_tlen_taken_from_second_mate_if_first_has_none(reads) -> None:
    """A fragment length of 0 on the first mate is filled in from the second."""
    reads.pair(
        "frag",
        left=(0, 60),
        right=(94, 154),
        left_bases={9: "C"},
        right_bases={112: "C"},
    )
    reads.segments[0].template_length = 0
    assert reads.extract().tlen.tolist() == [154]


# ---------------------------------------------------------------------------
# _merge_mate_calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first, second, expected",
    [
        pytest.param(
            ([1, 2], [1, 0]),
            ([5, 6], [0, 1]),
            ([1, 2, 5, 6], [1, 0, 0, 1]),
            id="second-right",
        ),
        pytest.param(
            ([5, 6], [0, 1]),
            ([1, 2], [1, 0]),
            ([1, 2, 5, 6], [1, 0, 0, 1]),
            id="second-left",
        ),
        pytest.param(
            ([1, 2, 3], [1, 0, 1]),
            ([2, 3, 4], [0, 1, 0]),
            ([1, 2, 3, 4], [1, 0, 1, 0]),
            id="agree",
        ),
        pytest.param(
            ([1, 2], [1, 0]), ([2, 3], [1, 0]), ([1, 2, 3], [1, -1, 0]), id="disagree"
        ),
        pytest.param(
            ([1, 2], [1, -1]),
            ([2, 3], [0, 1]),
            ([1, 2, 3], [1, 0, 1]),
            id="fill-from-second",
        ),
        pytest.param(
            ([1, 2], [1, 0]), ([2, 3], [-1, 1]), ([1, 2, 3], [1, 0, 1]), id="keep-first"
        ),
        pytest.param(
            ([1, 2], [-1, -1]),
            ([1, 2], [-1, -1]),
            ([1, 2], [-1, -1]),
            id="both-no-data",
        ),
        pytest.param(
            ([2], [1]),
            ([1, 2, 3], [0, 1, 0]),
            ([1, 2, 3], [0, 1, 0]),
            id="first-inside-second",
        ),
        pytest.param(([3], [1]), ([3], [0]), ([3], [-1]), id="single-shared"),
    ],
)
def test_merge_mate_calls(first, second, expected) -> None:
    """The merge rule, case by case; the result is always in column order."""
    assert _merge_mate_calls(*first, *second) == expected
    # The inputs are left untouched.
    before = ([*first[0]], [*first[1]], [*second[0]], [*second[1]])
    _merge_mate_calls(*first, *second)
    assert (first[0], first[1], second[0], second[1]) == before


def test_merge_is_symmetric_in_its_outcome() -> None:
    """Swapping the mates gives the same merged calls."""
    a = ([1, 2, 3, 5], [1, 0, -1, 1])
    b = ([2, 3, 4, 5], [1, 0, 1, -1])
    assert _merge_mate_calls(*a, *b) == _merge_mate_calls(*b, *a)
    assert _merge_mate_calls(*a, *b) == ([1, 2, 3, 4, 5], [1, -1, 0, 1, 1])


# ---------------------------------------------------------------------------
# No warnings on ordinary paired data
# ---------------------------------------------------------------------------


def test_paired_extraction_is_warning_free(reads) -> None:
    """Assembling fragments raises no warnings of its own."""
    reads.pair(
        "frag",
        left=(0, 60),
        right=(94, 154),
        left_bases={9: "C", 20: "C"},
        right_bases={101: "T", 112: "T"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert rows_of(reads.extract(filter_em_overconversion=True)) == [
            {0: 1, 1: 1, 2: 0, 3: 0}
        ]


# ---------------------------------------------------------------------------
# Strand-tag check
# ---------------------------------------------------------------------------

STRAND_CHECK_LINE = re.compile(r"Strand-tag check: ([\d,]+) of ([\d,]+) CpGs")


def strand_check_counts(output: str):
    """(contradictions, CpGs checked) from verbose output, or None if absent."""
    match = STRAND_CHECK_LINE.search(output)
    if match is None:
        return None
    return tuple(int(group.replace(",", "")) for group in match.groups())


@pytest.mark.parametrize(
    "strand, bases, expected",
    [
        # Top-strand tag: the G of each CpG must read G. An A there is the
        # bottom strand's conversion product.
        pytest.param(
            "f", {10: "G", 21: "G", 102: "G", 113: "G"}, (0, 4), id="top-clean"
        ),
        pytest.param(
            "f", {10: "A", 21: "G", 102: "A", 113: "G"}, (2, 4), id="top-two-a"
        ),
        # Any other base at the G is an ordinary mismatch, not a contradiction.
        pytest.param(
            "f", {10: "T", 21: "C", 102: "N", 113: "G"}, (0, 4), id="top-other"
        ),
        # Bottom-strand tag: the C of each CpG must read C; a T contradicts.
        pytest.param(
            "r", {9: "C", 20: "C", 101: "C", 112: "C"}, (0, 4), id="bottom-clean"
        ),
        pytest.param(
            "r", {9: "T", 20: "C", 101: "C", 112: "C"}, (1, 4), id="bottom-one-t"
        ),
        pytest.param(
            "r", {9: "T", 20: "T", 101: "T", 112: "T"}, (4, 4), id="bottom-all-t"
        ),
    ],
)
def test_strand_check_counts_conversions_on_the_wrong_base(
    reads, capsys, strand, bases, expected
) -> None:
    """Each CpG's never-converted base is examined; a conversion there counts."""
    reads.read("read", 0, 154, strand=strand, bases=bases)
    reads.extract(verbose=True)
    assert strand_check_counts(capsys.readouterr().out) == expected


def test_strand_check_needs_the_partner_base_to_be_covered(reads, capsys) -> None:
    """A CpG cut by the end of a read has no partner base to check."""
    reads.read("top_ends_on_c", 0, 10, bases={9: "C"})  # the G is past the read
    reads.read("bottom_starts_on_g", 21, 60, strand="r", bases={21: "G"})  # no C
    result = reads.extract(verbose=True)

    assert rows_of(result) == [{0: 1}, {1: 1}]  # both CpGs are still called
    assert strand_check_counts(capsys.readouterr().out) is None  # nothing checked


def test_strand_check_counts_both_mates(reads, capsys) -> None:
    """Every decoded read contributes, including both mates of a pair."""
    reads.pair(
        "frag",
        left=(0, 110),
        right=(95, 154),
        left_bases={9: "C", 10: "A", 20: "C", 21: "G", 101: "C", 102: "G"},
        right_bases={101: "C", 102: "A", 112: "C", 113: "G"},
    )
    reads.extract(verbose=True)
    # Left mate: 3 CpGs checked, 1 contradiction. Right mate: 2 checked, 1.
    assert strand_check_counts(capsys.readouterr().out) == (2, 5)


def test_strand_check_does_not_apply_to_bismark_reads(
    reads, capsys, monkeypatch
) -> None:
    """Bismark resolves the strand itself; its reads are not checked.

    The calls are still discarded, as for any aligner: with an A where the G
    of the CpG should be, the read shows no CpG.
    """
    monkeypatch.setattr(functions, "_STRAND_CHECK_MIN_CPGS", 1)
    reads.read(
        "bismark",
        0,
        154,
        bases={10: "A", 21: "A", 102: "A", 113: "A"},
        xm={9: "Z", 20: "Z", 101: "z", 112: "z"},
        xg="CT",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = reads.extract(verbose=True)
    assert rows_of(result) == [{0: -1, 1: -1, 2: -1, 3: -1}]
    assert strand_check_counts(capsys.readouterr().out) is None


def test_strand_warning_thresholds(reads, monkeypatch) -> None:
    """The warning needs enough CpGs and a rate strictly above the limit."""
    # 2 contradictions in 4 CpGs: 50%.
    reads.read("read", 0, 154, bases={10: "A", 21: "A", 102: "G", 113: "G"})

    # Too few CpGs checked to conclude anything (default minimum is 1,000).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reads.extract()

    monkeypatch.setattr(functions, "_STRAND_CHECK_MIN_CPGS", 4)
    with pytest.warns(RuntimeWarning) as recorded:
        reads.extract()
    assert len(recorded) == 1
    message = str(recorded[0].message)
    assert "2 of 4 CpGs (50.0%)" in message
    assert "strand tag" in message
    assert "non-directional" in message
    assert "reads.bam" in message

    # Exactly at the limit is not over it.
    monkeypatch.setattr(functions, "_STRAND_CONTRADICTION_WARNING_RATE", 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reads.extract()

    # One more CpG than was checked: below the minimum again.
    monkeypatch.setattr(functions, "_STRAND_CONTRADICTION_WARNING_RATE", 0.02)
    monkeypatch.setattr(functions, "_STRAND_CHECK_MIN_CPGS", 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reads.extract()


def test_default_strand_check_limits() -> None:
    """The shipped limits: over 2% of at least 1,000 CpGs."""
    assert functions._STRAND_CONTRADICTION_WARNING_RATE == 0.02
    assert functions._STRAND_CHECK_MIN_CPGS == 1000


# ---------------------------------------------------------------------------
# A call needs an intact CpG
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strand, bases, expected",
    [
        # Top strand: the call is on the C (9, 20); the G (10, 21) must read G.
        pytest.param(
            "f", {9: "T", 10: "G", 20: "C", 21: "G"}, {0: 0, 1: 1}, id="top-intact"
        ),
        pytest.param(
            "f", {9: "T", 10: "A", 20: "C", 21: "G"}, {0: -1, 1: 1}, id="top-g-is-a"
        ),
        pytest.param(
            "f", {9: "T", 10: "T", 20: "C", 21: "C"}, {0: -1, 1: -1}, id="top-g-is-c-t"
        ),
        pytest.param(
            "f", {9: "T", 10: "N", 20: "C", 21: "G"}, {0: -1, 1: 1}, id="top-g-is-n"
        ),
        # Bottom strand: the call is on the G; the C (9, 20) must read C.
        pytest.param(
            "r", {9: "C", 10: "A", 20: "C", 21: "G"}, {0: 0, 1: 1}, id="bottom-intact"
        ),
        pytest.param(
            "r", {9: "T", 10: "A", 20: "C", 21: "G"}, {0: -1, 1: 1}, id="bottom-c-is-t"
        ),
        pytest.param(
            "r",
            {9: "A", 10: "A", 20: "G", 21: "G"},
            {0: -1, 1: -1},
            id="bottom-c-is-a-g",
        ),
    ],
)
def test_call_requires_the_other_base_of_the_cpg(
    reads, strand, bases, expected
) -> None:
    """A call is kept only if the read shows a CpG, not just a C (or G).

    A molecule on which the CpG has mutated to CpA has no CpG to be
    methylated. Its lone C is converted like any other non-CpG cytosine, so
    a top-strand read shows T there — an "unmethylated CpG" unless the A
    that follows is noticed. The same goes, mirrored, for a bottom-strand
    read of a TpG molecule.
    """
    reads.read("read", 0, 60, strand=strand, bases=bases)
    assert rows_of(reads.extract()) == [expected]


@pytest.mark.parametrize(
    "strand, deleted_position, bases",
    [
        # Top strand: the second CpG's G (21) is deleted; its C reads T.
        pytest.param("f", 21, {9: "C", 10: "G", 20: "T"}, id="top-strand-g-deleted"),
        # Bottom strand: the second CpG's C (20) is deleted; its G (query
        # position 20 once the C is gone) reads A.
        pytest.param("r", 20, {9: "C", 10: "G", 20: "A"}, id="bottom-strand-c-deleted"),
    ],
)
def test_deleted_other_base_is_no_data(reads, strand, deleted_position, bases) -> None:
    """A CpG whose other base is deleted from the read is not a CpG there."""
    before = deleted_position
    after = 60 - deleted_position - 1
    segment = reads.read(
        "deletion",
        0,
        before + after,
        strand=strand,
        cigar=[(0, before), (2, 1), (0, after)],
    )
    sequence = ["N"] * (before + after)
    for position, base in bases.items():  # query positions
        sequence[position] = base
    segment.query_sequence = "".join(sequence)
    segment.cigartuples = [(0, before), (2, 1), (0, after)]
    assert segment.reference_end == 60

    assert rows_of(reads.extract()) == [{0: 1, 1: -1}]


@pytest.mark.parametrize("strand", ["f", "r"])
def test_base_inserted_inside_a_cpg_is_no_data(reads, strand) -> None:
    """C and G both present, but with a base between them: no CpG."""
    # 21 reference bases (0-20), one inserted base, then reference 21-59:
    # the second CpG's C is query position 20 and its G query position 22.
    cigar = [(0, 21), (1, 1), (0, 39)]
    segment = reads.read("insertion", 0, 61, strand=strand, cigar=cigar)
    sequence = ["N"] * 61
    for position, base in {9: "C", 10: "G", 20: "C", 21: "A", 22: "G"}.items():
        sequence[position] = base
    segment.query_sequence = "".join(sequence)
    segment.cigartuples = cigar
    assert segment.reference_end == 60

    assert rows_of(reads.extract()) == [{0: 1, 1: -1}]


def test_bismark_call_requires_an_intact_cpg_too(reads) -> None:
    """Bismark takes CpG context from the reference; the read must agree.

    The rule is applied from the read's bases for every aligner, which is
    what keeps Bismark and bwa-meth output identical for the same reads.
    """
    reads.read(
        "top",
        0,
        60,
        bases={9: "T", 10: "A", 20: "C", 21: "G"},
        xm={9: "z", 20: "Z"},
        xg="CT",
    )
    reads.read(
        "bottom",
        1,
        60,
        bases={9: "T", 10: "A", 20: "C", 21: "G"},
        xm={10: "z", 21: "Z"},
        xg="GA",
    )
    assert rows_of(reads.extract()) == [{0: -1, 1: 1}, {0: -1, 1: 1}]


def test_bismark_read_without_bases_is_decoded_from_its_call_string(reads) -> None:
    """A Bismark read stored without SEQ still yields its XM calls."""
    segment = reads.read("no_seq", 0, 60, xm={9: "Z", 20: "z"}, xg="CT")
    xm_tag = segment.get_tag("XM")
    segment.query_sequence = None
    segment.cigartuples = [(0, 60)]
    segment.set_tag("XM", xm_tag)
    assert segment.query_sequence is None

    assert rows_of(reads.extract()) == [{0: 1, 1: 0}]


# ---------------------------------------------------------------------------
# Base quality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quality, expected", [(0, -1), (19, -1), (20, 1), (40, 1)])
def test_call_on_a_low_quality_base_is_no_data(reads, quality, expected) -> None:
    """A call needs a base quality of at least 20 by default."""
    reads.read("read", 0, 60, bases={9: "C", 20: "T"}, qualities={9: quality})
    assert rows_of(reads.extract()) == [{0: expected, 1: 0}]


def test_min_base_quality_is_adjustable(reads) -> None:
    """A higher limit discards more calls; 0 turns the filter off."""
    reads.read("read", 0, 60, bases={9: "C", 20: "T"}, qualities={9: 25, 20: 10})
    assert rows_of(reads.extract()) == [{0: 1, 1: -1}]
    assert rows_of(reads.extract(min_base_quality=30)) == [{0: -1, 1: -1}]
    assert rows_of(reads.extract(min_base_quality=0)) == [{0: 1, 1: 0}]


def test_quality_of_the_other_base_does_not_discard_a_call(reads) -> None:
    """Only the base that carries the call needs the quality."""
    reads.read("read", 0, 60, bases={9: "C", 20: "T"}, qualities={10: 2, 21: 2})
    assert rows_of(reads.extract()) == [{0: 1, 1: 0}]


def test_reads_without_base_qualities_are_not_filtered(reads) -> None:
    """A BAM that stores no qualities cannot be filtered on them."""
    segment = reads.read("read", 0, 60, bases={9: "C", 20: "T"})
    segment.query_qualities = None
    assert rows_of(reads.extract()) == [{0: 1, 1: 0}]


def test_bismark_calls_are_filtered_by_base_quality(reads) -> None:
    """The base-quality rule applies to calls read from the XM string too."""
    reads.read("bismark", 0, 60, xm={9: "Z", 20: "z"}, xg="CT", qualities={20: 3})
    assert rows_of(reads.extract()) == [{0: 1, 1: -1}]
    assert rows_of(reads.extract(min_base_quality=0)) == [{0: 1, 1: 0}]


def test_mate_supplies_the_call_where_the_other_mate_has_none(reads) -> None:
    """A low-quality base in one mate leaves the other mate's call standing.

    Both mates cover the third CpG. The left mate reads C on a low-quality
    base, the right mate T on a good one: with the filter the fragment's
    call is the right mate's; without it the two disagree.
    """
    reads.pair(
        "frag",
        left=(0, 110),
        right=(95, 154),
        left_bases={9: "C", 20: "T", 101: "C"},
        right_bases={101: "T", 112: "C"},
        left_kwargs={"qualities": {101: 5}},
    )
    assert rows_of(reads.extract()) == [{0: 1, 1: 0, 2: 0, 3: 1}]
    assert rows_of(reads.extract(min_base_quality=0)) == [{0: 1, 1: 0, 2: -1, 3: 1}]


def test_low_quality_other_base_is_not_strand_checked(reads, capsys) -> None:
    """A base below the quality limit is no evidence about the strand tag."""
    reads.read(
        "read",
        0,
        154,
        bases={10: "A", 21: "A", 102: "G", 113: "G"},
        qualities={10: 5},
    )
    reads.extract(verbose=True)
    assert strand_check_counts(capsys.readouterr().out) == (1, 3)
    reads.extract(verbose=True, min_base_quality=0)
    assert strand_check_counts(capsys.readouterr().out) == (2, 4)


# ---------------------------------------------------------------------------
# Masking the 3' end of the fragment
# ---------------------------------------------------------------------------

ALL_METHYLATED = {0: 1, 1: 1, 2: 1, 3: 1}


@pytest.mark.parametrize(
    "mask, masked_columns",
    [
        (0, []),
        # The fragment spans [0, 154). The C of the last CpG is at 112: 41
        # bases from the last base of the fragment.
        (41, []),
        (42, [3]),
        (52, [3]),
        (53, [2, 3]),
        (154, [0, 1, 2, 3]),
    ],
)
def test_3prime_mask_of_a_top_strand_fragment(reads, mask, masked_columns) -> None:
    """The top strand's 3' end is the right-hand end of the fragment.

    The distance is measured from the end of the fragment, not of each read:
    the left mate ends at 110, nine bases past the third CpG, yet that CpG is
    only masked once the mask reaches it from the fragment's end at 154. The
    two mates agree on what is masked, so a masked CpG they both cover (the
    third) stays masked.
    """
    reads.pair(
        "frag",
        left=(0, 110),
        right=(95, 154),
        left_bases={9: "C", 20: "C", 101: "C"},
        right_bases={101: "C", 112: "C"},
    )
    expected = {**ALL_METHYLATED, **{column: -1 for column in masked_columns}}
    result = reads.extract(mask_3prime=mask)
    assert rows_of(result) == [expected]
    # Counted per fragment: the third CpG, covered by both mates, counts once.
    assert result.stats["cpg_masked_3prime"] == len(masked_columns)


@pytest.mark.parametrize(
    "mask, masked_columns",
    [
        (0, []),
        # The G of the first CpG is at 10: ten bases from the fragment start.
        (10, []),
        (11, [0]),
        (21, [0]),
        (22, [0, 1]),
        (154, [0, 1, 2, 3]),
    ],
)
def test_3prime_mask_of_a_bottom_strand_fragment(reads, mask, masked_columns) -> None:
    """The bottom strand's 3' end is the left-hand end of the fragment.

    The right mate starts at 5 but belongs to a fragment starting at 0, which
    is where the distance is measured from.
    """
    reads.pair(
        "frag",
        left=(0, 60),
        right=(5, 154),
        strand="r",
        left_bases={10: "G", 21: "G"},
        right_bases={10: "G", 21: "G", 102: "G", 113: "G"},
    )
    expected = {**ALL_METHYLATED, **{column: -1 for column in masked_columns}}
    assert rows_of(reads.extract(mask_3prime=mask)) == [expected]


def test_3prime_mask_of_single_end_reads_uses_the_read_end(reads) -> None:
    """Without a mate, the read's own end is the only fragment end known."""
    # Top strand, ends at 110: the C at 101 is 8 bases from the last base.
    reads.read("top", 0, 110, bases={9: "C", 20: "C", 101: "C"})
    # Bottom strand, starts at 5: the G at 10 is 5 bases from the start.
    reads.read(
        "bottom",
        5,
        154,
        flag=REVERSE,
        strand="r",
        bases={10: "G", 21: "G", 102: "G", 113: "G"},
    )
    top = {0: 1, 1: 1, 2: 1}
    assert rows_of(reads.extract(mask_3prime=5)) == [top, ALL_METHYLATED]
    assert rows_of(reads.extract(mask_3prime=6)) == [
        top,
        {**ALL_METHYLATED, 0: -1},
    ]
    assert rows_of(reads.extract(mask_3prime=9)) == [
        {**top, 2: -1},
        {**ALL_METHYLATED, 0: -1},
    ]


def test_3prime_mask_applies_to_bismark_reads(reads) -> None:
    """Masking works from positions, so it is the same for every aligner."""
    reads.read("top", 0, 154, xm={9: "Z", 20: "Z", 101: "Z", 112: "Z"}, xg="CT")
    reads.read("bottom", 0, 154, xm={10: "Z", 21: "Z", 102: "Z", 113: "Z"}, xg="GA")
    assert rows_of(reads.extract(mask_3prime=42)) == [
        {**ALL_METHYLATED, 3: -1},
        {**ALL_METHYLATED, 0: -1, 1: -1},
    ]


def test_3prime_mask_does_not_drop_the_row(reads) -> None:
    """A fragment whose calls are all masked is still a row: it covers the CpGs."""
    reads.read("read", 0, 60, bases={9: "C", 20: "T"})
    result = reads.extract(mask_3prime=60)
    assert rows_of(result) == [{0: -1, 1: -1}]


# ---------------------------------------------------------------------------
# Proper pairs
# ---------------------------------------------------------------------------


def test_improper_pairs_are_dropped_unless_kept(reads) -> None:
    """Paired reads not flagged as a proper pair are skipped by default.

    Single-end reads have no pair to be proper, and are always kept.
    """
    reads.pair(
        "proper",
        left=(0, 60),
        right=(94, 154),
        left_bases={9: "C", 20: "C"},
        right_bases={101: "C", 112: "C"},
    )
    reads.pair(
        "improper",
        left=(1, 60),
        right=(94, 153),
        left_bases={9: "T", 20: "T"},
        right_bases={101: "T", 112: "T"},
        left_kwargs={"flag": PAIRED | MATE_REVERSE | READ1},
        right_kwargs={"flag": PAIRED | REVERSE | READ2},
    )
    reads.read("single", 2, 60, bases={9: "C", 20: "T"})

    default = reads.extract()
    assert rows_of(default) == [ALL_METHYLATED, {0: 1, 1: 0}]
    assert default.stats["reads_dropped_improper_pair"] == 2

    kept = reads.extract(keep_improper_pairs=True)
    assert rows_of(kept) == [ALL_METHYLATED, {0: 0, 1: 0, 2: 0, 3: 0}, {0: 1, 1: 0}]
    assert kept.stats["reads_dropped_improper_pair"] == 0


# ---------------------------------------------------------------------------
# Mates are matched by position as well as by name
# ---------------------------------------------------------------------------


def test_pairs_sharing_a_name_are_matched_by_position(reads) -> None:
    """Two pairs with one name: each read 2 joins the read 1 that points to it.

    Regression: mates were matched on the name alone, so the second pair's
    read 2 was merged into the first pair's row.
    """
    reads.pair(
        "same",
        left=(0, 60),
        right=(94, 154),
        left_bases={9: "C", 20: "C"},
        right_bases={101: "C", 112: "C"},
    )
    reads.pair(
        "same",
        left=(1, 60),
        right=(93, 153),
        left_bases={9: "T", 20: "T"},
        right_bases={101: "T", 112: "T"},
    )
    result = reads.extract()
    assert rows_of(result) == [ALL_METHYLATED, {0: 0, 1: 0, 2: 0, 3: 0}]
    assert result.tlen.tolist() == [154, 152]
    assert rows_of(reads.extract(threads=2)) == rows_of(result)


def test_mate_must_be_where_the_first_mate_says(reads) -> None:
    """A read 2 the read 1 does not point to starts a fragment of its own."""
    reads.read(
        "frag",
        0,
        60,
        flag=PAIRED | PROPER | MATE_REVERSE | READ1,
        bases={9: "C", 20: "C"},
        mate_start=94,
        tlen=154,
    )
    reads.read(
        "frag",
        90,
        154,
        flag=PAIRED | PROPER | REVERSE | READ2,
        bases={101: "T", 112: "T"},
        mate_start=0,
        tlen=-154,
    )
    assert rows_of(reads.extract()) == [{0: 1, 1: 1}, {2: 0, 3: 0}]


# ---------------------------------------------------------------------------
# Extraction statistics and the reference check
# ---------------------------------------------------------------------------


def test_extraction_stats_count_what_was_discarded(reads) -> None:
    """Every discarded read and call is counted, by reason."""
    reads.pair(
        "pair",
        left=(0, 60),
        right=(94, 154),
        # Second CpG: T where its G should be (a mismatch, not a conversion).
        left_bases={9: "C", 20: "T", 21: "T"},
        # Third CpG on a low-quality base; the fourth is in the masked end.
        right_bases={101: "C", 112: "C"},
        right_kwargs={"qualities": {101: 3}},
    )
    # First CpG: A where its G should be (the other strand's conversion).
    reads.read("single", 0, 154, bases={9: "C", 10: "A", 20: "C", 101: "C", 112: "C"})
    reads.pair(
        "improper",
        left=(0, 60),
        right=(94, 154),
        left_kwargs={"flag": PAIRED | MATE_REVERSE | READ1},
        right_kwargs={"flag": PAIRED | REVERSE | READ2},
    )

    result = reads.extract(mask_3prime=42)

    assert rows_of(result) == [
        {0: 1, 1: -1, 2: -1, 3: -1},
        {0: -1, 1: 1, 2: 1, 3: -1},
    ]
    assert result.stats == {
        "reads_used": 3,
        "reads_dropped_improper_pair": 2,
        "reads_without_md": 0,
        "fragments_from_two_mates": 1,
        "cpg_observations": 8,
        "cpg_not_intact": 2,
        "cpg_reference_mismatch": 1,
        "cpg_low_base_quality": 1,
        "cpg_masked_3prime": 2,
        "strand_checks": 8,
        "strand_contradictions": 1,
    }
    assert reads.extract(mask_3prime=42, threads=2).stats == result.stats


def test_reference_mismatch_warning(reads, monkeypatch) -> None:
    """Reads that show no CpG where the reference has one: the wrong reference."""
    # Three of four CpGs do not match: A and G where a C or T is due, and a
    # T where the third CpG's G should be.
    reads.read("read", 0, 154, bases={9: "A", 20: "G", 101: "C", 102: "T", 112: "C"})

    # Too few CpGs to conclude anything (the default minimum is 1,000).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = reads.extract()
    assert rows_of(result) == [{0: -1, 1: -1, 2: -1, 3: 1}]
    assert result.stats["cpg_reference_mismatch"] == 3

    monkeypatch.setattr(functions, "_REFERENCE_MISMATCH_MIN_CPGS", 4)
    with pytest.warns(RuntimeWarning) as recorded:
        reads.extract()
    assert len(recorded) == 1
    message = str(recorded[0].message)
    assert "3 of 4 CpGs (75.0%)" in message
    assert "not aligned to the reference FASTA" in message
    assert "reads.bam" in message

    # Exactly at the limit is not over it.
    monkeypatch.setattr(functions, "_REFERENCE_MISMATCH_WARNING_RATE", 0.75)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reads.extract()


def test_wrong_strand_tags_do_not_look_like_a_wrong_reference(
    reads, monkeypatch
) -> None:
    """The other strand's conversion product is the strand check's business."""
    monkeypatch.setattr(functions, "_REFERENCE_MISMATCH_MIN_CPGS", 4)
    monkeypatch.setattr(functions, "_STRAND_CHECK_MIN_CPGS", 4)
    reads.read(
        "read",
        0,
        154,
        bases={
            9: "C",
            10: "A",
            20: "C",
            21: "A",
            101: "C",
            102: "A",
            112: "C",
            113: "A",
        },
    )
    with pytest.warns(RuntimeWarning) as recorded:
        result = reads.extract()

    assert len(recorded) == 1
    assert "strand tag" in str(recorded[0].message)
    assert rows_of(result) == [{0: -1, 1: -1, 2: -1, 3: -1}]
    assert result.stats["cpg_not_intact"] == 4
    assert result.stats["cpg_reference_mismatch"] == 0


def test_default_reference_check_limits() -> None:
    """The shipped limits: over 10% of at least 1,000 CpGs."""
    assert functions._REFERENCE_MISMATCH_WARNING_RATE == 0.10
    assert functions._REFERENCE_MISMATCH_MIN_CPGS == 1000


@pytest.mark.parametrize("option", ["min_base_quality", "mask_3prime"])
def test_negative_option_values_are_rejected(reads, option) -> None:
    """Neither a quality limit nor a mask length can be negative."""
    reads.read("read", 0, 60, bases={9: "C"})
    with pytest.raises(ValueError, match=f"{option} must be >= 0"):
        reads.extract(**{option: -1})


# ---------------------------------------------------------------------------
# 3' end mask: where the fragment ends
# ---------------------------------------------------------------------------

TLEN_CONVENTIONS = pytest.mark.parametrize(
    "outer_span",
    [False, True],
    ids=["tlen-between-5prime-ends", "tlen-outer-span"],
)


@TLEN_CONVENTIONS
def test_3prime_mask_of_top_strand_mates_extending_past_each_other(
    reads, outer_span
) -> None:
    """The fragment ends where its reverse-aligned mate does, whatever TLEN says.

    Forward mate [40, 140), reverse mate [10, 110). bwa reports the distance
    between the mates' starting ends (110 - 40 = 70) as the template length,
    Bismark the outer span of the two alignments (140 - 10 = 130). The top
    strand's 3' end is at 110 either way: the C at 112, which only the
    forward mate reaches, lies beyond it, and the C at 101 eight bases
    inside it.
    """
    tlen = 130 if outer_span else 70
    reads.read(
        "frag",
        40,
        140,
        flag=PAIRED | PROPER | MATE_REVERSE | READ1,
        bases={101: "C", 112: "C"},
        mate_start=10,
        tlen=tlen,
    )
    reads.read(
        "frag",
        10,
        110,
        flag=PAIRED | PROPER | REVERSE | READ2,
        bases={20: "C", 101: "C"},
        mate_start=40,
        tlen=-tlen,
    )
    assert rows_of(reads.extract()) == [{1: 1, 2: 1, 3: 1}]
    assert rows_of(reads.extract(mask_3prime=1)) == [{1: 1, 2: 1, 3: -1}]
    assert rows_of(reads.extract(mask_3prime=8)) == [{1: 1, 2: 1, 3: -1}]
    assert rows_of(reads.extract(mask_3prime=9)) == [{1: 1, 2: -1, 3: -1}]


@TLEN_CONVENTIONS
def test_3prime_mask_of_bottom_strand_mates_extending_past_each_other(
    reads, outer_span
) -> None:
    """The bottom strand's 3' end is where the forward-aligned mate starts.

    Reverse mate (read 1) [10, 110), forward mate (read 2) [40, 140): the
    fragment starts at 40. The Gs at 10 and 21, which only the reverse mate
    reaches, lie before it.
    """
    tlen = 130 if outer_span else 70
    reads.read(
        "frag",
        10,
        110,
        flag=PAIRED | PROPER | REVERSE | READ1,
        strand="r",
        bases={10: "G", 21: "G", 102: "G"},
        mate_start=40,
        tlen=-tlen,
    )
    reads.read(
        "frag",
        40,
        140,
        flag=PAIRED | PROPER | MATE_REVERSE | READ2,
        strand="r",
        bases={102: "G", 113: "G"},
        mate_start=10,
        tlen=tlen,
    )
    assert rows_of(reads.extract()) == [ALL_METHYLATED]
    assert rows_of(reads.extract(mask_3prime=1)) == [{0: -1, 1: -1, 2: 1, 3: 1}]
    assert rows_of(reads.extract(mask_3prime=62)) == [{0: -1, 1: -1, 2: 1, 3: 1}]
    assert rows_of(reads.extract(mask_3prime=63)) == [{0: -1, 1: -1, 2: -1, 3: 1}]


@TLEN_CONVENTIONS
def test_3prime_mask_uses_an_earlier_mate_that_covers_no_cpg(reads, outer_span) -> None:
    """A mate without CpGs opens no fragment, yet still marks the 3' end.

    The reverse mate [30, 100) lies between the CpGs and comes first in the
    BAM. Both CpGs the forward mate [40, 140) covers are beyond the
    fragment's end at 100.
    """
    tlen = 110 if outer_span else 60
    reads.read(
        "frag",
        40,
        140,
        flag=PAIRED | PROPER | MATE_REVERSE | READ1,
        bases={101: "C", 112: "C"},
        mate_start=30,
        tlen=tlen,
    )
    reads.read(
        "frag",
        30,
        100,
        flag=PAIRED | PROPER | REVERSE | READ2,
        mate_start=40,
        tlen=-tlen,
    )
    assert rows_of(reads.extract()) == [{2: 1, 3: 1}]
    assert rows_of(reads.extract(mask_3prime=1)) == [{2: -1, 3: -1}]
    assert rows_of(reads.extract(mask_3prime=1, threads=2)) == [{2: -1, 3: -1}]


def test_3prime_mask_uses_a_later_mate_that_covers_no_cpg(reads) -> None:
    """The second mate's coordinates replace the estimate from the first.

    The forward mate's TLEN claims a fragment of 154 bases; its reverse
    mate, which covers no CpG, ends at 100. The mate is believed.
    """
    reads.read(
        "frag",
        0,
        60,
        flag=PAIRED | PROPER | MATE_REVERSE | READ1,
        bases={9: "C", 20: "C"},
        mate_start=70,
        tlen=154,
    )
    reads.read(
        "frag",
        70,
        100,
        flag=PAIRED | PROPER | REVERSE | READ2,
        mate_start=0,
        tlen=-154,
    )
    # 79 bases from the end at 100 to the C at 20; 133 from an end at 154.
    assert rows_of(reads.extract(mask_3prime=79)) == [{0: 1, 1: 1}]
    assert rows_of(reads.extract(mask_3prime=80)) == [{0: 1, 1: -1}]


def test_3prime_mask_falls_back_on_tlen_when_the_mate_is_filtered(reads) -> None:
    """With its mate filtered out, a read locates the fragment end by TLEN.

    Not by its own end at 60, which would put the C at 20 only 39 bases
    from the end.
    """
    reads.pair(
        "frag",
        left=(0, 60),
        right=(94, 154),
        left_bases={9: "C", 20: "C"},
        right_bases={101: "C", 112: "C"},
        right_kwargs={"mapq": 0},
    )
    assert rows_of(reads.extract(mask_3prime=40)) == [{0: 1, 1: 1}]
    assert rows_of(reads.extract(mask_3prime=133)) == [{0: 1, 1: 1}]
    assert rows_of(reads.extract(mask_3prime=134)) == [{0: 1, 1: -1}]


def test_3prime_mask_forgets_ends_whose_mate_never_came(reads, monkeypatch) -> None:
    """Ends noted for mates that never arrive are discarded, not accumulated."""
    monkeypatch.setattr(functions, "_MAX_ENDS_SEEN", 1)
    # A reverse mate without CpGs whose forward mate (due at 40) is absent...
    reads.read(
        "orphan",
        30,
        100,
        flag=PAIRED | PROPER | REVERSE | READ2,
        mate_start=40,
        tlen=-60,
    )
    # ...then, past that position, a complete pair of the same shape.
    reads.read(
        "frag",
        50,
        100,
        flag=PAIRED | PROPER | REVERSE | READ2,
        mate_start=60,
        tlen=-40,
    )
    reads.read(
        "frag",
        60,
        154,
        flag=PAIRED | PROPER | MATE_REVERSE | READ1,
        bases={101: "C", 112: "C"},
        mate_start=50,
        tlen=40,
    )
    assert rows_of(reads.extract(mask_3prime=1)) == [{2: -1, 3: -1}]


# ---------------------------------------------------------------------------
# Filters that trust the BAM's own annotations say so when they bite hard
# ---------------------------------------------------------------------------


def test_warning_when_most_reads_are_improper_pairs(reads, monkeypatch) -> None:
    """An aligner that never sets the proper-pair flag must not go unnoticed.

    Every paired read would be dropped, and the output would simply be
    (nearly) empty.
    """
    for index in range(3):
        reads.pair(
            f"unflagged{index}",
            left=(index, 60),
            right=(94, 154 - index),
            left_bases={9: "C", 20: "C"},
            right_bases={101: "C", 112: "C"},
            left_kwargs={"flag": PAIRED | MATE_REVERSE | READ1},
            right_kwargs={"flag": PAIRED | REVERSE | READ2},
        )
    reads.read("single", 0, 60, bases={9: "C", 20: "T"})

    # Six of seven reads dropped, but far fewer than the 1,000 needed to say
    # anything about a real BAM.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = reads.extract()
    assert rows_of(result) == [{0: 1, 1: 0}]
    assert result.stats["reads_dropped_improper_pair"] == 6
    assert result.stats["reads_used"] == 1

    monkeypatch.setattr(functions, "_FILTER_MAJORITY_MIN_COUNT", 6)
    with pytest.warns(RuntimeWarning) as recorded:
        reads.extract()
    assert len(recorded) == 1
    message = str(recorded[0].message)
    assert "6 of 7 otherwise usable reads (85.7%)" in message
    assert "not flagged as a proper pair" in message
    assert "--keep-improper-pairs" in message

    # Kept, as the warning suggests: nothing dropped, nothing to warn about.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kept = reads.extract(keep_improper_pairs=True)
    assert len(rows_of(kept)) == 4
    assert kept.stats["fragments_from_two_mates"] == 3


def test_no_improper_pair_warning_at_half(reads, monkeypatch) -> None:
    """Exactly half is not "more than half"."""
    monkeypatch.setattr(functions, "_FILTER_MAJORITY_MIN_COUNT", 2)
    reads.pair(
        "improper",
        left=(0, 60),
        right=(94, 154),
        left_kwargs={"flag": PAIRED | MATE_REVERSE | READ1},
        right_kwargs={"flag": PAIRED | REVERSE | READ2},
    )
    reads.pair(
        "proper",
        left=(1, 60),
        right=(94, 153),
        left_bases={9: "C", 20: "C"},
        right_bases={101: "C", 112: "C"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = reads.extract()
    assert result.stats["reads_dropped_improper_pair"] == 2
    assert result.stats["reads_used"] == 2


def test_warning_when_most_calls_are_low_quality(reads, monkeypatch) -> None:
    """Placeholder base qualities must not silently empty the output."""
    placeholder = {position: 2 for position in range(154)}
    reads.read(
        "read",
        0,
        154,
        bases={9: "C", 20: "T", 101: "C", 112: "T"},
        qualities=placeholder,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = reads.extract()
    assert rows_of(result) == [{0: -1, 1: -1, 2: -1, 3: -1}]
    assert result.stats["cpg_low_base_quality"] == 4

    monkeypatch.setattr(functions, "_FILTER_MAJORITY_MIN_COUNT", 4)
    with pytest.warns(RuntimeWarning) as recorded:
        reads.extract()
    assert len(recorded) == 1
    message = str(recorded[0].message)
    assert "4 of 4 CpG calls (100.0%)" in message
    assert "minimum base quality of 20" in message
    assert "--min-base-quality 0" in message

    # With the filter off, as the warning suggests, the calls are back.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unfiltered = reads.extract(min_base_quality=0)
    assert rows_of(unfiltered) == [{0: 1, 1: 0, 2: 1, 3: 0}]


def test_default_majority_warning_limits() -> None:
    """The shipped limits: over half, of at least 1,000."""
    assert functions._FILTER_MAJORITY_WARNING_RATE == 0.5
    assert functions._FILTER_MAJORITY_MIN_COUNT == 1000
