"""Fragment assembly: how reads become rows.

Each row of the output is one sequenced DNA fragment. These tests build small
BAMs by hand to pin down each rule: which mates are combined, how their calls
are merged where they overlap, what happens when a mate is missing or
filtered, the order rows come out in, and how the opt-in filters apply to a
fragment rather than a read.

``tests/test_simulated.py`` checks the same behaviour in bulk against
simulated truth; here every case is spelled out.
"""

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
    ):
        """Add one read covering reference [start, end).

        ``bases`` maps 0-based reference positions to read bases (everything
        else is N). ``xm`` likewise maps reference positions to Bismark call
        letters, turning the read into a Bismark-style one; otherwise it
        carries a ``YD`` tag with the given ``strand``. Positions are
        reference positions, so with a custom ``cigar`` they refer to an
        ungapped layout of the query starting at ``start``.
        """
        length = end - start
        sequence = ["N"] * length
        for position, base in (bases or {}).items():
            sequence[position - start] = base
        segment = pysam.AlignedSegment(self.header)
        segment.query_name = name
        segment.query_sequence = "".join(sequence)
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
    """A mate mapped to another chromosome is not combined."""
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
    result = reads.extract()

    # chr2's CpGs are columns 4-7.
    assert rows_of(result) == [{0: 1, 1: 1}, {6: 0, 7: 0}]
    for threads in (2,):
        assert rows_of(reads.extract(threads=threads)) == rows_of(result)


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
    """A CpG the mates disagree on is no data, which keeps the fragment."""
    reads.pair(
        "disagree",
        left=(0, 110),
        right=(95, 154),
        left_bases={9: "T", 20: "T", 101: "T"},
        right_bases={101: "C", 112: "T"},
    )
    result = reads.extract(filter_em_overconversion=True, em_overconversion_min_cpgs=3)
    assert rows_of(result) == [{0: 0, 1: 0, 2: -1, 3: 0}]


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
