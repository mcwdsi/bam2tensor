"""End-to-end extraction tests against simulated reads with known truth.

``tests/simulate.py`` generates paired-end bisulfite reads from the bundled
FASTA with a known methylation state at every CpG. These tests check the
extractor's output value-for-value against that truth: every row, column,
methylation call and fragment length, for several hundred fragments spread
over three chromosomes and both strands.

The expected output is built by ``_expected`` below, which states the
extractor's contract in a few lines: one row per fragment, the calls of its
mates merged, in the order fragments first appear.
"""

import bisect
import collections
import re
import warnings

import numpy as np
import pysam
import pytest
import scipy.sparse
from click.testing import CliRunner

from bam2tensor import __main__, embedding, functions
from bam2tensor.inspect import main as inspect_main
from bam2tensor.metadata import (
    OUTPUT_FORMAT_VERSION,
    read_npz_metadata,
    read_npz_tlen,
)
from tests.paths import TEST_FASTA
from tests.simulate import SimulatedDataset, SimulatedRead, simulate_bam

CHROMOSOMES = ["chr1", "chr2", "chr3"]
SEED = 20261105


@pytest.fixture(scope="module")
def sim_embedding():
    """Embedding of the three simulated chromosomes."""
    return embedding.GenomeMethylationEmbedding(
        "simulated",
        expected_chromosomes=CHROMOSOMES,
        fasta_source=TEST_FASTA,
        skip_cache=True,
    )


@pytest.fixture(scope="module")
def biscuit_dataset(tmp_path_factory) -> SimulatedDataset:
    """Biscuit/bwameth-style reads (YD strand tag) from both strands."""
    return simulate_bam(
        str(tmp_path_factory.mktemp("biscuit")),
        TEST_FASTA,
        aligner="biscuit",
        chromosomes=CHROMOSOMES,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def bismark_dataset(tmp_path_factory) -> SimulatedDataset:
    """Bismark-style reads (XM call string, XG strand tag) from both strands.

    Same seed as ``biscuit_dataset``: the very same molecules and reads,
    written in the other tag dialect.
    """
    return simulate_bam(
        str(tmp_path_factory.mktemp("bismark")),
        TEST_FASTA,
        aligner="bismark",
        chromosomes=CHROMOSOMES,
        seed=SEED,
    )


# ---------------------------------------------------------------------------
# Expected output, derived from the simulation truth
# ---------------------------------------------------------------------------


def _passes_read_filters(read: SimulatedRead, quality_limit: int) -> bool:
    """Whether a read survives the MAPQ and duplicate filters."""
    return read.mapq >= quality_limit and not read.duplicate


def _merge_call(seen, state):
    """Combine two mates' calls for one CpG (``seen`` is None if first seen).

    Agreement keeps the call, a lone call beats "no data", and a
    methylated/unmethylated disagreement is "no data".
    """
    if seen is None or seen == -1:
        return state
    if state != -1 and state != seen:
        return -1
    return seen


def _expected_fragments(
    dataset: SimulatedDataset,
    emb,
    chrom: str,
    quality_limit: int = 20,
    non_converted_threshold=None,
    em_overconversion_min_cpgs=None,
):
    """Fragments the extractor should report on one chromosome, in row order.

    Returns a list of ``(name, {column: state}, fragment_length)``.

    A fragment is opened by the first of its reads (in BAM order) that passes
    the read filters and covers a CpG, and collects the calls of both mates.
    It is dropped if either mate fails the non-converted filter, or if its
    merged calls trip the EM over-conversion filter.
    """
    calls: dict[str, dict[int, int]] = {}
    lengths: dict[str, int] = {}
    non_converted: set[str] = set()
    for read in dataset.reads:
        if read.chrom != chrom or not _passes_read_filters(read, quality_limit):
            continue
        if (
            non_converted_threshold is not None
            and read.retained_non_cpg >= non_converted_threshold
        ):
            non_converted.add(read.name)
            continue
        if not read.cpg_calls:
            continue
        fragment = calls.setdefault(read.name, {})
        lengths.setdefault(read.name, abs(read.tlen))
        for position, state in read.cpg_calls:
            column = emb.genomic_position_to_embedding(chrom, position)
            fragment[column] = _merge_call(fragment.get(column), state)

    fragments = []
    for name, fragment in calls.items():
        if name in non_converted:
            continue
        states = list(fragment.values())
        if (
            em_overconversion_min_cpgs is not None
            and len(states) >= em_overconversion_min_cpgs
            and all(state == 0 for state in states)
        ):
            continue
        fragments.append((name, dict(sorted(fragment.items())), lengths[name]))
    return fragments


def _expected(dataset: SimulatedDataset, emb, **kwargs):
    """Build the COO triplets and fragment lengths the extractor should produce.

    Rows follow the order chromosomes appear in the embedding and, within a
    chromosome, the order fragments first appear in the BAM.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    tlens: list[int] = []
    for chrom in emb.cpg_sites_dict:
        for _, fragment, length in _expected_fragments(dataset, emb, chrom, **kwargs):
            for column, state in fragment.items():
                rows.append(len(tlens))
                cols.append(column)
                data.append(state)
            tlens.append(length)
    return rows, cols, data, tlens


def _assert_matches_truth(result, expected, emb) -> None:
    """Assert an ExtractionResult equals the expected triplets exactly."""
    rows, cols, data, tlens = expected
    assert result.matrix.shape == (len(tlens), emb.total_cpg_sites)
    assert result.matrix.row.tolist() == rows
    assert result.matrix.col.tolist() == cols
    assert result.matrix.data.tolist() == data
    assert result.tlen.tolist() == tlens
    assert result.tlen.dtype == np.int32


# ---------------------------------------------------------------------------
# The simulated data itself
# ---------------------------------------------------------------------------


def test_simulated_dataset_covers_every_case(biscuit_dataset, sim_embedding) -> None:
    """Guard the fixture: it must contain each case the tests rely on."""
    reads = biscuit_dataset.reads
    assert len(reads) == 2 * 150 * len(CHROMOSOMES)
    assert {read.chrom for read in reads} == set(CHROMOSOMES)

    states = {state for read in reads for _, state in read.cpg_calls}
    assert states == {1, 0, -1}

    for attribute in ("bottom_strand", "is_reverse", "is_read1"):
        values = {getattr(read, attribute) for read in reads}
        assert values == {True, False}, attribute
    # Every strand / mate / orientation combination of a directional library.
    assert {(r.bottom_strand, r.is_read1, r.is_reverse) for r in reads} == {
        (False, True, False),  # top strand:    read 1 forward
        (False, False, True),  # top strand:    read 2 reverse
        (True, True, True),  # bottom strand: read 1 reverse
        (True, False, False),  # bottom strand: read 2 forward
    }
    assert any(read.duplicate for read in reads)
    assert any(read.mapq < 20 for read in reads)
    assert any(read.non_converted for read in reads)
    assert any(not read.cpg_calls for read in reads)

    # How the two mates of each usable pair relate at the CpGs they share.
    by_fragment = collections.defaultdict(list)
    for read in reads:
        if _passes_read_filters(read, 20):
            by_fragment[read.name].append(read)
    seen: collections.Counter = collections.Counter()
    for mates in by_fragment.values():
        contributing = [mate for mate in mates if mate.cpg_calls]
        if len(mates) == 1:
            seen["one mate filtered out"] += 1
        if len(contributing) == 1 and len(mates) == 2:
            seen["one mate covers no CpG"] += 1
        if len(contributing) != 2:
            continue
        first, second = (dict(mate.cpg_calls) for mate in contributing)
        shared = first.keys() & second.keys()
        seen["mates overlap at a CpG" if shared else "mates share no CpG"] += 1
        for position in shared:
            a, b = first[position], second[position]
            if a == b and a != -1:
                seen["agree"] += 1
            elif -1 in (a, b) and a != b:
                seen["one mate has no data"] += 1
            elif a != b:
                seen["disagree"] += 1
    for case in (
        "one mate filtered out",
        "one mate covers no CpG",
        "mates overlap at a CpG",
        "mates share no CpG",
        "agree",
        "one mate has no data",
        "disagree",
    ):
        assert seen[case] >= 5, f"{case}: {seen[case]}"

    # Every simulated CpG position is a real column of the embedding.
    for read in reads:
        for position, _ in read.cpg_calls:
            sim_embedding.genomic_position_to_embedding(read.chrom, position)

    # A meaningful number of rows survive the default filters, including
    # fragments the EM over-conversion filter would remove.
    fragments = [
        fragment
        for chrom in CHROMOSOMES
        for fragment in _expected_fragments(biscuit_dataset, sim_embedding, chrom)
    ]
    assert len(fragments) > 250
    assert any(
        len(calls) >= 3 and set(calls.values()) == {0} for _, calls, _ in fragments
    )


def test_simulation_is_deterministic(tmp_path) -> None:
    """The same seed gives the same reads; a different seed does not."""

    def run(name: str, seed: int) -> SimulatedDataset:
        (tmp_path / name).mkdir()
        return simulate_bam(
            str(tmp_path / name),
            TEST_FASTA,
            aligner="biscuit",
            chromosomes=["chr1"],
            fragments_per_chromosome=20,
            seed=seed,
        )

    first, again, other = run("a", 1), run("b", 1), run("c", 2)

    assert first.reads == again.reads
    assert first.reads != other.reads


# ---------------------------------------------------------------------------
# Biscuit / bwameth dialect
# ---------------------------------------------------------------------------


def test_biscuit_extraction_matches_truth(biscuit_dataset, sim_embedding) -> None:
    """Every row, column, call and TLEN matches what was simulated."""
    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
    )
    _assert_matches_truth(
        result, _expected(biscuit_dataset, sim_embedding), sim_embedding
    )


def test_biscuit_quality_limit_is_applied(biscuit_dataset, sim_embedding) -> None:
    """quality_limit=0 admits exactly the low-MAPQ reads that 20 excludes."""
    strict = _expected(biscuit_dataset, sim_embedding, quality_limit=20)
    lenient = _expected(biscuit_dataset, sim_embedding, quality_limit=0)
    assert len(lenient[3]) > len(strict[3])

    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
        quality_limit=0,
    )
    _assert_matches_truth(result, lenient, sim_embedding)


def test_biscuit_debug_mode_gives_identical_output(
    biscuit_dataset, sim_embedding, capsys
) -> None:
    """debug=True adds checks and logging but must not change the result."""
    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
        debug=True,
    )
    capsys.readouterr()  # discard the (large) debug log
    _assert_matches_truth(
        result, _expected(biscuit_dataset, sim_embedding), sim_embedding
    )


FILTER_CASES = [
    pytest.param(3, None, id="non-converted"),
    pytest.param(None, 3, id="em-overconversion"),
    pytest.param(None, 1, id="em-overconversion-min1"),
    pytest.param(3, 3, id="both"),
]


def _filter_kwargs(non_converted_threshold, em_min_cpgs) -> dict:
    """Extraction keyword arguments for a FILTER_CASES entry."""
    kwargs: dict = {}
    if non_converted_threshold is not None:
        kwargs.update(
            filter_non_converted=True, non_converted_threshold=non_converted_threshold
        )
    if em_min_cpgs is not None:
        kwargs.update(
            filter_em_overconversion=True, em_overconversion_min_cpgs=em_min_cpgs
        )
    return kwargs


@pytest.mark.parametrize("non_converted_threshold, em_min_cpgs", FILTER_CASES)
def test_biscuit_filters_match_truth(
    biscuit_dataset, sim_embedding, non_converted_threshold, em_min_cpgs
) -> None:
    """The opt-in filters drop exactly the fragments the simulation says they should.

    The non-converted filter is evaluated per read and removes the whole
    fragment; the EM over-conversion filter looks at the merged fragment.
    """
    # The simulator keeps every read well clear of the threshold, so each
    # read's status is unambiguous.
    assert all(
        read.retained_non_cpg == 0 or read.retained_non_cpg > 4
        for read in biscuit_dataset.reads
    )
    assert any(read.retained_non_cpg for read in biscuit_dataset.reads)

    unfiltered = _expected(biscuit_dataset, sim_embedding)
    expected = _expected(
        biscuit_dataset,
        sim_embedding,
        non_converted_threshold=non_converted_threshold,
        em_overconversion_min_cpgs=em_min_cpgs,
    )
    assert len(expected[3]) < len(unfiltered[3])  # the filter has something to do

    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
        **_filter_kwargs(non_converted_threshold, em_min_cpgs),
    )
    _assert_matches_truth(result, expected, sim_embedding)


@pytest.mark.parametrize("threads", [2, 3, 5])
@pytest.mark.parametrize("filtered", [False, True], ids=["unfiltered", "filtered"])
def test_biscuit_parallel_matches_truth_and_serial(
    biscuit_dataset, sim_embedding, threads, filtered
) -> None:
    """--threads N reproduces the serial result exactly, on several hundred reads."""
    kwargs = _filter_kwargs(3, 3) if filtered else {}
    expected = _expected(
        biscuit_dataset,
        sim_embedding,
        non_converted_threshold=3 if filtered else None,
        em_overconversion_min_cpgs=3 if filtered else None,
    )

    serial = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
        threads=1,
        **kwargs,
    )
    parallel = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
        threads=threads,
        **kwargs,
    )

    _assert_matches_truth(parallel, expected, sim_embedding)
    # Bitwise-identical to the serial path, including dtypes.
    assert parallel.matrix.shape == serial.matrix.shape
    for attribute in ("row", "col", "data"):
        ours, theirs = getattr(parallel.matrix, attribute), getattr(
            serial.matrix, attribute
        )
        assert ours.dtype == theirs.dtype, attribute
        assert np.array_equal(ours, theirs), attribute
    assert parallel.tlen.dtype == serial.tlen.dtype
    assert np.array_equal(parallel.tlen, serial.tlen)


def test_rows_follow_chromosome_then_first_appearance(
    biscuit_dataset, sim_embedding
) -> None:
    """Rows are grouped by chromosome and ordered by where each fragment starts."""
    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
    )

    # The reads' own start positions give an independent check of the order:
    # a fragment's row position is set by its first read that covers a CpG.
    first_read_start: dict[tuple[str, str], int] = {}
    for read in biscuit_dataset.reads:  # BAM order
        if _passes_read_filters(read, 20) and read.cpg_calls:
            first_read_start.setdefault((read.chrom, read.name), read.start)
    row_keys = [
        (CHROMOSOMES.index(chrom), first_read_start[(chrom, name)])
        for chrom in CHROMOSOMES
        for name, _, _ in _expected_fragments(biscuit_dataset, sim_embedding, chrom)
    ]
    assert len(row_keys) == result.matrix.shape[0]
    assert row_keys == sorted(row_keys)

    # Within every row the CpG columns are strictly increasing.
    csr = result.matrix.tocsr()
    assert csr.has_sorted_indices
    coo_rows, coo_cols = result.matrix.row, result.matrix.col
    same_row = coo_rows[1:] == coo_rows[:-1]
    assert (coo_cols[1:][same_row] > coo_cols[:-1][same_row]).all()


# ---------------------------------------------------------------------------
# One row per fragment
# ---------------------------------------------------------------------------


def test_each_fragment_is_one_row(biscuit_dataset, sim_embedding) -> None:
    """A read pair yields a single row holding the calls of both mates."""
    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
    )
    usable = [
        read
        for read in biscuit_dataset.reads
        if _passes_read_filters(read, 20) and read.cpg_calls
    ]
    fragments = {(read.chrom, read.name) for read in usable}

    assert result.matrix.shape[0] == len(fragments)
    assert result.matrix.shape[0] < len(usable)  # pairs really were combined

    # No (row, column) is stored twice: a CpG both mates cover is one entry.
    coordinates = set(zip(result.matrix.row.tolist(), result.matrix.col.tolist()))
    assert len(coordinates) == result.matrix.nnz

    # Calls from both mates of a pair, including read 2, reach the output.
    total_calls_by_read2_only = 0
    for chrom in CHROMOSOMES:
        by_name: dict[str, dict[bool, set[int]]] = collections.defaultdict(dict)
        for read in usable:
            if read.chrom == chrom:
                by_name[read.name][read.is_read1] = {p for p, _ in read.cpg_calls}
        for mates in by_name.values():
            total_calls_by_read2_only += len(
                mates.get(False, set()) - mates.get(True, set())
            )
    assert total_calls_by_read2_only > 100
    read1_calls = len(
        {(r.chrom, r.name, p) for r in usable if r.is_read1 for p, _ in r.cpg_calls}
    )
    assert result.matrix.nnz == read1_calls + total_calls_by_read2_only


def test_fragment_lengths(biscuit_dataset, sim_embedding) -> None:
    """tlen is the fragment length: non-negative, the same whichever mate reports it."""
    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
    )
    assert (result.tlen >= 0).all()
    assert result.tlen.min() >= 100  # the simulated read length
    assert result.tlen.max() < 300
    # Rows opened by the right-hand mate (negative TLEN in the BAM) included.
    opened_by_right_mate = 0
    for chrom in CHROMOSOMES:
        names = {
            n for n, _, _ in _expected_fragments(biscuit_dataset, sim_embedding, chrom)
        }
        seen: set[str] = set()
        for read in biscuit_dataset.reads:
            if (
                read.chrom == chrom
                and read.name in names
                and read.name not in seen
                and _passes_read_filters(read, 20)
                and read.cpg_calls
            ):
                seen.add(read.name)
                opened_by_right_mate += read.tlen < 0
    assert opened_by_right_mate > 5


def test_single_end_reads_are_one_fragment_each(sim_embedding, tmp_path) -> None:
    """Unpaired reads: every read is its own row, with fragment length 0."""
    dataset = simulate_bam(
        str(tmp_path),
        TEST_FASTA,
        aligner="biscuit",
        chromosomes=CHROMOSOMES,
        fragments_per_chromosome=80,
        seed=SEED,
        paired=False,
    )
    assert not dataset.paired
    assert len(dataset.reads) == 80 * len(CHROMOSOMES)
    assert {read.is_reverse for read in dataset.reads} == {True, False}

    result = functions.extract_methylation_data_from_bam(
        input_bam=dataset.bam_path, genome_methylation_embedding=sim_embedding
    )
    _assert_matches_truth(result, _expected(dataset, sim_embedding), sim_embedding)
    usable = [r for r in dataset.reads if _passes_read_filters(r, 20) and r.cpg_calls]
    assert result.matrix.shape[0] == len(usable)
    assert (result.tlen == 0).all()


# ---------------------------------------------------------------------------
# Bismark dialect
# ---------------------------------------------------------------------------


def test_bismark_extraction_matches_truth(bismark_dataset, sim_embedding) -> None:
    """Bismark reads from both strands decode to the simulated calls.

    Bismark writes the call for a top-strand alignment on the C of the CpG
    and for a bottom-strand alignment on the G. Regression: the XM string
    used to be read at the C for every read, so every bottom-strand call —
    half of all calls — came out as -1.
    """
    expected = _expected(bismark_dataset, sim_embedding)
    assert len(expected[3]) > 250
    assert any(read.bottom_strand for read in bismark_dataset.reads)

    result = functions.extract_methylation_data_from_bam(
        input_bam=bismark_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
    )
    _assert_matches_truth(result, expected, sim_embedding)

    # Only genuine no-data cases are -1: nowhere near half the calls.
    no_data = (result.matrix.data == -1).mean()
    assert 0 < no_data < 0.1


@pytest.mark.parametrize("strand", ["top", "bottom"])
def test_bismark_single_strand_matches_truth(sim_embedding, tmp_path, strand) -> None:
    """Each strand on its own, so neither can mask a problem with the other."""
    dataset = simulate_bam(
        str(tmp_path),
        TEST_FASTA,
        aligner="bismark",
        chromosomes=CHROMOSOMES,
        fragments_per_chromosome=80,
        seed=SEED,
        strands=(strand,),
    )
    assert {read.bottom_strand for read in dataset.reads} == {strand == "bottom"}
    result = functions.extract_methylation_data_from_bam(
        input_bam=dataset.bam_path, genome_methylation_embedding=sim_embedding
    )
    _assert_matches_truth(result, _expected(dataset, sim_embedding), sim_embedding)
    assert {1, 0} <= set(result.matrix.data.tolist())


def test_bismark_without_xg_tag_matches_truth(sim_embedding, tmp_path) -> None:
    """Without XG, the strand is inferred from the bases the calls sit on."""
    dataset = simulate_bam(
        str(tmp_path),
        TEST_FASTA,
        aligner="bismark",
        chromosomes=CHROMOSOMES,
        fragments_per_chromosome=80,
        seed=SEED,
        bismark_xg_tag=False,
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=dataset.bam_path, genome_methylation_embedding=sim_embedding
    )
    _assert_matches_truth(result, _expected(dataset, sim_embedding), sim_embedding)


@pytest.mark.parametrize("non_converted_threshold, em_min_cpgs", FILTER_CASES)
def test_bismark_filters_match_truth(
    bismark_dataset, sim_embedding, non_converted_threshold, em_min_cpgs
) -> None:
    """The opt-in filters, driven by the XM string, match the simulation."""
    expected = _expected(
        bismark_dataset,
        sim_embedding,
        non_converted_threshold=non_converted_threshold,
        em_overconversion_min_cpgs=em_min_cpgs,
    )
    assert len(expected[3]) < len(_expected(bismark_dataset, sim_embedding)[3])

    result = functions.extract_methylation_data_from_bam(
        input_bam=bismark_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
        **_filter_kwargs(non_converted_threshold, em_min_cpgs),
    )
    _assert_matches_truth(result, expected, sim_embedding)


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("em_filter", [False, True], ids=["unfiltered", "em-filter"])
def test_both_dialects_give_identical_output(
    biscuit_dataset, bismark_dataset, sim_embedding, threads, em_filter
) -> None:
    """The same molecules give the same matrix whichever aligner wrote the BAM.

    The two fixtures hold identical reads; only the tags differ. One path
    reads Bismark's call string, the other the bases themselves, so equal
    output shows both decode strand and position the same way.
    """
    assert biscuit_dataset.reads == bismark_dataset.reads

    kwargs = _filter_kwargs(None, 3) if em_filter else {}
    from_bases = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
        threads=threads,
        **kwargs,
    )
    from_call_string = functions.extract_methylation_data_from_bam(
        input_bam=bismark_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
        threads=threads,
        **kwargs,
    )
    assert from_bases.matrix.shape == from_call_string.matrix.shape
    for attribute in ("row", "col", "data"):
        assert np.array_equal(
            getattr(from_bases.matrix, attribute),
            getattr(from_call_string.matrix, attribute),
        ), attribute
    assert np.array_equal(from_bases.tlen, from_call_string.tlen)


# ---------------------------------------------------------------------------
# Command line, end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threads", ["1", "2"])
def test_cli_output_matches_truth(
    biscuit_dataset, sim_embedding, tmp_path, threads
) -> None:
    """The .npz written by the CLI holds exactly the simulated calls."""
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(
        __main__.main,
        [
            "--input-path",
            biscuit_dataset.bam_path,
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "simulated_cli",
            "--expected-chromosomes",
            ",".join(CHROMOSOMES),
            "--output-dir",
            str(output_dir),
            "--threads",
            threads,
            "--filter-non-converted",
            "--filter-em-overconversion",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Biscuit/bwameth (YD tag)" in result.output

    rows, cols, data, tlens = _expected(
        biscuit_dataset,
        sim_embedding,
        non_converted_threshold=3,
        em_overconversion_min_cpgs=3,
    )
    npz_path = str(output_dir / "simulated.biscuit.methylation.npz")

    matrix = scipy.sparse.load_npz(npz_path)
    assert matrix.shape == (len(tlens), sim_embedding.total_cpg_sites)
    assert matrix.row.tolist() == rows
    assert matrix.col.tolist() == cols
    assert matrix.data.tolist() == data
    assert (
        f"{len(tlens):,} fragments x {sim_embedding.total_cpg_sites:,} CpG sites"
        in result.output
    )

    stored_tlen = read_npz_tlen(npz_path)
    assert stored_tlen is not None
    assert stored_tlen.tolist() == tlens

    metadata = read_npz_metadata(npz_path)
    assert metadata is not None
    assert metadata["expected_chromosomes"] == CHROMOSOMES
    assert metadata["total_cpg_sites"] == sim_embedding.total_cpg_sites
    assert metadata["filters"]["non_converted_reads"]["enabled"] is True
    assert metadata["filters"]["em_overconversion"]["enabled"] is True
    # The output format is recorded, so consumers can tell fragment rows
    # from the per-read rows written by earlier releases.
    assert metadata["format_version"] == OUTPUT_FORMAT_VERSION == 2
    assert metadata["row_unit"] == "fragment"

    inspected = CliRunner().invoke(inspect_main, [npz_path])
    assert inspected.exit_code == 0, inspected.output
    assert f"Fragments:       {len(tlens):,}" in inspected.output
    assert "Format:          v2 (one row per fragment)" in inspected.output
    assert "Reads:" not in inspected.output


# ---------------------------------------------------------------------------
# Column layout: chromosome order and chromosomes absent from the BAM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 2])
def test_reordered_expected_chromosomes_match_truth(biscuit_dataset, threads) -> None:
    """Columns follow expected_chromosomes even when it differs from FASTA order.

    Rows follow the order chromosomes are stored in the embedding (FASTA
    order), columns the requested chromosome order; the two are independent.
    """
    reordered = embedding.GenomeMethylationEmbedding(
        "simulated_reordered",
        expected_chromosomes=["chr3", "chr1", "chr2"],
        fasta_source=TEST_FASTA,
        skip_cache=True,
    )
    assert list(reordered.cpg_sites_dict) == CHROMOSOMES  # FASTA order
    assert reordered.chromosome_offsets["chr3"] == 0

    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=reordered,
        threads=threads,
    )
    _assert_matches_truth(result, _expected(biscuit_dataset, reordered), reordered)

    # chr1 reads come first (row order) yet sit after all chr3 columns.
    first_row_columns = result.matrix.col[result.matrix.row == 0]
    assert first_row_columns.min() >= len(reordered.cpg_sites_dict["chr3"])


@pytest.mark.parametrize("threads", [1, 2])
def test_embedding_subset_of_bam_chromosomes_matches_truth(
    biscuit_dataset, threads
) -> None:
    """Reads on chromosomes outside the embedding are ignored."""
    subset = embedding.GenomeMethylationEmbedding(
        "simulated_subset",
        expected_chromosomes=["chr2"],
        fasta_source=TEST_FASTA,
        skip_cache=True,
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=subset,
        threads=threads,
    )
    expected = _expected(biscuit_dataset, subset)
    assert len(expected[3]) > 50
    _assert_matches_truth(result, expected, subset)
    assert result.matrix.shape[1] == len(subset.cpg_sites_dict["chr2"])


@pytest.mark.parametrize("threads", [1, 3])
def test_chromosome_absent_from_bam_keeps_later_columns_aligned(
    sim_embedding, tmp_path, threads
) -> None:
    """A chromosome with no reads in the BAM does not shift later columns."""
    dataset = simulate_bam(
        str(tmp_path),
        TEST_FASTA,
        aligner="biscuit",
        chromosomes=["chr1", "chr3"],  # nothing on chr2
        fragments_per_chromosome=60,
        seed=SEED,
    )
    result = functions.extract_methylation_data_from_bam(
        input_bam=dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
        threads=threads,
    )
    _assert_matches_truth(result, _expected(dataset, sim_embedding), sim_embedding)

    chr2_start = sim_embedding.chromosome_offsets["chr2"]
    chr3_start = sim_embedding.chromosome_offsets["chr3"]
    columns = result.matrix.col
    assert not ((columns >= chr2_start) & (columns < chr3_start)).any()
    assert (columns >= chr3_start).any()


# ---------------------------------------------------------------------------
# Output array contract
# ---------------------------------------------------------------------------


def test_output_dtypes(biscuit_dataset, sim_embedding) -> None:
    """Calls are int64, indices int32 (for matrices this size), TLEN int32.

    These are the dtypes written to the .npz, so they are part of the output
    format: downstream code sees them after ``scipy.sparse.load_npz``.
    """
    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
    )
    assert isinstance(result.matrix, scipy.sparse.coo_matrix)
    assert result.matrix.dtype == np.int64
    assert result.matrix.data.dtype == np.int64
    assert result.matrix.row.dtype == np.int32
    assert result.matrix.col.dtype == np.int32
    assert result.tlen.dtype == np.int32
    assert result.tlen.ndim == 1

    # The arrays are ordinary, writable, self-contained NumPy arrays.
    for array_ in (
        result.matrix.data,
        result.matrix.row,
        result.matrix.col,
        result.tlen,
    ):
        assert array_.flags.writeable
        assert array_.flags.c_contiguous

    # Row indices are non-decreasing and every row has at least one call.
    assert (np.diff(result.matrix.row) >= 0).all()
    assert len(np.unique(result.matrix.row)) == result.matrix.shape[0]


@pytest.mark.parametrize("threads", [1, 2])
def test_empty_result_has_same_dtypes(
    biscuit_dataset, sim_embedding, tmp_path, threads
):
    """No surviving reads gives an empty matrix with the usual dtypes."""
    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
        quality_limit=255,  # nothing passes
        threads=threads,
    )
    assert result.matrix.shape == (0, sim_embedding.total_cpg_sites)
    assert result.matrix.nnz == 0
    assert result.matrix.data.dtype == np.int64
    assert result.matrix.row.dtype == np.int32
    assert result.matrix.col.dtype == np.int32
    assert result.tlen.dtype == np.int32
    assert len(result.tlen) == 0

    # An empty result still saves and loads.
    npz_path = str(tmp_path / "empty.npz")
    scipy.sparse.save_npz(npz_path, result.matrix)
    assert scipy.sparse.load_npz(npz_path).shape == result.matrix.shape


def test_columns_beyond_int32_use_64bit_indices(biscuit_dataset, tmp_path) -> None:
    """A genome with more than 2**31 CpG columns gets 64-bit column indices.

    No real FASTA that large is practical in a test, so shift a real
    embedding's column space past the int32 limit instead.
    """
    huge = embedding.GenomeMethylationEmbedding(
        "simulated_huge",
        expected_chromosomes=["chr1"],
        fasta_source=TEST_FASTA,
        skip_cache=True,
    )
    normal = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path, genome_methylation_embedding=huge
    )

    shift = 2**31
    huge.chromosome_offsets = {"chr1": shift}
    huge.total_cpg_sites += shift
    shifted = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path, genome_methylation_embedding=huge
    )

    assert shifted.matrix.shape == (
        normal.matrix.shape[0],
        normal.matrix.shape[1] + shift,
    )
    assert shifted.matrix.col.dtype == np.int64
    assert shifted.matrix.row.dtype == np.int64
    assert np.array_equal(
        shifted.matrix.col, normal.matrix.col.astype(np.int64) + shift
    )
    assert np.array_equal(shifted.matrix.row, normal.matrix.row)
    assert np.array_equal(shifted.matrix.data, normal.matrix.data)
    assert np.array_equal(shifted.tlen, normal.tlen)

    # And such a matrix survives a save / load round trip.
    npz_path = str(tmp_path / "huge.npz")
    scipy.sparse.save_npz(npz_path, shifted.matrix)
    loaded = scipy.sparse.load_npz(npz_path)
    assert loaded.shape == shifted.matrix.shape
    assert np.array_equal(loaded.col, shifted.matrix.col)


# ---------------------------------------------------------------------------
# Parallel path plumbing
# ---------------------------------------------------------------------------


def test_workers_are_not_sent_the_embedding(biscuit_dataset, sim_embedding) -> None:
    """Parallel extraction works with an embedding that cannot be pickled.

    Each worker receives only the CpG positions of the chromosome it is
    processing. Sending the whole embedding to every worker cost about a
    gigabyte of memory per worker, and seconds of start-up, for a human genome.
    """
    unpicklable = embedding.GenomeMethylationEmbedding(
        "simulated_unpicklable",
        expected_chromosomes=CHROMOSOMES,
        fasta_source=TEST_FASTA,
        skip_cache=True,
    )
    unpicklable.not_picklable = lambda: None  # type: ignore[attr-defined]

    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=unpicklable,
        threads=2,
    )
    _assert_matches_truth(
        result, _expected(biscuit_dataset, sim_embedding), sim_embedding
    )


# ---------------------------------------------------------------------------
# Non-converted filter: the count itself
# ---------------------------------------------------------------------------


def test_retained_cytosine_count_is_exact(biscuit_dataset, sim_embedding) -> None:
    """For every simulated read, the reference-validated count equals the truth.

    With the CpG index supplying CpG context the count is exact for both
    strands and both mates, including reads that begin or end inside a CpG.
    """
    cpgs = {
        chrom: {position - 1 for position in positions}
        for chrom, positions in sim_embedding.cpg_sites_dict.items()
    }
    with pysam.AlignmentFile(biscuit_dataset.bam_path) as bam:
        segments = list(bam.fetch())
    assert len(segments) == len(biscuit_dataset.reads)

    off_by_one_without_index = 0
    for segment, read in zip(segments, biscuit_dataset.reads):
        assert segment.query_name == read.name
        exact = functions.count_non_cpg_retained_reference(
            segment, read.bottom_strand, cpgs[read.chrom]
        )
        assert exact == read.retained_non_cpg, read.name
        inferred = functions.count_non_cpg_retained_reference(
            segment, read.bottom_strand
        )
        assert inferred - exact in (0, 1)
        off_by_one_without_index += inferred - exact

    # The cases the index exists for do occur: methylated CpGs cut by a read end.
    assert off_by_one_without_index > 0


# ---------------------------------------------------------------------------
# Strand-tag check
# ---------------------------------------------------------------------------

STRAND_CHECK_LINE = re.compile(r"Strand-tag check: ([\d,]+) of ([\d,]+) CpGs")
STRAND_WARNING = re.compile(r"([\d,]+) of ([\d,]+) CpGs \(")


def _counts(pattern, text: str) -> tuple[int, int]:
    """(contradictions, CpGs checked) parsed from a report line or warning."""
    match = pattern.search(text)
    assert match is not None, text
    contradictions, checked = (int(g.replace(",", "")) for g in match.groups())
    return contradictions, checked


def _expected_strand_check(dataset: SimulatedDataset, emb) -> tuple[int, int]:
    """Count contradictions straight from the simulated reads.

    For every read the extractor decodes, and every CpG whose C and G the
    read both covers, look at the base the read's *tag* says is never
    converted: the G if tagged top strand, the C if tagged bottom strand.
    """
    contradictions = checked = 0
    for read in dataset.reads:
        if not _passes_read_filters(read, 20):
            continue
        tagged_bottom = read.bottom_strand != read.mistagged
        sites = emb.cpg_sites_dict[read.chrom]
        end = read.start + len(read.sequence)
        first = bisect.bisect_left(sites, read.start + 1)
        last = bisect.bisect_right(sites, end - 1)
        for position in sites[first:last]:  # 1-based position of the CpG's C
            partner = position - 1 if tagged_bottom else position
            checked += 1
            base = read.sequence[partner - read.start]
            contradictions += base == ("T" if tagged_bottom else "A")
    return contradictions, checked


@pytest.fixture(scope="module")
def mistagged_dataset(tmp_path_factory) -> SimulatedDataset:
    """Biscuit-style reads in which 40% of fragments carry the wrong strand tag."""
    return simulate_bam(
        str(tmp_path_factory.mktemp("mistagged")),
        TEST_FASTA,
        aligner="biscuit",
        chromosomes=CHROMOSOMES,
        seed=SEED,
        mistagged_fraction=0.4,
    )


@pytest.mark.parametrize("threads", [1, 2])
def test_correctly_tagged_reads_pass_the_strand_check(
    biscuit_dataset, sim_embedding, capsys, threads
) -> None:
    """Sequencing errors alone stay far below the warning level."""
    expected = _expected_strand_check(biscuit_dataset, sim_embedding)
    contradictions, checked = expected
    # The simulator misreads a small share of partner bases, as real
    # sequencing does: some contradictions, well under 2%.
    assert checked > 1000
    assert 0 < contradictions < 0.02 * checked

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        functions.extract_methylation_data_from_bam(
            input_bam=biscuit_dataset.bam_path,
            genome_methylation_embedding=sim_embedding,
            threads=threads,
            verbose=True,
        )
    assert _counts(STRAND_CHECK_LINE, capsys.readouterr().out) == expected


@pytest.mark.parametrize("threads", [1, 3])
def test_mistagged_reads_trigger_the_strand_warning(
    mistagged_dataset, sim_embedding, threads
) -> None:
    """Reads whose tag contradicts their bases are reported, with exact counts.

    This is what bwa-meth does to most complementary-strand pairs of a
    non-directional library: the bases are fine, the tag is wrong, and the
    extractor — trusting the tag — reads the wrong base of every CpG.
    """
    assert any(read.mistagged for read in mistagged_dataset.reads)
    assert not all(read.mistagged for read in mistagged_dataset.reads)
    expected = _expected_strand_check(mistagged_dataset, sim_embedding)
    assert expected[0] > 0.02 * expected[1]

    with pytest.warns(RuntimeWarning) as recorded:
        result = functions.extract_methylation_data_from_bam(
            input_bam=mistagged_dataset.bam_path,
            genome_methylation_embedding=sim_embedding,
            threads=threads,
        )

    assert len(recorded) == 1
    message = str(recorded[0].message)
    # The same totals whether counted in one process or summed over workers.
    assert _counts(STRAND_WARNING, message) == expected
    assert mistagged_dataset.bam_path in message
    assert "non-directional" in message
    # It is a warning: the (unreliable) result is still returned.
    assert result.matrix.shape[0] > 0


def test_mistagged_reads_are_decoded_as_methylated(
    mistagged_dataset, biscuit_dataset, sim_embedding
) -> None:
    """Why the warning matters: wrong tags inflate methylation.

    A mis-tagged read is decoded on the base its strand never converts, so
    every CpG in it reads as methylated.
    """

    def methylated_fraction(dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = functions.extract_methylation_data_from_bam(
                input_bam=dataset.bam_path, genome_methylation_embedding=sim_embedding
            )
        calls = result.matrix.data[result.matrix.data >= 0]
        return float(calls.mean())

    correct = methylated_fraction(biscuit_dataset)
    inflated = methylated_fraction(mistagged_dataset)
    assert inflated > correct + 0.05


def test_bismark_reads_are_not_strand_checked(
    bismark_dataset, sim_embedding, capsys
) -> None:
    """Bismark's own strand resolution is trusted: no check, no report line."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        functions.extract_methylation_data_from_bam(
            input_bam=bismark_dataset.bam_path,
            genome_methylation_embedding=sim_embedding,
            verbose=True,
        )
    assert "Strand-tag check" not in capsys.readouterr().out


def test_cli_reports_the_strand_warning(mistagged_dataset, tmp_path) -> None:
    """The CLI shows the warning in the BAM's report and still writes the file."""
    result = CliRunner().invoke(
        __main__.main,
        [
            "--input-path",
            mistagged_dataset.bam_path,
            "--reference-fasta",
            TEST_FASTA,
            "--genome-name",
            "simulated_mistagged",
            "--expected-chromosomes",
            ",".join(CHROMOSOMES),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "  WARNING: " in result.output
    assert "strand tag" in result.output
    assert "Processed:     1 BAM(s)" in result.output
    assert (tmp_path / "out" / "simulated.biscuit.methylation.npz").exists()
