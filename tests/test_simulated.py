"""End-to-end extraction tests against simulated reads with known truth.

``tests/simulate.py`` generates paired-end bisulfite reads from the bundled
FASTA with a known methylation state at every CpG. These tests check the
extractor's output value-for-value against that truth: every row, column,
methylation call and template length, for a few hundred reads spread over
three chromosomes and both strands.
"""

import numpy as np
import pytest
import scipy.sparse
from click.testing import CliRunner

from bam2tensor import __main__, embedding, functions
from bam2tensor.metadata import read_npz_metadata, read_npz_tlen
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
def bismark_top_dataset(tmp_path_factory) -> SimulatedDataset:
    """Bismark-style reads (XM call string) from the original top strand."""
    return simulate_bam(
        str(tmp_path_factory.mktemp("bismark_top")),
        TEST_FASTA,
        aligner="bismark",
        chromosomes=CHROMOSOMES,
        seed=SEED,
        strands=("top",),
    )


@pytest.fixture(scope="module")
def bismark_bottom_dataset(tmp_path_factory) -> SimulatedDataset:
    """Bismark-style reads (XM call string) from the original bottom strand."""
    return simulate_bam(
        str(tmp_path_factory.mktemp("bismark_bottom")),
        TEST_FASTA,
        aligner="bismark",
        chromosomes=CHROMOSOMES,
        seed=SEED,
        strands=("bottom",),
    )


# ---------------------------------------------------------------------------
# Expected output, derived from the simulation truth
# ---------------------------------------------------------------------------


def _on_parent_strand(read: SimulatedRead) -> bool:
    """Whether the Biscuit/bwameth/gem3 path keeps this read.

    That path keeps a read only when its alignment orientation agrees with
    its bisulfite strand tag, i.e. one mate of each pair. (Bismark-style
    reads are not subject to this filter.)
    """
    return read.bottom_strand == read.is_reverse


def _is_em_overconverted(read: SimulatedRead, min_cpgs: int) -> bool:
    """The EM over-conversion rule: enough CpGs, every one called unmethylated."""
    states = [state for _, state in read.cpg_calls]
    return len(states) >= min_cpgs and all(state == 0 for state in states)


def _expected(
    dataset: SimulatedDataset,
    emb,
    quality_limit: int = 20,
    non_converted_threshold=None,
    em_overconversion_min_cpgs=None,
):
    """Build the COO triplets and TLENs the extractor should produce.

    Rows follow the order chromosomes appear in the embedding and, within a
    chromosome, the order of reads in the BAM. A read contributes a row only
    if it passes the read filters and covers at least one CpG.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    tlens: list[int] = []
    for chrom in emb.cpg_sites_dict:
        for read in dataset.reads:
            if read.chrom != chrom:
                continue
            if read.mapq < quality_limit or read.duplicate:
                continue
            if dataset.aligner == "biscuit" and not _on_parent_strand(read):
                continue
            if (
                non_converted_threshold is not None
                and read.retained_non_cpg >= non_converted_threshold
            ):
                continue
            if not read.cpg_calls:
                continue
            if em_overconversion_min_cpgs is not None and _is_em_overconverted(
                read, em_overconversion_min_cpgs
            ):
                continue
            for position, state in read.cpg_calls:
                rows.append(len(tlens))
                cols.append(emb.genomic_position_to_embedding(chrom, position))
                data.append(state)
            tlens.append(read.tlen)
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
    assert any(read.duplicate for read in reads)
    assert any(read.mapq < 20 for read in reads)
    assert any(read.non_converted for read in reads)
    assert any(not read.cpg_calls for read in reads)
    assert any(_is_em_overconverted(read, 3) for read in reads)
    assert {read.tlen > 0 for read in reads} == {True, False}

    # Every simulated CpG position is a real column of the embedding.
    for read in reads:
        for position, _ in read.cpg_calls:
            sim_embedding.genomic_position_to_embedding(read.chrom, position)

    # A meaningful number of rows survive the default filters.
    assert len(_expected(biscuit_dataset, sim_embedding)[3]) > 200


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
    """The opt-in filters drop exactly the reads the simulation says they should."""
    # The reference-validated count can differ from the true count by one at a
    # read boundary, so make sure no simulated read sits near the threshold.
    assert all(
        read.retained_non_cpg == 0 or read.retained_non_cpg > 4
        for read in biscuit_dataset.reads
    )

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


def test_biscuit_rows_follow_chromosome_then_coordinate_order(
    biscuit_dataset, sim_embedding
) -> None:
    """Rows are grouped by chromosome and sorted by position within each."""
    result = functions.extract_methylation_data_from_bam(
        input_bam=biscuit_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
    )
    csr = result.matrix.tocsr()
    first_columns = [int(csr.indices[csr.indptr[row]]) for row in range(csr.shape[0])]
    # Columns are ordered by (chromosome, position), so a read's first column
    # never decreases from one row to the next.
    assert first_columns == sorted(first_columns)
    chromosomes_seen = [
        sim_embedding.embedding_to_genomic_position(column)[0]
        for column in first_columns
    ]
    assert sorted(set(chromosomes_seen)) == CHROMOSOMES


# ---------------------------------------------------------------------------
# Bismark dialect
# ---------------------------------------------------------------------------


def test_bismark_top_strand_extraction_matches_truth(
    bismark_top_dataset, sim_embedding
) -> None:
    """Top-strand Bismark reads (both mates) decode to the simulated calls."""
    expected = _expected(bismark_top_dataset, sim_embedding)
    # Both mates of each pair are kept, forward and reverse alike.
    assert len(expected[3]) > 400

    result = functions.extract_methylation_data_from_bam(
        input_bam=bismark_top_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
    )
    _assert_matches_truth(result, expected, sim_embedding)


@pytest.mark.parametrize("non_converted_threshold, em_min_cpgs", FILTER_CASES)
def test_bismark_top_strand_filters_match_truth(
    bismark_top_dataset, sim_embedding, non_converted_threshold, em_min_cpgs
) -> None:
    """The opt-in filters, driven by the XM string, match the simulation."""
    expected = _expected(
        bismark_top_dataset,
        sim_embedding,
        non_converted_threshold=non_converted_threshold,
        em_overconversion_min_cpgs=em_min_cpgs,
    )
    assert len(expected[3]) < len(_expected(bismark_top_dataset, sim_embedding)[3])

    result = functions.extract_methylation_data_from_bam(
        input_bam=bismark_top_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
        **_filter_kwargs(non_converted_threshold, em_min_cpgs),
    )
    _assert_matches_truth(result, expected, sim_embedding)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bismark writes the CpG call of a bottom-strand (XG:Z:GA) alignment on "
        "the G of the CpG, but the extractor reads the XM string at the C, so "
        "every such call currently decodes as -1 (no data). Tracked with the "
        "strand-handling review; remove this marker when that is resolved."
    ),
)
def test_bismark_bottom_strand_extraction_matches_truth(
    bismark_bottom_dataset, sim_embedding
) -> None:
    """Bottom-strand Bismark reads decode to the simulated calls."""
    result = functions.extract_methylation_data_from_bam(
        input_bam=bismark_bottom_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
    )
    _assert_matches_truth(
        result, _expected(bismark_bottom_dataset, sim_embedding), sim_embedding
    )


def test_bismark_bottom_strand_rows_and_columns_match_truth(
    bismark_bottom_dataset, sim_embedding
) -> None:
    """Bottom-strand Bismark reads land in the right rows and columns.

    Independent of how the calls themselves decode (see the xfail above), the
    set of reads kept, their order, the CpG columns they touch and their
    TLENs must match the simulation.
    """
    rows, cols, _, tlens = _expected(bismark_bottom_dataset, sim_embedding)
    result = functions.extract_methylation_data_from_bam(
        input_bam=bismark_bottom_dataset.bam_path,
        genome_methylation_embedding=sim_embedding,
    )
    assert result.matrix.shape == (len(tlens), sim_embedding.total_cpg_sites)
    assert result.matrix.row.tolist() == rows
    assert result.matrix.col.tolist() == cols
    assert result.tlen.tolist() == tlens


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
    assert f"{len(tlens):,} reads x {sim_embedding.total_cpg_sites:,} CpG sites" in (
        result.output
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
    assert 0 < len(expected[3]) < len(_expected(biscuit_dataset, subset, 0)[3]) + 1
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
