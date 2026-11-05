"""Command-line interface for bam2tensor.

This module provides the CLI entry point for bam2tensor, allowing users to
extract methylation data from bisulfite-sequencing (BS-seq) and enzymatic
methylation sequencing (EM-seq) BAM files via the command line.

Example:
    Basic usage with a single BAM file::

        $ bam2tensor --input-path sample.bam --reference-fasta ref.fa --genome-name hg38

    Process all BAM files in a directory::

        $ bam2tensor --input-path /path/to/bams/ --reference-fasta ref.fa --genome-name hg38

    With verbose output and custom quality threshold::

        $ bam2tensor --input-path sample.bam --reference-fasta ref.fa \\
            --genome-name hg38 --quality-limit 30 --verbose

    Auto-download a reference genome::

        $ bam2tensor --input-path sample.bam --download-reference hg38
"""

import math
import os
import sys
import time
import warnings
import click

import scipy.sparse

from bam2tensor.embedding import GenomeMethylationEmbedding

from bam2tensor import __version__
from bam2tensor.functions import (
    detect_aligner,
    extract_methylation_data_from_bam,
)
from bam2tensor.metadata import (
    OUTPUT_FORMAT_VERSION,
    OUTPUT_ROW_UNIT,
    compute_cpg_index_crc32,
    write_npz_metadata,
    write_npz_tlen,
)
from bam2tensor.reference import (
    KNOWN_GENOMES,
    download_reference as download_reference_fn,
    list_available_genomes,
)


def _format_elapsed(seconds: float) -> str:
    """Format an elapsed time as a human-readable string.

    For durations under 60 seconds, returns seconds only. For longer
    durations, returns minutes and seconds.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        A formatted string such as ``"12.34 seconds"`` or
        ``"3 min 7.12 sec"``.

    Example:
        >>> _format_elapsed(5.123)
        '5.12 seconds'

        >>> _format_elapsed(185.7)
        '3 min 5.70 sec'
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    mins = math.floor(seconds / 60)
    secs = seconds - mins * 60
    return f"{mins} min {secs:.2f} sec"


def get_input_bams(input_path: str) -> list:
    """Find all BAM files to process from a given input path.

    Determines whether the input is a single BAM file or a directory,
    and returns a list of BAM file paths to process. If a directory
    is provided, it recursively searches for all files with a .bam
    extension.

    Args:
        input_path: Path to a single .bam file or a directory containing
            BAM files. Directories are searched recursively.

    Returns:
        A list of absolute paths to .bam files to process. If input_path
        is a single file, returns a single-element list. If input_path
        is a directory, returns all .bam files found recursively.

    Raises:
        ValueError: If input_path does not exist or is neither a file
            nor a directory.

    Example:
        >>> # xdoctest: +SKIP
        >>> get_input_bams("/path/to/sample.bam")
        ['/path/to/sample.bam']

        >>> get_input_bams("/path/to/bam_directory/")
        ['/path/to/bam_directory/sample1.bam',
         '/path/to/bam_directory/subdir/sample2.bam']
    """
    # Check if input_path is a file or a directory
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        # Recursively find all .bam files in this path, and add them to a list
        bams_to_process = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".bam"):
                    bams_to_process.append(os.path.join(root, file))
        return bams_to_process
    else:
        raise ValueError(f"Input path {input_path} is not a file or a directory.")


def get_output_path(input_bam: str, output_dir: str | None = None) -> str:
    """Compute the output file path for a given input BAM file.

    Generates the path for the ``.methylation.npz`` output file. If an
    output directory is specified, the file is placed there using only the
    basename of the input file. Otherwise, the output file is placed next
    to the input BAM file.

    Args:
        input_bam: Path to the input BAM file.
        output_dir: Optional output directory. If None, the output file
            is placed in the same directory as the input BAM.

    Returns:
        The full path to the output ``.methylation.npz`` file.

    Example:
        >>> get_output_path("/data/sample.bam")
        '/data/sample.methylation.npz'

        >>> get_output_path("/data/sample.bam", "/output")
        '/output/sample.methylation.npz'
    """
    basename = os.path.splitext(os.path.basename(input_bam))[0] + ".methylation.npz"
    if output_dir:
        return os.path.join(output_dir, basename)
    return os.path.splitext(input_bam)[0] + ".methylation.npz"


def validate_input_output(
    bams_to_process: list,
    overwrite: bool,
    output_dir: str | None = None,
) -> None:
    """Validate that input BAM files are readable and output paths are writable.

    Checks each BAM file in the list to ensure it can be read, and verifies
    that the corresponding output .methylation.npz file can be written.
    Running these checks up front means a permissions problem is reported
    before any (potentially hours-long) extraction starts, rather than
    when the first result is saved.

    Args:
        bams_to_process: List of paths to BAM files to validate.
        overwrite: If True, existing output files will be overwritten, so
            they must be writable. If False and an output file exists, it
            will be skipped during processing (and is not validated here).
        output_dir: Optional output directory for .methylation.npz files.
            If None, output files are placed next to the input BAM files.

    Raises:
        ValueError: If any input BAM file is not readable, if the output
            directory is not writable, or if ``overwrite`` is set and an
            existing output file is not writable.

    Note:
        Output files are named by replacing the .bam extension with
        .methylation.npz (e.g., sample.bam -> sample.methylation.npz).
    """

    for bam_file in bams_to_process:
        if not os.access(bam_file, os.R_OK):
            raise ValueError(f"Input file is not readable: {bam_file}")

        output_file = get_output_path(bam_file, output_dir)
        if os.path.exists(output_file):
            # Without --overwrite the BAM is skipped, so nothing is written.
            if overwrite and not os.access(output_file, os.W_OK):
                raise ValueError(
                    f"Output file exists and is not writable: {output_file}"
                )
        elif not os.access(os.path.dirname(os.path.abspath(output_file)), os.W_OK):
            raise ValueError(f"Output file path is not writable: {output_file}")


@click.command(
    help="Extract per-fragment methylation data from an aligned bisulfite-seq or EM-seq .bam file and export the data as a SciPy sparse matrix (one row per sequenced DNA fragment)."
)
@click.version_option()
@click.option(
    "--input-path",
    help="Input .bam file OR directory to recursively process.",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
)
@click.option(
    "--genome-name",
    help="A custom string referring to your genome name, used to save a cache file (e.g. hg38, hg39-no-alt, etc.).",
    required=False,
    default=None,
    type=str,
)
@click.option(
    "--expected-chromosomes",
    # Useful to filter out alt chromosomes, etc.
    help="A comma-separated list of chromosomes to expect in the .fa genome. Defaults to hg38 chromosomes.",
    required=False,
    default=None,
)
@click.option(
    "--reference-fasta",
    help="Reference genome fasta file (critical to determine CpG sites).",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--quality-limit",
    help="Quality filter for aligned reads (default = 20)",
    default=20,
    type=int,
)
@click.option(
    "--filter-non-converted",
    help=(
        "Drop fragments in which a read has >= --non-converted-threshold "
        "retained non-CpG cytosines, the signature of incomplete "
        "bisulfite/EM-seq conversion (port of "
        "nebiolabs/mark-nonconverted-reads). Default: off."
    ),
    is_flag=True,
)
@click.option(
    "--non-converted-threshold",
    help=(
        "Minimum count of retained non-CpG cytosines in a read for its "
        "fragment to be dropped (default = 3, matches NEB "
        "mark-nonconverted-reads)."
    ),
    default=3,
    type=int,
)
@click.option(
    "--filter-em-overconversion",
    help=(
        "Drop EM-seq fragments whose covered CpGs are all called unmethylated "
        "and cover at least --em-overconversion-min-cpgs sites (heuristic "
        "for the fragment-level over-conversion artifact described in "
        "Loyfer et al. bioRxiv 2026.03.24.713040). Default: off."
    ),
    is_flag=True,
)
@click.option(
    "--em-overconversion-min-cpgs",
    help=(
        "Minimum covered CpG count required before the EM over-conversion "
        "filter will drop a fragment (default = 3)."
    ),
    default=3,
    type=int,
)
@click.option(
    "--threads",
    help=(
        "Number of worker processes for per-chromosome extraction "
        "(default = 1). With threads > 1, chromosomes are fanned out "
        "across worker processes. Output is bitwise-identical to the "
        "single-threaded path. Recommended: number of physical cores "
        "for whole-genome WGBS/EM-seq BAMs."
    ),
    default=1,
    type=int,
)
@click.option("--verbose", help="Verbose output.", is_flag=True)
@click.option("--skip-cache", help="De-novo generate CpG sites (slow).", is_flag=True)
@click.option(
    "--debug",
    help="Debug mode (extensive validity checking + debug messages).",
    is_flag=True,
    type=bool,
)
@click.option("--overwrite", help="Overwrite output file if it exists.", is_flag=True)
@click.option(
    "--output-dir",
    help="Output directory for .methylation.npz files. Defaults to same directory as the input BAM.",
    required=False,
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    "--download-reference",
    help="Download and cache a known reference genome (e.g. hg38, hg19, mm10, T2T-CHM13). Use --list-genomes to see options.",
    required=False,
    default=None,
    type=click.Choice(sorted(KNOWN_GENOMES.keys()), case_sensitive=False),
)
@click.option(
    "--list-genomes",
    help="List available reference genomes for download and exit.",
    is_flag=True,
)
def main(
    input_path: str,
    genome_name: str | None,
    expected_chromosomes: str | None,
    reference_fasta: str | None,
    quality_limit: int,
    filter_non_converted: bool,
    non_converted_threshold: int,
    filter_em_overconversion: bool,
    em_overconversion_min_cpgs: int,
    threads: int,
    verbose: bool,
    skip_cache: bool,
    debug: bool,
    overwrite: bool,
    output_dir: str | None,
    download_reference: str | None,
    list_genomes: bool,
) -> None:
    """Extract methylation data from BAM files and save as sparse matrices.

    Main entry point for the bam2tensor CLI. Processes one or more BAM files,
    extracting per-fragment CpG methylation data and saving the results as
    SciPy sparse COO matrices in .npz format.

    The workflow is:
        1. Load or generate a CpG site index for the reference genome
        2. Find all BAM files to process (single file or directory)
        3. Validate input/output permissions
        4. For each BAM file:
           a. Extract methylation states from aligned reads
           b. Build a sparse matrix (rows=fragments, columns=CpG sites)
           c. Save to .methylation.npz file

    Args:
        input_path: Path to a BAM file or directory to process recursively.
        genome_name: Identifier for the genome (e.g., "hg38"). Used for
            naming the CpG site cache file. If not provided, inferred from
            ``--download-reference``.
        expected_chromosomes: Comma-separated list of chromosomes to process.
            Chromosomes not in this list are skipped. If not provided,
            defaults to hg38 chromosomes or is inferred from
            ``--download-reference``.
        reference_fasta: Path to the reference genome FASTA file. Must match
            the genome used for alignment. Not required if
            ``--download-reference`` is used.
        quality_limit: Minimum mapping quality (MAPQ) threshold. Reads below
            this quality are excluded.
        filter_non_converted: If True, drop fragments in which a read
            has at least ``non_converted_threshold`` retained non-CpG
            cytosines — indicating incomplete bisulfite/EM-seq conversion.
        non_converted_threshold: Threshold used by the non-converted
            read filter.
        filter_em_overconversion: If True, drop fragments whose covered
            CpGs are all called unmethylated and cover at least
            ``em_overconversion_min_cpgs`` sites — heuristic for EM-seq
            fragment-level over-conversion (Loyfer et al. 2026).
        em_overconversion_min_cpgs: Minimum covered CpG count required
            before the over-conversion filter will drop a fragment.
        threads: Number of worker processes for per-chromosome
            extraction. Default 1.
        verbose: If True, print detailed progress information.
        skip_cache: If True, regenerate the CpG site index even if a cache
            file exists.
        debug: If True, enable extensive validation and debug output.
        overwrite: If True, overwrite existing output files. Otherwise,
            skip BAM files that already have output.
        output_dir: Optional output directory for .methylation.npz files.
            If not provided, output files are written next to the input BAMs.
        download_reference: If specified, download and cache this reference
            genome (e.g., "hg38") instead of requiring ``--reference-fasta``.
        list_genomes: If True, list available reference genomes and exit.

    Raises:
        UsageError: If neither ``--reference-fasta`` nor
            ``--download-reference`` is specified, or if ``--genome-name``
            is missing when not using ``--download-reference``.
        ClickException: If the CpG site index cannot be built for the
            requested configuration (for example, an expected chromosome
            is absent from the reference FASTA, or the cache on disk was
            built for a different chromosome list).

    Note:
        This function is decorated with Click options and is typically
        invoked via the command line rather than called directly.

        A BAM that cannot be processed (missing index, chromosome names
        that do not match the reference, unreadable file) is reported and
        skipped so the remaining BAMs still run. If any BAM failed, the
        process exits with status 1 after printing the summary.
    """
    # Handle --list-genomes: print available genomes and exit
    if list_genomes:
        genomes = list_available_genomes()
        print("Available reference genomes for --download-reference:\n")
        for name, info in sorted(genomes.items()):
            print(f"  {name:12s}  {info['description']}")
            print(f"               Chromosomes: {info['expected_chromosomes'][:50]}...")
            print()
        sys.exit(0)

    # Handle --download-reference: download genome and set defaults
    if download_reference:
        genome_info = KNOWN_GENOMES[download_reference]
        fasta_path = download_reference_fn(download_reference, verbose=verbose)
        reference_fasta = str(fasta_path)
        if genome_name is None:
            genome_name = download_reference
        if expected_chromosomes is None:
            expected_chromosomes = genome_info["expected_chromosomes"]

    # Validate that we have a reference FASTA (either via --reference-fasta or --download-reference)
    if reference_fasta is None:
        raise click.UsageError(
            "Either --reference-fasta or --download-reference must be specified."
        )

    # Apply defaults for genome-name and expected-chromosomes
    if genome_name is None:
        raise click.UsageError(
            "--genome-name is required when not using --download-reference."
        )
    if expected_chromosomes is None:
        expected_chromosomes = ",".join(
            ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
        )

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    time_start = time.time()

    # ── Header ──────────────────────────────────────────────────────────
    print(f"bam2tensor v{__version__}")
    print("By Nick Semenkovich <semenko@alum.mit.edu>.")
    print("=" * 72)

    # ── Configuration ───────────────────────────────────────────────────
    chrom_list = expected_chromosomes.split(",")
    if len(chrom_list) <= 4:
        chrom_display = expected_chromosomes
    else:
        chrom_display = (
            f"{len(chrom_list)} ({chrom_list[0]}, {chrom_list[1]}, "
            f"... {chrom_list[-2]}, {chrom_list[-1]})"
        )

    print(f"  Genome:        {genome_name}")
    print(f"  Reference:     {reference_fasta}")
    print(f"  Chromosomes:   {chrom_display}")
    print(f"  Quality limit: MAPQ >= {quality_limit}")
    if threads > 1:
        print(f"  Threads:       {threads} worker process(es)")
    if filter_non_converted:
        print(
            f"  Filters:       non-converted reads (>= "
            f"{non_converted_threshold} retained non-CpG Cs)"
        )
    if filter_em_overconversion:
        print(
            f"                 EM over-conversion (all-unmethylated, >= "
            f"{em_overconversion_min_cpgs} CpGs)"
        )
    if output_dir:
        print(f"  Output dir:    {output_dir}")
    else:
        print("  Output:        next to input BAMs")
    print()

    # ── Load CpG index ──────────────────────────────────────────────────
    print(f"Loading CpG site index for {genome_name}...")
    try:
        genome_methylation_embedding = GenomeMethylationEmbedding(
            genome_name=genome_name,
            expected_chromosomes=chrom_list,
            fasta_source=reference_fasta,
            skip_cache=skip_cache,
            verbose=verbose,
        )
    except ValueError as e:
        # Configuration problems (chromosome names absent from the FASTA, a
        # cache built for a different chromosome list): report the message,
        # not a traceback.
        raise click.ClickException(str(e)) from e
    n_chroms = len(genome_methylation_embedding.cpg_sites_dict)
    cpg_crc32 = compute_cpg_index_crc32(genome_methylation_embedding)
    print(
        f"  Total CpG sites: {genome_methylation_embedding.total_cpg_sites:,}"
        f" across {n_chroms} chromosome(s)"
    )
    print(f"  CpG index CRC32: {cpg_crc32}")
    print(f"  Index loaded in {_format_elapsed(time.time() - time_start)}")

    # ── Discover BAM files ──────────────────────────────────────────────
    bams_to_process = get_input_bams(input_path)
    n_bams = len(bams_to_process)
    print(f"\nFound {n_bams} BAM file(s):")
    for idx, bam_file in enumerate(bams_to_process, 1):
        print(f"  {idx}. {bam_file}")

    validate_input_output(
        bams_to_process=bams_to_process,
        overwrite=overwrite,
        output_dir=output_dir,
    )

    # ── Process each BAM ────────────────────────────────────────────────
    errors_list: list[Exception] = []
    skip_count = 0
    for i, input_bam in enumerate(bams_to_process):
        time_bam = time.time()
        output_file = get_output_path(input_bam, output_dir)
        bam_basename = os.path.basename(input_bam)

        print()
        print("=" * 72)
        print(f"[{i + 1}/{n_bams}] {bam_basename}")
        print("-" * 72)

        if os.path.exists(output_file) and not overwrite:
            print("  Skipped: output already exists (use --overwrite to replace)")
            skip_count += 1
            continue

        # Detect aligner
        aligner = detect_aligner(input_bam)
        print(f"  Aligner:       {aligner}")

        # Extract
        print("  Extracting methylation data...")
        try:
            # Record warnings raised during extraction (e.g. the non-converted
            # filter could not evaluate reads lacking an MD tag) so they are
            # shown with this BAM's report rather than as a raw Python warning.
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", RuntimeWarning)
                extraction_result = extract_methylation_data_from_bam(
                    input_bam=input_bam,
                    genome_methylation_embedding=genome_methylation_embedding,
                    quality_limit=quality_limit,
                    filter_non_converted=filter_non_converted,
                    non_converted_threshold=non_converted_threshold,
                    filter_em_overconversion=filter_em_overconversion,
                    em_overconversion_min_cpgs=em_overconversion_min_cpgs,
                    threads=threads,
                    verbose=verbose,
                    debug=debug,
                )
        except (OSError, ValueError) as e:
            # Problems with this particular BAM (missing index, chromosome
            # names that do not match the reference, unreadable/truncated
            # file). Report it and carry on, so one bad file does not abort a
            # whole directory run; the exit status reflects the failure.
            print(f"  ERROR: {e}")
            errors_list.append(e)
            continue
        for caught in caught_warnings:
            print(f"  WARNING: {caught.message}")

        # Matrix stats
        n_fragments = extraction_result.matrix.shape[0]
        n_cpgs = extraction_result.matrix.shape[1]
        n_data = extraction_result.matrix.nnz
        print(
            f"  Result:        {n_fragments:,} fragments x {n_cpgs:,} CpG sites"
            f" ({n_data:,} data points)"
        )

        # Save
        scipy.sparse.save_npz(output_file, extraction_result.matrix, compressed=True)
        write_npz_tlen(output_file, extraction_result.tlen)
        write_npz_metadata(
            output_file,
            {
                "bam2tensor_version": __version__,
                "format_version": OUTPUT_FORMAT_VERSION,
                "row_unit": OUTPUT_ROW_UNIT,
                "genome_name": genome_name,
                "expected_chromosomes": chrom_list,
                "total_cpg_sites": genome_methylation_embedding.total_cpg_sites,
                "cpg_index_crc32": cpg_crc32,
                "filters": {
                    "non_converted_reads": {
                        "enabled": filter_non_converted,
                        "threshold": non_converted_threshold,
                    },
                    "em_overconversion": {
                        "enabled": filter_em_overconversion,
                        "min_cpgs": em_overconversion_min_cpgs,
                    },
                },
            },
        )
        print(f"  Output:        {output_file}")
        print(f"  Time:          {_format_elapsed(time.time() - time_bam)}")

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("Summary")
    print("-" * 72)
    processed = n_bams - skip_count - len(errors_list)
    print(f"  Processed:     {processed} BAM(s)")
    if skip_count:
        print(f"  Skipped:       {skip_count} (existing output)")
    if errors_list:
        print(f"  Errors:        {len(errors_list)}")
        for error in errors_list:
            print(f"                 - {error}")
    print(f"  Total time:    {_format_elapsed(time.time() - time_start)}")
    print("=" * 72)

    # A run in which any BAM failed must not look like a success to a
    # scheduler or pipeline.
    if errors_list:
        sys.exit(1)


if __name__ == "__main__":
    main(prog_name="bam2tensor")  # pylint: disable=no-value-for-parameter
