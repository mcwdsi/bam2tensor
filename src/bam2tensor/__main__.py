"""Command-line interface for bam2tensor.

This module provides the CLI entry point for bam2tensor, allowing users to
extract methylation data from BAM files via the command line.

Example:
    Basic usage with a single BAM file::

        $ bam2tensor --input-path sample.bam --reference-fasta ref.fa --genome-name hg38

    Process all BAM files in a directory::

        $ bam2tensor --input-path /path/to/bams/ --reference-fasta ref.fa --genome-name hg38

    With verbose output and custom quality threshold::

        $ bam2tensor --input-path sample.bam --reference-fasta ref.fa \\
            --genome-name hg38 --quality-limit 30 --verbose
"""

import os
import time
import click

import scipy.sparse

from bam2tensor.embedding import GenomeMethylationEmbedding

from bam2tensor.functions import (
    extract_methylation_data_from_bam,
)


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


def validate_input_output(bams_to_process: list, overwrite: bool) -> None:
    """Validate that input BAM files are readable and output paths are writable.

    Checks each BAM file in the list to ensure it can be read, and verifies
    that the corresponding output .methylation.npz file can be written.
    Prints warnings if output files already exist and --overwrite is specified.

    Args:
        bams_to_process: List of paths to BAM files to validate.
        overwrite: If True, allows overwriting existing output files.
            If False and an output file exists, it will be skipped during
            processing (not validated here).

    Raises:
        ValueError: If any input BAM file is not readable, or if the
            output directory is not writable.

    Note:
        Output files are named by replacing the .bam extension with
        .methylation.npz (e.g., sample.bam -> sample.methylation.npz).
    """

    for bam_file in bams_to_process:
        if not os.access(bam_file, os.R_OK):
            raise ValueError(f"Input file is not readable: {bam_file}")

        output_file = os.path.splitext(bam_file)[0] + ".methylation.npz"
        if os.path.exists(output_file):
            if overwrite and os.access(output_file, os.W_OK):
                print(
                    "\t\tOutput file exists and --overwrite specified. Will overwrite existing .methylation.npz file."
                )
        else:
            if not os.access(os.path.dirname(os.path.abspath(output_file)), os.W_OK):
                raise ValueError(f"Output file path is not writable: {output_file}")


@click.command(
    help="Extract read-level methylation data from an aligned .bam file and export the data as a SciPy sparse matrix."
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
    required=True,
    type=str,
)
@click.option(
    "--expected-chromosomes",
    # Useful to filter out alt chromosomes, etc.
    help="A comma-separated list of chromosomes to expect in the .fa genome. Defaults to hg38 chromosomes.",
    required=False,
    default=",".join(["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]),
)
@click.option(
    "--reference-fasta",
    help="Reference genome fasta file (critical to determine CpG sites).",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--quality-limit",
    help="Quality filter for aligned reads (default = 20)",
    default=20,
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
def main(
    input_path: str,
    genome_name: str,
    expected_chromosomes: str,
    reference_fasta: str,
    quality_limit: int,
    verbose: bool,
    skip_cache: bool,
    debug: bool,
    overwrite: bool,
) -> None:
    """Extract methylation data from BAM files and save as sparse matrices.

    Main entry point for the bam2tensor CLI. Processes one or more BAM files,
    extracting read-level CpG methylation data and saving the results as
    SciPy sparse COO matrices in .npz format.

    The workflow is:
        1. Load or generate a CpG site index for the reference genome
        2. Find all BAM files to process (single file or directory)
        3. Validate input/output permissions
        4. For each BAM file:
           a. Extract methylation states from aligned reads
           b. Build a sparse matrix (rows=reads, columns=CpG sites)
           c. Save to .methylation.npz file

    Args:
        input_path: Path to a BAM file or directory to process recursively.
        genome_name: Identifier for the genome (e.g., "hg38"). Used for
            naming the CpG site cache file.
        expected_chromosomes: Comma-separated list of chromosomes to process.
            Chromosomes not in this list are skipped.
        reference_fasta: Path to the reference genome FASTA file. Must match
            the genome used for alignment.
        quality_limit: Minimum mapping quality (MAPQ) threshold. Reads below
            this quality are excluded.
        verbose: If True, print detailed progress information.
        skip_cache: If True, regenerate the CpG site index even if a cache
            file exists.
        debug: If True, enable extensive validation and debug output.
        overwrite: If True, overwrite existing output files. Otherwise,
            skip BAM files that already have output.

    Note:
        This function is decorated with Click options and is typically
        invoked via the command line rather than called directly.
    """
    time_start = time.time()
    # Print run information
    print(f"Genome name: {genome_name}")
    print(f"Reference fasta: {reference_fasta}")
    print(f"Expected chromosomes: {expected_chromosomes}")
    print(f"Input path: {input_path}")
    print(f"\nLoading (or generating) methylation embedding for: {genome_name}")

    # Create (or load) a GenomeMethylationEmbedding object
    genome_methylation_embedding = GenomeMethylationEmbedding(
        genome_name=genome_name,
        expected_chromosomes=expected_chromosomes.split(","),
        fasta_source=reference_fasta,
        skip_cache=skip_cache,
        verbose=verbose,
    )

    print(f"\nTime elapsed: {time.time() - time_start:.2f} seconds")

    #################################################
    # Establish input/output for .bam files
    #################################################

    bams_to_process = get_input_bams(input_path)

    print(f"\nFound {len(bams_to_process)} .bam file(s) to process:")
    for bam_file in bams_to_process:
        print(f"\t{bam_file}")

    validate_input_output(
        bams_to_process=bams_to_process,
        overwrite=overwrite,
    )

    #################################################
    # Operate over the input BAM files
    #################################################

    errors_list = []
    skip_count = 0
    for i, input_bam in enumerate(bams_to_process):
        time_bam = time.time()
        output_file = os.path.splitext(input_bam)[0] + ".methylation.npz"
        # Extract methylation data as a COO sparse matrix
        print("\n" + "=" * 80)
        print(f"Processing BAM file {i+1} of {len(bams_to_process)}")
        print(f"\nExtracting methylation data from: {input_bam}")

        if os.path.exists(output_file) and not overwrite:
            print(
                "\tOutput file already exists and --overwrite not specified. Skipping this .bam."
            )
            skip_count += 1
            continue

        try:
            methylation_data_coo = extract_methylation_data_from_bam(
                input_bam=input_bam,
                genome_methylation_embedding=genome_methylation_embedding,
                quality_limit=quality_limit,
                verbose=verbose,
                debug=debug,
            )
        except FileNotFoundError as e:  # Likely no .bam index file
            print(f"Error: {e}")
            errors_list.append(e)
            continue

        print(f"\nWriting methylation data to: {output_file}")

        # Save the matrix, which is an ndarray of shape (n_reads, n_cpgs), to a file
        scipy.sparse.save_npz(output_file, methylation_data_coo, compressed=True)

        # Report performance time
        print(f"\nTime for this bam: {time.time() - time_bam:.2f} seconds")
        print(f"\nTotal time elapsed: {time.time() - time_start:.2f} seconds")

    if errors_list:
        print(f"\n{len(errors_list)} errors occurred during processing:")
        for error in errors_list:
            print(f"\t{error}")

    print("\nRun complete.")
    print(f"\n{len(bams_to_process) - skip_count} .bam files were processed.")
    print(f"\t{skip_count} .bam files were skipped due to existing output files.")
    print(f"\t{len(errors_list)} .bam files had errors (missing index files?).")
    print(f"\nTotal time elapsed: {time.time() - time_start:.2f} seconds")


if __name__ == "__main__":
    main(prog_name="bam2tensor")  # pylint: disable=no-value-for-parameter
