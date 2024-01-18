# Import modules
import click
import os
import time

# Third party modules
import scipy.sparse

from bam2tensor.embedding import GenomeMethylationEmbedding

from bam2tensor.functions import (
    extract_methylation_data_from_bam,
)


def get_input_bams(input_path: str) -> list:
    """Determine if the input is a path or file, and return a list of .bam files to process.

    Args
    ----------
    input_path (str): Input path or file.

    Returns
    ----------
    list: List of .bam files to process.

    Raises
    ----------
    ValueError: If input_path is not a file or a directory.
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
    """Validate the input and output files."""

    # Check input/output validity
    for bam_file in bams_to_process:
        if not os.access(bam_file, os.R_OK):
            raise ValueError(f"Input file is not readable: {bam_file}")

        output_file = os.path.splitext(bam_file)[0] + ".methylation.npz"
        # Check if the output files exist or are writable
        if os.path.exists(output_file):
            if overwrite:
                print(
                    "\t\tOutput file exists and --overwrite specified. Will overwrite existing .methylation.npz file."
                )
                if not os.access(output_file, os.W_OK):
                    raise ValueError(f"Output file is not writable: {output_file}")
            else:
                raise ValueError(
                    f"Output file exists and --overwrite not specified: {output_file}"
                )
        # Otherwise, check the path is writable
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
)
@click.option(
    "--genome-name",
    help="Genome name (e.g. hg38, mm39, etc.)",
    required=True,
)
@click.option(
    "--reference-fasta",
    help="Reference genome fasta file (critical to determine CpG sites).",
    required=True,
)
@click.option(
    "--quality-limit",
    help="Quality filter for aligned reads (default = 20)",
    default=20,
    type=int,
)
@click.option(
    "--window-size", help="Window size (default = 150)", default=150, type=int
)
@click.option("--verbose", help="Verbose output.", is_flag=True)
@click.option("--skip-cache", help="De-novo generate CpG sites (slow).", is_flag=True)
@click.option(
    "--debug",
    help="Debug mode (extensive validity checking + debug messages).",
    is_flag=True,
)
@click.option("--overwrite", help="Overwrite output file if it exists.", is_flag=True)
def main(
    input_path: str,
    genome_name: str,
    reference_fasta: str,
    quality_limit: int,
    window_size: int,
    verbose: bool,
    skip_cache: bool,
    debug: bool,
    overwrite: bool,
) -> None:
    """Bam2Tensor."""
    time_start = time.time()
    # Print run information
    print(f"Reference fasta: {reference_fasta}")
    print(f"Input path: {input_path}")

    print(f"\nLoading (or generating) methylation embedding named: {reference_fasta}")

    ## Globals
    HG38_CHROMOSOMES = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
    # MM39_CHROMOSOMES = ["chr" + str(i) for i in range(1, 20)] + ["chrX", "chrY"]
    # TEST_CHROMOSOMES = ["chr1"]

    # dict: {chromosome: index}, e.g. {"chr1": 0, "chr2": 1, ...}
    # CHROMOSOMES_DICT = {ch: idx for idx, ch in enumerate(CHROMOSOMES)}

    # Create (or load) a GenomeMethylationEmbedding object
    genome_methylation_embedding = GenomeMethylationEmbedding(
        genome_name=genome_name,
        expected_chromosomes=HG38_CHROMOSOMES,
        fasta_source=reference_fasta,
        window_size=window_size,
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

    for i, input_bam in enumerate(bams_to_process):
        time_bam = time.time()
        output_file = os.path.splitext(input_bam)[0] + ".methylation.npz"
        # Extract methylation data as a COO sparse matrix
        print("\n" + "=" * 80)
        print(f"Processing BAM file {i+1} of {len(bams_to_process)}")
        print(f"\nExtracting methylation data from: {input_bam}")

        methylation_data_coo = extract_methylation_data_from_bam(
            input_bam=input_bam,
            genome_methylation_embedding=genome_methylation_embedding,
            quality_limit=quality_limit,
            verbose=verbose,
            debug=debug,
        )

        print(f"\nWriting methylation data to: {output_file}")

        # Save the matrix, which is an ndarray of shape (n_reads, n_cpgs), to a file
        scipy.sparse.save_npz(output_file, methylation_data_coo, compressed=True)

        # Report performance time
        print(f"\nTime for this bam: {time.time() - time_bam:.2f} seconds")
        print(f"\nTotal time elapsed: {time.time() - time_start:.2f} seconds")

    print("\nRun complete.")


if __name__ == "__main__":
    main(prog_name="bam2tensor")  # pylint: disable=no-value-for-parameter
