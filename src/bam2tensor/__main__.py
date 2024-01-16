# Import modules
import click
import os
import sys
import time

# Third party modules
import scipy.sparse

from bam2tensor.embedding import GenomeMethylationEmbedding

from bam2tensor.functions import (
    extract_methylation_data_from_bam,
    CHROMOSOMES,
)


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
    if verbose:
        print(f"Reference fasta: {reference_fasta}")
        print(f"Input path: {input_path}")

    # Check if input_path is a file or a directory
    if os.path.isfile(input_path):
        bams_to_process = [input_path]
    elif os.path.isdir(input_path):
        # Recursively find all .bam files in this path, and add them to a list
        bams_to_process = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".bam"):
                    bams_to_process.append(os.path.join(root, file))
    else:
        raise ValueError(f"Input path {input_path} is not a file or a directory.")

    if verbose:
        print(f"\nFound {len(bams_to_process)} .bam file(s) to process:")
        for bam_file in bams_to_process:
            print(f"\t{bam_file}")

    # Check input/output validity
    for bam_file in bams_to_process:
        assert os.access(bam_file, os.R_OK), f"Input file is not readable: {bam_file}"

        output_file = os.path.splitext(bam_file)[0] + ".methylation.npz"
        # Check if the output files exist or are writable
        if os.path.exists(output_file):
            if overwrite:
                if verbose:
                    print(
                        f"\tOutput file exists and --overwrite specified. Will overwrite: {output_file}"
                    )
                assert os.access(
                    output_file, os.W_OK
                ), f"Output file is not writable: {output_file}"
            else:
                print(
                    f"Exiting. An output file exists and --overwrite not specified: {output_file}"
                )
                sys.exit(1)
        # Otherwise, check the path is writable
        else:
            assert os.access(
                os.path.dirname(os.path.abspath(output_file)), os.W_OK
            ), f"Output file path is not writable: {output_file}"

    # Create (or load) a GenomeMethylationEmbedding object
    genome_methylation_embedding = GenomeMethylationEmbedding(
        genome_name="hg38",
        expected_chromosomes=CHROMOSOMES,
        fasta_source=reference_fasta,
    )

    genome_methylation_embedding.load_cpg_sites()

    assert genome_methylation_embedding.embedding_loaded

    print(genome_methylation_embedding)

    if verbose:
        print(f"\nTime elapsed: {time.time() - time_start:.2f} seconds")

    return
    #################################################
    # Operate over the input BAM files
    #################################################

    for i, input_bam in enumerate(bams_to_process):
        time_bam = time.time()
        output_file = os.path.splitext(input_bam)[0] + ".methylation.npz"
        # Extract methylation data as a COO sparse matrix
        if verbose:
            print("\n" + "=" * 80)
            print(f"Processing BAM file {i+1} of {len(bams_to_process)}")
            print(f"\nExtracting methylation data from: {input_bam}")

        methylation_data_coo = extract_methylation_data_from_bam(
            input_bam=input_bam,
            total_cpg_sites=total_cpg_sites,
            chr_to_cpg_to_embedding_dict=chr_to_cpg_to_embedding_dict,
            cpgs_per_chr_cumsum=cpgs_per_chr_cumsum,
            windowed_cpg_sites_dict=windowed_cpg_sites_dict,
            windowed_cpg_sites_dict_reverse=windowed_cpg_sites_dict_reverse,
            quality_limit=quality_limit,
            verbose=verbose,
            debug=debug,
        )  # TODO: simplify these inputs!

        # Return validity test (TODO: move to the function itself?)
        assert methylation_data_coo.shape[1] == total_cpg_sites

        if verbose:
            print(f"\nWriting methylation data to: {output_file}")

        # Save the matrix, which is an ndarray of shape (n_reads, n_cpgs), to a file
        scipy.sparse.save_npz(output_file, methylation_data_coo, compressed=True)

        # Report performance time
        if verbose:
            print(f"\nTime for this bam: {time.time() - time_bam:.2f} seconds")
            print(f"\nTotal time elapsed: {time.time() - time_start:.2f} seconds")

    if verbose:
        print("\nRun complete.")


if __name__ == "__main__":
    main(prog_name="bam2tensor")  # pragma: no cover
