"""Command-line interface."""
import click


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
# @click.option("--output-file", help="Output file. Default is to use the extension .methylation.npz. (Only available for single .bam file input.)", default=None)
@click.option("--overwrite", help="Overwrite output file if it exists.", is_flag=True)
def main() -> None:
    """Bam2Tensor."""


if __name__ == "__main__":
    main(prog_name="bam2tensor")  # pragma: no cover
