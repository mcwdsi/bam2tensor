"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Bam2Tensor."""


if __name__ == "__main__":
    main(prog_name="bam2tensor")  # pragma: no cover
