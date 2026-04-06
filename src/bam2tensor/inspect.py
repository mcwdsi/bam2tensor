"""Inspect command for bam2tensor .npz output files.

Provides a CLI entry point (``bam2tensor-inspect``) that prints a summary
of one or more ``.methylation.npz`` files, including matrix dimensions,
sparsity, file size, and embedded provenance metadata.

Example:
    Inspect a single file::

        $ bam2tensor-inspect sample.methylation.npz

    Inspect multiple files::

        $ bam2tensor-inspect *.methylation.npz
"""

import os
import sys

import click
import numpy as np
import scipy.sparse

from bam2tensor.metadata import read_npz_metadata, read_npz_tlen


def _format_size(nbytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        nbytes: Number of bytes.

    Returns:
        A string such as ``"14.2 MB"`` or ``"832 bytes"``.

    Example:
        >>> _format_size(14_200_000)
        '13.5 MB'

        >>> _format_size(500)
        '500 bytes'

        >>> _format_size(2048)
        '2.0 KB'
    """
    if nbytes < 1024:
        return f"{nbytes} bytes"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.1f} MB"
    else:
        return f"{nbytes / (1024 * 1024 * 1024):.1f} GB"


def inspect_npz(npz_path: str) -> None:
    """Print a human-readable summary of a .methylation.npz file.

    Loads the sparse matrix and any embedded metadata, then prints
    matrix dimensions, data-point counts, sparsity, provenance
    information, and file size.

    Args:
        npz_path: Path to the ``.npz`` file to inspect.

    Example:
        >>> # xdoctest: +SKIP
        >>> inspect_npz("sample.methylation.npz")
        sample.methylation.npz
          Reads:           1,423
          CpG sites:       28,217,448
          ...
    """
    # Load matrix
    matrix = scipy.sparse.load_npz(npz_path)
    n_reads, n_cpgs = matrix.shape
    n_data = matrix.nnz
    total_cells = int(np.prod(matrix.shape)) if n_reads > 0 else 0
    sparsity = 1 - (n_data / total_cells) if total_cells > 0 else 0.0
    file_size = os.path.getsize(npz_path)

    # Load metadata (may be None for old files)
    meta = read_npz_metadata(npz_path)

    # Print summary
    print(os.path.basename(npz_path))

    if meta and "genome_name" in meta:
        print(f"  Genome:          {meta['genome_name']}")
    if meta and "expected_chromosomes" in meta:
        chroms = meta["expected_chromosomes"]
        n_chr = len(chroms)
        if n_chr <= 4:
            chrom_display = ", ".join(chroms)
        else:
            chrom_display = (
                f"{n_chr} ({chroms[0]}, {chroms[1]}, "
                f"... {chroms[-2]}, {chroms[-1]})"
            )
        print(f"  Chromosomes:     {chrom_display}")

    print(f"  Reads:           {n_reads:,}")
    print(f"  CpG sites:       {n_cpgs:,}")
    print(f"  Data points:     {n_data:,} (sparsity: {sparsity:.2%})")

    # TLEN / fragment length statistics
    tlen = read_npz_tlen(npz_path)
    if tlen is not None:
        nonzero = np.abs(tlen)[np.abs(tlen) > 0]
        if len(nonzero) > 0:
            print(
                f"  Fragment len:    median {np.median(nonzero):.0f}, "
                f"mean {np.mean(nonzero):.0f}, "
                f"range [{nonzero.min()}, {nonzero.max()}]"
            )
        else:
            print("  Fragment len:    all zero (single-end data)")

    if meta and "cpg_index_crc32" in meta:
        print(f"  CpG index CRC32: {meta['cpg_index_crc32']}")
    if meta and "bam2tensor_version" in meta:
        print(f"  bam2tensor:      v{meta['bam2tensor_version']}")
    elif meta is None:
        print("  Metadata:        none (produced by older bam2tensor)")

    print(f"  File size:       {_format_size(file_size)}")


@click.command(help="Inspect bam2tensor .methylation.npz output files.")
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
def main(files: tuple[str, ...]) -> None:
    """Inspect one or more .methylation.npz files.

    Prints a summary of each file including matrix dimensions, sparsity,
    embedded metadata, and file size.

    Args:
        files: One or more paths to ``.methylation.npz`` files.
    """
    for i, path in enumerate(files):
        if i > 0:
            print()
        try:
            inspect_npz(path)
        except Exception as e:
            print(f"{os.path.basename(path)}", file=sys.stderr)
            print(f"  Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
