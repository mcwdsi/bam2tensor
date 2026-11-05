"""Metadata utilities for bam2tensor .npz output files.

This module provides functions to embed and retrieve provenance metadata
inside the ``.methylation.npz`` files produced by bam2tensor.  The metadata
is stored as a ``metadata.json`` entry appended to the ZIP archive that
underlies every ``.npz`` file.  ``scipy.sparse.load_npz`` silently ignores
this extra entry, so existing downstream code is unaffected.

Output format versions:
    The ``format_version`` metadata entry identifies how the matrix in a
    file is laid out, independently of the bam2tensor release that wrote it
    (see :data:`OUTPUT_FORMAT_VERSION`):

    - **2** — One row per sequenced DNA fragment (``"row_unit":
      "fragment"``). The two mates of a read pair are combined into a
      single row, and every read is decoded on the base that carries its
      call, so both strands of every aligner dialect are represented.
      ``tlen`` holds the fragment length (absolute BAM TLEN).
    - **1** — Files with no ``format_version`` entry (releases up to and
      including 2.8). One row per read. For Bismark BAMs both mates were
      separate rows and calls from bottom-strand alignments were recorded
      as ``-1``; for Biscuit, bwameth and gem3 BAMs only read 1 of each
      pair was used. ``tlen`` holds the signed BAM TLEN of each read.

Example:
    Writing metadata (done automatically by the CLI)::

        >>> # xdoctest: +SKIP
        >>> from bam2tensor.metadata import write_npz_metadata, read_npz_metadata
        >>> write_npz_metadata("sample.methylation.npz", {
        ...     "bam2tensor_version": "2.2",
        ...     "genome_name": "hg38",
        ... })
        >>> read_npz_metadata("sample.methylation.npz")
        {'bam2tensor_version': '2.2', 'genome_name': 'hg38'}

    Reading metadata from an existing file::

        >>> # xdoctest: +SKIP
        >>> meta = read_npz_metadata("sample.methylation.npz")
        >>> if meta is not None:
        ...     print(meta["genome_name"])
        hg38
"""

import hashlib
import io
import json
import zipfile
import zlib
from typing import TYPE_CHECKING

import numpy as np

# Layout version of the matrix written to ``.methylation.npz`` files, stored
# as ``format_version`` in the embedded metadata. Increment it whenever the
# meaning of rows, columns, values or ``tlen`` changes; the module docstring
# lists what each version means.
OUTPUT_FORMAT_VERSION = 2

# What a matrix row represents in the current format (``row_unit``).
OUTPUT_ROW_UNIT = "fragment"

if TYPE_CHECKING:
    # Avoid a runtime circular import: embedding.py imports compute_fasta_sha256
    # from this module, and this module only needs the embedding type for
    # annotations.
    from bam2tensor.embedding import GenomeMethylationEmbedding


def compute_fasta_sha256(fasta_source: str) -> str:
    """Compute the SHA-256 of a FASTA file's bytes on disk.

    Used to stamp the CpG-site cache (see
    :py:class:`bam2tensor.embedding.GenomeMethylationEmbedding`) so a
    cache can be rejected when the underlying FASTA changes (e.g. a
    user swaps a soft-masked build for a hard-masked one).

    Args:
        fasta_source: Path to the reference FASTA file.

    Returns:
        The hex-encoded SHA-256 digest of the file's bytes.
    """
    h = hashlib.sha256()
    with open(fasta_source, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_cpg_index_crc32(embedding: "GenomeMethylationEmbedding") -> str:
    """Compute a CRC32 checksum over the CpG site positions in an embedding.

    The checksum captures the exact column mapping of the sparse matrix:
    which chromosomes are included, in what order, and which genomic
    positions are CpG sites within each chromosome.  Two embeddings with
    the same checksum will produce identical column semantics.

    The checksum is defined over the text ``chrom\tpos1,pos2,...``, one
    line per chromosome in ``expected_chromosomes`` order, joined by
    newlines. It is computed incrementally, one chromosome at a time, so
    the whole-genome text (hundreds of megabytes for a human reference)
    is never held in memory at once.

    Args:
        embedding: A fully initialised GenomeMethylationEmbedding whose
            ``cpg_sites_dict`` is populated.

    Returns:
        The CRC32 checksum as an 8-character lowercase hexadecimal string.

    Example:
        >>> # xdoctest: +SKIP
        >>> from bam2tensor.embedding import GenomeMethylationEmbedding
        >>> emb = GenomeMethylationEmbedding(
        ...     genome_name="hg38",
        ...     expected_chromosomes=["chr1"],
        ...     fasta_source="ref.fa",
        ... )
        >>> compute_cpg_index_crc32(emb)
        'a1b2c3d4'
    """
    # Deterministic byte representation:
    #   chrom\tpos1,pos2,...  (one line per chromosome, in order, joined by \n)
    # zlib.crc32 accepts a running value, so feeding it line by line gives
    # exactly the checksum of the concatenated text.
    crc = 0
    for index, chrom in enumerate(embedding.expected_chromosomes):
        positions = embedding.cpg_sites_dict.get(chrom, [])
        line = chrom + "\t" + ",".join(map(str, positions))
        if index:
            crc = zlib.crc32(b"\n", crc)
        crc = zlib.crc32(line.encode("utf-8"), crc)
    return format(crc & 0xFFFFFFFF, "08x")


def write_npz_metadata(
    npz_path: str,
    metadata: dict,
) -> None:
    """Append a ``metadata.json`` entry to an existing ``.npz`` file.

    The metadata is serialised as compact JSON and appended to the ZIP
    archive.  ``scipy.sparse.load_npz`` ignores unrecognised entries, so
    the file remains fully compatible with existing code.

    Args:
        npz_path: Path to the ``.npz`` file (must already exist).
        metadata: A JSON-serialisable dictionary of metadata to embed.

    Example:
        >>> # xdoctest: +SKIP
        >>> write_npz_metadata("out.npz", {"genome_name": "hg38"})
    """
    with zipfile.ZipFile(npz_path, "a") as zf:
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))


def read_npz_metadata(npz_path: str) -> dict | None:
    """Read the ``metadata.json`` entry from a ``.npz`` file.

    Args:
        npz_path: Path to the ``.npz`` file.

    Returns:
        The metadata dictionary, or ``None`` if the file does not contain
        a ``metadata.json`` entry (e.g. files produced by older versions).

    Example:
        >>> # xdoctest: +SKIP
        >>> meta = read_npz_metadata("sample.methylation.npz")
        >>> meta["genome_name"]
        'hg38'
    """
    with zipfile.ZipFile(npz_path, "r") as zf:
        if "metadata.json" in zf.namelist():
            return json.loads(zf.read("metadata.json"))
    return None


def write_npz_tlen(npz_path: str, tlen: np.ndarray) -> None:
    """Append a ``tlen.npy`` entry to an existing ``.npz`` file.

    The array is serialised with :func:`numpy.save` and appended to the
    ZIP archive.  ``scipy.sparse.load_npz`` ignores this extra entry, so
    the file remains compatible with existing sparse-matrix code.

    Args:
        npz_path: Path to the ``.npz`` file (must already exist).
        tlen: A 1-D numpy array of template lengths, one per matrix row.

    Example:
        >>> # xdoctest: +SKIP
        >>> write_npz_tlen("out.npz", np.array([300, -300, 0], dtype=np.int32))
    """
    buf = io.BytesIO()
    np.save(buf, tlen)
    with zipfile.ZipFile(npz_path, "a") as zf:
        zf.writestr("tlen.npy", buf.getvalue())


def read_npz_tlen(npz_path: str) -> np.ndarray | None:
    """Read the ``tlen.npy`` entry from a ``.npz`` file.

    Args:
        npz_path: Path to the ``.npz`` file.

    Returns:
        The template-length array (one entry per matrix row), or ``None``
        if the file does not contain a ``tlen.npy`` entry (e.g. files
        produced by older versions). In format version 2 files the values
        are fragment lengths (non-negative); in earlier files they are
        the signed BAM TLEN of each read.

    Example:
        >>> # xdoctest: +SKIP
        >>> tlen = read_npz_tlen("sample.methylation.npz")
        >>> if tlen is not None:
        ...     print(f"Median fragment length: {np.median(np.abs(tlen)):.0f}")
    """
    with zipfile.ZipFile(npz_path, "r") as zf:
        if "tlen.npy" in zf.namelist():
            buf = io.BytesIO(zf.read("tlen.npy"))
            return np.load(buf, allow_pickle=False)
    return None
