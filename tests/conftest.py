"""Shared pytest configuration for the bam2tensor test suite."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_working_directory(tmp_path, monkeypatch):
    """Run every test in its own empty working directory.

    bam2tensor writes CpG index caches (``<genome>.cache.json.gz``) into the
    current working directory, and by default writes its output next to the
    input BAM. Running each test from a private temporary directory keeps
    those files out of the repository checkout and stops tests from seeing
    each other's caches.

    Tests must therefore refer to bundled fixtures by absolute path; use the
    constants in ``tests/paths.py``.
    """
    monkeypatch.chdir(tmp_path)
