"""Utility helpers for tabular data operations."""

from __future__ import annotations

import hashlib

import polars as pl


def add_cell_id_hash(
    profiles: pl.DataFrame, seed: int = 0, force: bool = False
) -> pl.DataFrame:
    """Add deterministic hash IDs in ``Metadata_cell_id`` as the first column.

    Parameters
    ----------
    profiles : pl.DataFrame
        Input profiles table.
    seed : int, optional
        Seed mixed into each row signature for reproducible alternative hashes.
    force : bool, optional
        If ``False`` and ``Metadata_cell_id`` already exists, the input is returned unchanged.

    Returns
    -------
    pl.DataFrame
        DataFrame including ``Metadata_cell_id`` as the first column.
    """
    if not isinstance(profiles, pl.DataFrame):
        raise TypeError("profiles must be a Polars DataFrame")

    if "Metadata_cell_id" in profiles.columns and not force:
        return profiles

    work_df = (
        profiles.drop("Metadata_cell_id")
        if "Metadata_cell_id" in profiles.columns
        else profiles
    )

    def _row_hash(row: dict[str, object]) -> str:
        row_items = "|".join(f"{key}={row[key]}" for key in work_df.columns)
        payload = f"{seed}|{row_items}".encode()
        return hashlib.md5(payload).hexdigest()  # nosec B324

    hashes = [_row_hash(row) for row in work_df.iter_rows(named=True)]

    return work_df.with_columns(pl.Series("Metadata_cell_id", hashes)).select(
        ["Metadata_cell_id", *work_df.columns]
    )
