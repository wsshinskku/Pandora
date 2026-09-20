"""Validated import of exported UE/cell SINR samples; no fabricated QuaDRiGa traces."""

import csv

import numpy as np


def import_sinr_csv(path, slots, users, cells, bandwidth_mhz):
    if min(slots, users, cells) < 1 or not np.isfinite(bandwidth_mhz) or bandwidth_mhz <= 0:
        raise ValueError("Positive trace dimensions and bandwidth are required")
    sinr = np.full((slots, users, cells), np.nan)
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"slot", "ue", "cell", "sinr_db"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires slot,ue,cell,sinr_db columns (zero-based indices)")
        for row in reader:
            key = tuple(int(row[k]) for k in ("slot", "ue", "cell"))
            if any(index < 0 or index >= limit for index, limit in zip(key, sinr.shape)):
                raise ValueError("Trace index outside configured dimensions")
            if np.isfinite(sinr[key]):
                raise ValueError(f"Duplicate SINR sample {key}")
            value = float(row["sinr_db"])
            if not np.isfinite(value) or abs(value) > 300:
                raise ValueError("SINR must be finite dB with magnitude <= 300")
            sinr[key] = value
    if not np.isfinite(sinr).all():
        raise ValueError("Trace is incomplete; one value per slot/UE/cell is required")
    return bandwidth_mhz * np.log2(1 + np.power(10.0, sinr / 10))
