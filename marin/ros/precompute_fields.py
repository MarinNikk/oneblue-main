# -*- coding: utf-8 -*-
"""Predizračunavanje (pre-compute): iz velikih NetCDF datoteka napravi mali cache/fields.npz.

Ovaj skript se pokreće JEDNOM iz projektnog .venv-a, jer samo ondje postoji xarray:
    ../../.venv/bin/python precompute_fields.py
Rezultat je predmemorija (npz) koju onda čita ROS čvor. Poanta: ROS 2 Jazzy vrti
sistemski Python gdje xarray nije instaliran, pa mu ostavljamo gotov npz koji se
učitava samo s numpy-em. Tako izbjegavamo miješanje dvaju Python okruženja.
"""
from pathlib import Path
import numpy as np
import xarray as xr

# Putanje relativno od ove datoteke (neovisno o tome odakle pokrećemo skript).
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UO = ROOT / "data/raw/adriatic_currents_uo_detided_2024_2026.nc"  # u: istočna komponenta brzine
VO = ROOT / "data/raw/adriatic_currents_vo_detided_2024_2026.nc"  # v: sjeverna komponenta brzine
STRIDE = 3   # prorjeđivanje mreže: uzmi svaku treću točku da cache bude manji i brži


def main():
    """Pročitaj u i v iz NetCDF-a, prorijedi mrežu i spremi sve u komprimirani npz.

    Ne prima ni vraća ništa; radi nuspojavu: zapisuje cache/fields.npz. U npz idu
    osi lat i lon, pune serije polja U i V (svi dani), te lista datuma (times).
    Sve spremamo kao float32 da datoteka bude manja (dovoljna preciznost za prikaz).
    """
    u = xr.open_dataset(UO)["uo_detided"]
    v = xr.open_dataset(VO)["vo_detided"]
    lat = u.latitude.values[::STRIDE].astype("float32")
    lon = u.longitude.values[::STRIDE].astype("float32")
    # [:, ::STRIDE, ::STRIDE]: sve vremenske korake, ali prorijeđeno po lat i lon.
    U = u.values[:, ::STRIDE, ::STRIDE].astype("float32")
    V = v.values[:, ::STRIDE, ::STRIDE].astype("float32")
    times = np.array([str(t)[:10] for t in u.time.values])  # datumi skraćeni na YYYY-MM-DD
    out = HERE / "cache"
    out.mkdir(exist_ok=True)                                 # napravi mapu cache ako ne postoji
    np.savez_compressed(out / "fields.npz", lat=lat, lon=lon, u=U, v=V, times=times)
    mb = (out / "fields.npz").stat().st_size / 1e6
    print(f"spremljeno {out/'fields.npz'} | polja {U.shape} | {mb:.1f} MB")


if __name__ == "__main__":
    main()
