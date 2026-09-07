# -*- coding: utf-8 -*-
"""Preuzimanje Copernicus mjesečne temperature i saliniteta za obje regije.

Drugi korak postupka: skidamo modelske (Copernicus) podatke s kojima ćemo
kasnije usporediti in-situ mjerenja. Koristimo proizvod MEDSEA reanalize, tj.
mjesečne prosjeke površine za cijelo Sredozemlje. Reanaliza je rekonstrukcija
prošlog stanja mora: model koji je naknadno usklađen s dostupnim mjerenjima.

Za svaku regiju (Cipar, Katalonija) i svaku varijablu (temperatura, salinitet)
skida se samo malo područje oko mjernih postaja, a rezultat se sprema u NetCDF
(.nc) datoteku u podmapu data/.

Prije prvog pokretanja treba se jednom prijaviti na Copernicus:
`../../.venv/bin/copernicusmarine login`.
"""
from pathlib import Path
import copernicusmarine as cm

from insitu import load_all, region_boxes

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

# Za svaku varijablu: (ID Copernicus skupa podataka, kratko ime varijable u datoteci).
# "thetao" je standardna oznaka za temperaturu mora, "so" za salinitet.
DATASETS = {
    "temp": ("cmems_mod_med_phy-temp_my_4.2km_P1M-m", "thetao"),
    "sal":  ("cmems_mod_med_phy-sal_my_4.2km_P1M-m", "so"),
}
START, END = "1994-01-01", "2022-12-31"   # razdoblje koje reanaliza pokriva
DEPTH_MIN, DEPTH_MAX = 0.0, 5.0           # skidamo samo površinski sloj (0-5 m)


def main():
    # Okviri područja dolaze iz stvarnih koordinata mjerenja (vidi insitu.py).
    # Tako skidamo samo ono što nam treba, a ne cijelo Sredozemlje.
    boxes = region_boxes(load_all())
    for region, (lon0, lon1, lat0, lat1) in boxes.items():
        for var, (dsid, short) in DATASETS.items():
            fname = f"cop_{region.lower()}_{var}.nc"
            print(f"\n>>> {region} / {var} -> {fname}")
            # cm.subset skine samo zadani "izrezak" (po prostoru, dubini i vremenu).
            # minimum/maximum_* definiraju granice, pa je datoteka mala i brzo se skine.
            cm.subset(
                dataset_id=dsid,
                variables=[short],
                minimum_longitude=lon0, maximum_longitude=lon1,
                minimum_latitude=lat0, maximum_latitude=lat1,
                minimum_depth=DEPTH_MIN, maximum_depth=DEPTH_MAX,
                start_datetime=START, end_datetime=END,
                output_directory=str(OUT), output_filename=fname,
                overwrite=True,
            )
    print("\nGotovo. Datoteke u:", OUT)


if __name__ == "__main__":
    main()
