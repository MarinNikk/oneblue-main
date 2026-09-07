# -*- coding: utf-8 -*-
"""Jezgra za učitavanje polja morskih struja i pretvorbu u strukture spremne za ROS.

Ova datoteka namjerno NE uvozi ROS (rclpy). Razlog: ROS 2 Jazzy radi na sistemskom
Pythonu 3.12, a projektni .venv je Python 3.13, pa se ti svjetovi ne miješaju. Zato
je sva logika (čitanje podataka, grid, vektori) ovdje odvojena i koristi samo numpy.
current_publisher.py je onda tanak ROS sloj koji zove ove funkcije.

Podatke dohvaćamo na dva načina: ako postoji predizračunata predmemorija (cache)
cache/fields.npz, čitamo nju (brzo, treba samo numpy); inače padamo na izravno
čitanje NetCDF datoteka preko xarray-a (to radi samo iz .venv-a gdje xarray postoji).
Predmemoriju izrađuje precompute_fields.py.
"""
from pathlib import Path
import numpy as np

# Putanje računamo relativno od ove datoteke, da kod radi neovisno o tome odakle se pokreće.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE / "cache" / "fields.npz"               # predizračunati npz (brzi put, samo numpy)
UO = ROOT / "data/raw/adriatic_currents_uo_detided_2024_2026.nc"  # istočna komponenta brzine (u)
VO = ROOT / "data/raw/adriatic_currents_vo_detided_2024_2026.nc"  # sjeverna komponenta brzine (v)
DEG_M = 111_320.0     # približno metara po stupnju geografske širine (samo za referencu)
# Veličina jedne ćelije grida u RViz-u. Držimo 1.0 namjerno: kad bi ćelija bila stvarnih
# ~12 km, karta bi u RViz-u ispala ogromna i nepregledna. S 1.0 je 1 ćelija = 1 jedinica.
DISPLAY_SCALE = 1.0


def _from_cache(t_index):
    """Učitaj jedno vremensko polje iz predmemorije cache/fields.npz.

    t_index je indeks dana u nizu (npr. -1 = zadnji dostupni dan). Vraća rječnik
    s lat, lon, u, v za taj dan, datumom i ukupnim brojem dostupnih polja (n_times).
    Koristimo samo numpy, pa ovo radi i pod sistemskim Pythonom gdje nema xarray-a.
    """
    z = np.load(CACHE, allow_pickle=False)
    return dict(lat=z["lat"].astype(float), lon=z["lon"].astype(float),
                u=z["u"][t_index].astype(float), v=z["v"][t_index].astype(float),
                time=str(z["times"][t_index]), n_times=int(z["u"].shape[0]))


def _from_netcdf(t_index, stride):
    """Rezervni put: učitaj isto polje izravno iz NetCDF datoteka preko xarray-a.

    Koristi se samo kad predmemorije nema. xarray uvozimo unutar funkcije da cijeli
    modul ne ovisi o njemu (pod ROS-om ga ionako nema). stride prorjeđuje mrežu, npr.
    stride=3 uzima svaku treću točku da polje bude manje i brže za crtanje.
    """
    import xarray as xr
    u = xr.open_dataset(UO)["uo_detided"]
    v = xr.open_dataset(VO)["vo_detided"]
    return dict(lat=u.latitude.values[::stride].astype(float),
                lon=u.longitude.values[::stride].astype(float),
                u=u.values[t_index, ::stride, ::stride].astype(float),
                v=v.values[t_index, ::stride, ::stride].astype(float),
                time=str(u.time.values[t_index])[:10], n_times=int(u.shape[0]))


def load_field(t_index=-1, stride=3):
    """Vrati jedno dnevno polje struja kao rječnik, spremno za daljnju obradu.

    t_index: koji dan (zadano -1, tj. zadnji). stride: prorjeđivanje kad se čita iz NetCDF-a.
    Vraća rječnik: lat (uvijek rastuće, od juga prema sjeveru), lon, u i v komponente
    brzine (na kopnu su NaN jer ondje nema mjerenja), te datum polja.
    Sam bira izvor: predmemorija ako postoji, inače NetCDF.
    """
    d = _from_cache(t_index) if CACHE.exists() else _from_netcdf(t_index, stride)
    # Os latitude mora ići rastuće (jug -> sjever) da grid i strelice budu pravilno
    # orijentirani u RViz-u. Ako je zapisana obrnuto, okrećemo i lat i pripadne podatke.
    if d["lat"][0] > d["lat"][-1]:
        d["lat"], d["u"], d["v"] = d["lat"][::-1], d["u"][::-1], d["v"][::-1]
    return d


def grid_geometry(lat, lon):
    """Izračunaj geometriju za nav_msgs/OccupancyGrid (širina, visina, rezolucija, ishodište).

    OccupancyGrid je ROS poruka koja predstavlja 2D rešetku (kao piksele toplinske karte).
    width = broj stupaca (po longitudi), height = broj redaka (po latitudi), resolution =
    veličina ćelije (koristimo DISPLAY_SCALE radi preglednosti u RViz-u, vidi gore).
    """
    return dict(resolution=float(DISPLAY_SCALE),
                width=int(len(lon)), height=int(len(lat)),
                origin_lon=float(lon.min()), origin_lat=float(lat.min()))


def speed_to_occupancy(u, v, vmax=None):
    """Pretvori komponente brzine (u, v) u vrijednosti toplinske karte za OccupancyGrid.

    OccupancyGrid očekuje cijele brojeve; vrijednosti 0 i -1 imaju posebno značenje
    (slobodno / nepoznato), pa korisni raspon za "intenzitet" skaliramo u 0..100.
    Ovdje: iznos brzine (hypot(u, v)) dijelimo s vmax i množimo sa 100. vmax je zadano
    99. percentil brzine (da rijetki ekstremi ne preuzmu cijelu skalu). Kopno (NaN)
    označavamo s -1. Vraća (occ polje int8, iskorišteni vmax).
    Raspored je row-major (redak po redak) počevši od juga, kako grid očekuje.
    """
    speed = np.hypot(u, v)                          # iznos brzine iz istočne i sjeverne komponente
    vmax = float(vmax or np.nanpercentile(speed, 99))
    occ = np.clip(speed / vmax * 100.0, 0, 100)     # skaliraj u 0..100 i odreži na granice
    occ = np.where(np.isnan(speed), -1, occ).astype(np.int8)  # kopno -> -1 (nepoznato)
    return occ, vmax


def current_vectors(lat, lon, u, v, subsample=2):
    """Vrati listu strelica (x, y, yaw, brzina) za geometry_msgs/PoseArray.

    PoseArray je ROS poruka, niz položaja i orijentacija, koji RViz crta kao strelice.
    Svaku strelicu stavljamo u središte morske ćelije: zato (j + 0.5) i (i + 0.5), jer
    cijeli broj j/i je rub ćelije, a +0.5 je sredina. Tako se strelice lijepo poklope
    s toplinskom kartom iz OccupancyGrid-a. Koristimo iste jedinice (DISPLAY_SCALE po ćeliji).
    yaw je smjer struje izračunat iz arctan2(v, u): u je prema istoku, v prema sjeveru.
    subsample prorjeđuje strelice (npr. svaka druga ćelija) da ih ne bude previše.
    """
    out = []
    for i in range(0, len(lat), subsample):
        for j in range(0, len(lon), subsample):
            uu, vv = u[i, j], v[i, j]
            if np.isnan(uu) or np.isnan(vv):        # preskoči kopno, ondje nema struje
                continue
            x = (j + 0.5) * DISPLAY_SCALE           # +0.5: središte ćelije, ne rub
            y = (i + 0.5) * DISPLAY_SCALE
            out.append((float(x), float(y), float(np.arctan2(vv, uu)),
                        float(np.hypot(uu, vv))))
    return out


if __name__ == "__main__":
    # Brza provjera bez ROS-a: učitaj polje i ispiši osnovne brojke (koji izvor,
    # koliko dana, veličina grida, koliko morskih ćelija i strelica). Korisno za
    # ocjenu radi li jezgra ispravno prije nego se pokrene ROS čvor.
    f = load_field()
    g = grid_geometry(f["lat"], f["lon"])
    occ, vmax = speed_to_occupancy(f["u"], f["v"])
    vecs = current_vectors(f["lat"], f["lon"], f["u"], f["v"])
    print(f"izvor: {'cache' if CACHE.exists() else 'NetCDF'} | dostupno polja: {f['n_times']}")
    print(f"datum polja: {f['time']}")
    print(f"grid: {g['width']}x{g['height']} | rezolucija {g['resolution']:.0f} m")
    print(f"morskih ćelija: {int((occ>=0).sum())} od {occ.size} | vmax {vmax:.3f} m/s")
    print(f"OccupancyGrid: min {occ[occ>=0].min()} max {occ.max()} (kopno -1) | "
          f"PoseArray strelica: {len(vecs)}")
