# -*- coding: utf-8 -*-
"""Učitavanje in-situ mjerenja (Cipar + Katalonija) u zajednički oblik.

Ovo je prvi korak usporedbe modelskih i izmjerenih podataka. Cilj cijelog
postupka je provjeriti koliko su Copernicus modelski podaci pouzdani u odnosu na
neovisna obalna mjerenja (tzv. in-situ mjerenja, dakle mjerenja izvršena izravno
na terenu).

Važna napomena: dvije Excel datoteke koje ovdje učitavamo NISU mjerenja struja
i NISU s Jadrana. Jedna je Cipar (kvaliteta obalne vode), druga je Katalonska
obala. Zato struje ne možemo usporediti, nego samo dvije varijable koje su
zajedničke s Copernicusom: temperaturu i salinitet (slanost mora).

Funkcije ovdje svaki Excel očiste i svedu na jedan zajednički "dugi" oblik, tj.
jedan redak po mjerenju sa stupcima:
region, station, lat, lon, date, depth, var ('temp' ili 'sal'), value.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from pyproj import Transformer

warnings.filterwarnings("ignore")
# ROOT je korijen cijelog projekta: ova datoteka je dva nivoa duboko
# (marin/copernicus_vs_insitu/), pa parents[2] vraća glavnu mapu projekta.
ROOT = Path(__file__).resolve().parents[2]
DFMR = ROOT / "data/raw/DFMR measurements.xlsx"          # ciparska mjerenja
CAT = ROOT / "data/raw/20260213_Annex Catalan Coast.xlsx"  # katalonska mjerenja

# Zanima nas samo površina mora, jer ćemo kasnije usporediti s površinskim slojem
# Copernicus modela. Zato zadržavamo samo uzorke plići ili jednaki ovoj dubini (u metrima).
SURFACE_MAX_DEPTH = 3.0


def load_cyprus():
    """Učitaj i očisti ciparska DFMR mjerenja (temperatura i salinitet).

    Ulaz: Excel datoteka s dva lista, "Measurements" (same izmjerene vrijednosti)
    i "Stations" (koordinate postaja). Ta dva lista spajamo preko ID-a postaje.
    Izlaz: DataFrame u dugom obliku sa stupcima station, lat, lon, date, depth,
    value, var ('temp' ili 'sal') i region ('Cipar').
    """
    m = pd.read_excel(DFMR, sheet_name="Measurements")
    st = pd.read_excel(DFMR, sheet_name="Stations")[["StationId", "Latitude", "Longitude"]]
    # Mjerenja nemaju koordinate u sebi, nego samo ID postaje. Zato ih spajamo
    # (merge) s listom postaja da svako mjerenje dobije svoj lat/lon.
    m = m.merge(st, left_on="Station Id", right_on="StationId", how="left")
    m["date"] = pd.to_datetime(m["Sampling Date"], errors="coerce")
    # Preimenujemo stupce u kratka, zajednička imena kakva koristi cijeli projekt.
    m = m.rename(columns={"Latitude": "lat", "Longitude": "lon",
                          "Sampling Depth": "depth", "Station Id": "station"})
    out = []
    # Temperatura i salinitet su u zasebnim stupcima. Prolazimo kroz oba i svaki
    # pretvaramo u isti oblik (stupac "value" + oznaka "var"), pa ih spojimo.
    # Tako dobivamo dugi format: jedan redak = jedno mjerenje jedne varijable.
    for var, col in [("temp", "Temperature (°C)"), ("sal", "Salinity (0/00) (psu)")]:
        s = m[["station", "lat", "lon", "date", "depth", col]].dropna(subset=[col, "date", "lat"])
        s = s.rename(columns={col: "value"})
        s["var"], s["region"] = var, "Cipar"
        out.append(s)
    df = pd.concat(out, ignore_index=True)
    # Sigurnosni filter: Cipar je otprilike na 20-40 stupnjeva geo. širine i dužine.
    # Time izbacujemo eventualne pogrešno upisane koordinate (npr. nule ili greške).
    return df[(df["lat"].between(20, 40)) & (df["lon"].between(20, 40))]


def load_catalan():
    """Učitaj i očisti katalonska obalna mjerenja (temperatura i salinitet).

    Posebnost ove datoteke: koordinate NISU u lat/lon, nego u projekcijskom
    sustavu UTM ETRS89 zona 31N (metri x i y na karti). Zato ih prvo moramo
    pretvoriti u geografsku širinu i dužinu (lat/lon), jer jedino tako možemo
    te točke povezati s Copernicus gridom. Pretvorbu radi biblioteka pyproj.
    Izlaz je isti dugi oblik kao kod load_cyprus(), s region = 'Katalonija'.
    """
    c = pd.read_excel(CAT, sheet_name="Dades de camp")
    # Transformer pretvara koordinate iz jednog sustava u drugi:
    # EPSG:25831 = UTM ETRS89 zona 31N (ulazni, u metrima),
    # EPSG:4326 = obični lat/lon u stupnjevima (izlazni, WGS84).
    # always_xy=True znači da je redoslijed (x=istok, y=sjever) odnosno (lon, lat).
    tr = Transformer.from_crs("EPSG:25831", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(c["UTM X"].values, c["UTM Y"].values)
    c["lon"], c["lat"] = lon, lat
    c["date"] = pd.to_datetime(c["dia"], errors="coerce")
    # Stupci su na katalonskom, pa ih preimenujemo u zajednička imena projekta.
    c = c.rename(columns={"Estació": "station", "Profunditat": "depth",
                          "Valor": "value", "Descripció Variable": "descr"})
    # Datoteka sadrži više varijabli. Zadržavamo samo temperaturu i salinitet
    # (to su jedine zajedničke s Copernicusom); ostale varijable ostanu bez oznake.
    vmap = {"Temperatura aigua (mostra)": "temp", "Salinitat (mostra)": "sal"}
    c["var"] = c["descr"].map(vmap)
    # dropna na "var" izbacuje sve retke koji nisu temperatura ni salinitet.
    c = c.dropna(subset=["var", "value", "date", "lat"])
    c["region"] = "Katalonija"
    df = c[["region", "station", "lat", "lon", "date", "depth", "var", "value"]]
    # Sigurnosni filter na katalonsko područje (slično kao kod Cipra).
    return df[(df["lat"].between(38, 44)) & (df["lon"].between(0, 5))]


def load_all(surface_only=True):
    """Spoji obje regije u jedan skup i izbaci loše/nerealne vrijednosti.

    Parametar surface_only (True po zadanom): ako je True, zadržavamo samo
    površinske uzorke (dubina <= SURFACE_MAX_DEPTH), jer ćemo ih uspoređivati
    s površinskim slojem modela. Vraća očišćeni DataFrame u dugom obliku.
    """
    df = pd.concat([load_cyprus(), load_catalan()], ignore_index=True)
    # Dubina i vrijednost ponekad dođu kao tekst iz Excela; pretvaramo ih u brojeve.
    # Ako dubina nedostaje, pretpostavljamo površinu (0.0 m).
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce").fillna(0.0)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    # Izbacujemo fizikalno nemoguće vrijednosti (greške u unosu, senzoru ili jedinicama).
    # Salinitet mora smislenog raspona je otprilike 20-42 psu.
    df = df[~((df["var"] == "sal") & ((df["value"] < 20) | (df["value"] > 42)))]
    # Temperatura površine Sredozemlja realno ide otprilike 5-35 stupnjeva C.
    df = df[~((df["var"] == "temp") & ((df["value"] < 5) | (df["value"] > 35)))]
    if surface_only:
        df = df[df["depth"] <= SURFACE_MAX_DEPTH]
    return df.reset_index(drop=True)


def region_boxes(df, pad=0.15):
    """Izračunaj pravokutni okvir (bounding box) oko mjerenja svake regije.

    Za svaku regiju vraća granice (lon_min, lon_max, lat_min, lat_max). Dodaje
    mali rub (pad, u stupnjevima) da okvir sigurno obuhvati sve postaje. Ovaj
    okvir koristi download.py da s Copernicusa skine samo to malo područje,
    a ne cijelo Sredozemlje.
    """
    boxes = {}
    for r, g in df.groupby("region"):
        boxes[r] = (float(g.lon.min()) - pad, float(g.lon.max()) + pad,
                    float(g.lat.min()) - pad, float(g.lat.max()) + pad)
    return boxes


if __name__ == "__main__":
    # Brza provjera pri ručnom pokretanju: ispiše koliko ima površinskih uzoraka
    # po regiji i varijabli, njihov vremenski raspon, raspon vrijednosti i okvire.
    df = load_all()
    print(f"ukupno površinskih uzoraka: {len(df)}")
    for (r, v), g in df.groupby(["region", "var"]):
        print(f"  {r:12s} {v:4s}: {len(g):6d} | "
              f"{g.date.min().date()} -> {g.date.max().date()} | "
              f"raspon {g.value.min():.2f}-{g.value.max():.2f}")
    print("\ngranice (lon_min, lon_max, lat_min, lat_max):")
    for r, b in region_boxes(df).items():
        print(f"  {r:12s}: {tuple(round(x,3) for x in b)}")
