# -*- coding: utf-8 -*-
"""Usporedba Copernicus reanalize s in-situ mjerenjima (temperatura i salinitet).

Treći i zadnji korak postupka: spajamo izmjerene vrijednosti (Cipar +
Katalonija) s modelskim vrijednostima i brojčano ocjenjujemo koliko se model
slaže s mjerenjima. Za svako mjerenje nađemo najbližu Copernicus ćeliju i
najbliži mjesec (to nazivamo kolokacijom), pa računamo metrike: bias
(pristranost), RMSE, MAE i koeficijent korelacije r.

Rezultat (tablica u JSON-u, spojeni podaci i dvije slike) sprema se u
marin/results/. Glavni nalaz: temperatura se izvrsno slaže (r oko 0.89-0.94),
a salinitet loše, jer model na 4.2 km razlučivosti ne vidi obalno slađenje
(riječni utok) pa drži gotovo konstantan salinitet otvorenog mora.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from insitu import load_all

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parents[1] / "marin/results"
OUT.mkdir(parents=True, exist_ok=True)

# Za svaku varijablu: (kratko ime u Copernicus datoteci, lijepa oznaka za grafove).
VARS = {"temp": ("thetao", "Temperatura [°C]"), "sal": ("so", "Salinitet [psu]")}
REGIONS = ["Cipar", "Katalonija"]
COP_END = "2022-12-31"   # model ide samo do ovog datuma, pa kasnija mjerenja izbacujemo


def collocate():
    """Spoji svako in-situ mjerenje s odgovarajućom modelskom vrijednošću.

    Kolokacija znači: za točku i datum mjerenja pronaći vrijednost modela na
    najbližoj ćeliji grida i najbližem mjesecu. Vraća DataFrame mjerenja s
    dodanim stupcem "model" (vrijednost Copernicusa na tom mjestu i vremenu).
    """
    ins = load_all()
    # Mjerenja nakon kraja reanalize nemamo s čime usporediti, pa ih maknemo.
    ins = ins[ins["date"] <= COP_END]
    out = []
    for region in REGIONS:
        for var, (short, _) in VARS.items():
            sub = ins[(ins.region == region) & (ins["var"] == var)].copy()
            if sub.empty:
                continue
            ds = xr.open_dataset(DATA / f"cop_{region.lower()}_{var}.nc")
            da = ds[short].isel(depth=0)                       # uzmi površinski sloj
            # Pripremimo sve mjerne točke kao nizove (vrijeme, lat, lon). Dimenzija
            # "s" znači "uzorak", tj. svaki redak je jedno mjerenje.
            pts = dict(
                time=xr.DataArray(sub["date"].values, dims="s"),
                latitude=xr.DataArray(sub["lat"].values, dims="s"),
                longitude=xr.DataArray(sub["lon"].values, dims="s"),
            )
            # Ovo je sama kolokacija: method="nearest" za svaku točku uzme vrijednost
            # najbliže ćelije grida i najbližeg mjeseca. Grid i mjerenje rijetko padnu
            # točno na istu koordinatu/datum, pa "najbliže" je praktično rješenje.
            sub["model"] = da.sel(**pts, method="nearest").values
            out.append(sub)
    # Neke obalne točke padnu na ćeliju koja je kod modela kopno (vrijednost NaN).
    # Takve izbacujemo jer model tamo nema more pa usporedba nema smisla.
    df = pd.concat(out, ignore_index=True).dropna(subset=["model"])
    return df


def metrics(obs, mod):
    """Izračunaj mjere slaganja između mjerenja (obs) i modela (mod).

    obs i mod su nizovi iste duljine (mjereno i modelski predviđeno). Vraća:
    N (broj parova), bias (prosječna razlika model minus mjereno; pokazuje
    sustavno precjenjivanje/podcjenjivanje), RMSE (korijen srednje kvadratne
    pogreške, kažnjava velika odstupanja), MAE (srednja apsolutna pogreška) i
    r (koeficijent korelacije, koliko isti oblik prate, od -1 do 1).
    """
    d = mod - obs   # razlika po svakoj točki; pozitivno = model je veći od mjerenja
    return dict(N=int(len(obs)), bias=float(d.mean()),
                RMSE=float(np.sqrt((d ** 2).mean())), MAE=float(np.abs(d).mean()),
                r=float(np.corrcoef(obs, mod)[0, 1]),
                obs_mean=float(obs.mean()), mod_mean=float(mod.mean()))


def summarize(df):
    """Izračunaj metrike zasebno za svaku kombinaciju regije i varijable.

    Vraća rječnik oblika {"Cipar / temp": {metrike...}, ...}. Preskače grupe s
    premalo podataka (2 ili manje), jer na tako malo parova metrike nisu pouzdane.
    """
    rows = {}
    for region in REGIONS:
        for var in VARS:
            g = df[(df.region == region) & (df["var"] == var)]
            if len(g) > 2:
                rows[f"{region} / {var}"] = metrics(g["value"].values, g["model"].values)
    return rows


def _scatter(df):
    """Nacrtaj raspršeni dijagram (scatter) mjereno naspram modela, 2x2 mreža.

    Svaki podgraf je jedna regija + varijabla. Točke koje leže blizu crtkane
    linije 1:1 znače dobro slaganje modela i mjerenja. Sliku sprema kao PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for i, region in enumerate(REGIONS):
        for j, (var, (_, lab)) in enumerate(VARS.items()):
            ax = axes[i, j]
            g = df[(df.region == region) & (df["var"] == var)]
            if g.empty:
                ax.set_visible(False); continue
            m = metrics(g["value"].values, g["model"].values)
            ax.scatter(g["value"], g["model"], s=6, alpha=0.25, color="#2c6fbb",
                       edgecolors="none")
            lo = min(g["value"].min(), g["model"].min())
            hi = max(g["value"].max(), g["model"].max())
            # Crtkana linija 1:1 (y = x): idealan slučaj gdje je model jednak mjerenju.
            ax.plot([lo, hi], [lo, hi], "--", color="#444", lw=1)
            ax.set_xlabel(f"Mjereno - {lab}")
            ax.set_ylabel(f"Copernicus - {lab}")
            ax.set_title(f"{region} · {lab.split(' ')[0]}\n"
                         f"N={m['N']}  bias={m['bias']:+.2f}  RMSE={m['RMSE']:.2f}  r={m['r']:.2f}",
                         fontsize=10)
    fig.suptitle("Copernicus reanaliza naspram in-situ mjerenja", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "copernicus_scatter.png", dpi=150)


def _seasonal(df):
    """Nacrtaj sezonski ciklus: prosjek po mjesecu, mjereno naspram modela.

    Za svaki mjesec (1-12) uzmemo prosjek mjerenja i prosjek modela i nacrtamo
    dvije krivulje. Lijepo pokazuje prati li model godišnji hod temperature i
    saliniteta. Sliku sprema kao PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # Iz datuma izvučemo broj mjeseca (1-12) da možemo grupirati po mjesecu
    # preko svih godina i dobiti prosječni sezonski ciklus.
    df = df.copy(); df["month"] = df["date"].dt.month
    for i, region in enumerate(REGIONS):
        for j, (var, (_, lab)) in enumerate(VARS.items()):
            ax = axes[i, j]
            g = df[(df.region == region) & (df["var"] == var)]
            if g.empty:
                ax.set_visible(False); continue
            o = g.groupby("month")["value"].mean()
            m = g.groupby("month")["model"].mean()
            ax.plot(o.index, o.values, "-o", color="#1b7837", label="mjereno", ms=4)
            ax.plot(m.index, m.values, "-s", color="#2c6fbb", label="Copernicus", ms=4)
            ax.set_title(f"{region} · {lab.split(' ')[0]}", fontsize=10)
            ax.set_xlabel("mjesec"); ax.set_ylabel(lab); ax.set_xticks(range(1, 13))
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Sezonski ciklus: Copernicus naspram mjerenja", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "copernicus_seasonal.png", dpi=150)


def main():
    # Cijeli tijek: spoji mjerenja s modelom, izračunaj metrike, ispiši tablicu
    # te spremi rezultate (JSON + parquet) i dvije slike u marin/results/.
    df = collocate()
    res = summarize(df)
    print(f"{'Regija / var':22s}{'N':>7s}{'bias':>8s}{'RMSE':>8s}{'MAE':>8s}{'r':>7s}")
    print("-" * 60)
    for k, m in res.items():
        print(f"{k:22s}{m['N']:>7d}{m['bias']:>8.2f}{m['RMSE']:>8.2f}"
              f"{m['MAE']:>8.2f}{m['r']:>7.2f}")
    with open(OUT / "copernicus_vs_insitu.json", "w") as f:
        json.dump({"note": "Copernicus MEDSEA_MULTIYEAR_PHY_006_004 (mjesečno, površina) "
                   "vs in-situ; 1994-2022", "metrics": res}, f, indent=2, ensure_ascii=False)
    df.to_parquet(OUT / "copernicus_matched.parquet")
    _scatter(df)
    _seasonal(df)
    print(f"\nSpremljeno u {OUT}: copernicus_vs_insitu.json, "
          f"copernicus_scatter.png, copernicus_seasonal.png, copernicus_matched.parquet")


if __name__ == "__main__":
    main()
