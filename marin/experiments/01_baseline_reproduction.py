# -*- coding: utf-8 -*-
"""
marin eksperiment 01: reprodukcija temeljnih (baseline) modela na svježe
preuzetim CMEMS podacima.

Ponovno računa temeljne modele n-dnevnog prosjeka na stvarnim podacima o
strujama s uklonjenom plimom (razdoblje 2024-2026, preuzeto u data/raw/).
Pokreće prozor n=1 (perzistencija) i n=7 te ih usporeduje. Cilj je dvostruk:
(a) provjeriti da cijela obrada radi od pocetka do kraja na stvarnim podacima i
(b) grubo usporediti s vec spremljenim results/*.json (koji su racunati na
drugom razdoblju, pa tocno poklapanje NE ocekujemo, zelimo samo isti red
velicine).

Izlaz se zapisuje u marin/results/, spremljena mapa results/ se NE dira.

Pokretanje:
    .venv/bin/python -m marin.experiments.01_baseline_reproduction
  ili
    .venv/bin/python marin/experiments/01_baseline_reproduction.py
"""

import json
import sys
from pathlib import Path


# Skripta se pokrece kao samostalna datoteka, pa Python ne zna gdje je korijen
# repozitorija. Zato rucno dodajemo korijen (dva nivoa iznad ove datoteke:
# parents[0]=experiments, parents[1]=marin, parents[2]=korijen repoa) na pocetak
# putanje za import. Bez toga `import src...` nize ne bi uspio.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.table import Table

from src.baselines import run_mean_baseline_experiment
from src.rbf_lstm.utils.io_utils import load_ocean_data


console = Console()

# Putanje do ulaznih datoteka i izlaznog JSON-a.
# UO = istok-zapad komponenta struje (u), VO = sjever-jug komponenta (v).
DATA_DIR = "data/raw"
UO_FILE = "adriatic_currents_uo_detided_2024_2026.nc"
VO_FILE = "adriatic_currents_vo_detided_2024_2026.nc"
OUT = Path("marin/results/baseline_reproduction.json")


def main() -> None:
    """Ucita podatke, pokrene temeljne modele za prozore n=1 i n=7, ispise
    usporednu tablicu na test skupu i spremi sve rezultate u JSON.

    Ne prima argumente i ne vraca nista; nuspojava je ispis u konzolu i
    zapisivanje datoteke marin/results/baseline_reproduction.json.
    """
    console.print("[bold blue]marin/01 - baseline reproduction[/bold blue]\n")

    # Ucitavanje obje komponente struje (u i v) iz NetCDF datoteka.
    # .values vadi cisti numpy niz oblika [vrijeme, sirina, duzina].
    uo_ds, vo_ds = load_ocean_data(DATA_DIR, UO_FILE, VO_FILE)
    u = uo_ds.uo_detided.values
    v = vo_ds.vo_detided.values
    console.print(f"Loaded U {u.shape}, V {v.shape}  (timesteps={len(u)})\n")

    # Vrtimo dva temeljna modela: n=1 je perzistencija (sutra = danas),
    # n=7 je tjedni prosjek. Split 70/15/15 = trening/validacija/test.
    all_results: dict[str, dict] = {}
    for window in (1, 7):
        res = run_mean_baseline_experiment(
            u_data=u, v_data=v, window_size=window, split_ratios=(0.7, 0.15, 0.15)
        )
        all_results[f"{window}day_mean"] = res

    # Usporedna tablica: samo test skup, oba prozora jedan ispod drugoga.
    # R2 zasebno za u i v komponentu, MAE i RMSE na jacini struje (magnitudi).
    table = Table(title="Mean baselines - TEST split", header_style="bold magenta")
    table.add_column("Window", style="cyan")
    for col in ("R2 U", "R2 V", "MAE mag", "RMSE mag"):
        table.add_column(col, justify="right")
    for window in (1, 7):
        m = all_results[f"{window}day_mean"]["test"]
        table.add_row(
            f"{window}-day",
            f"{m['r2_u']:.4f}",
            f"{m['r2_v']:.4f}",
            f"{m['mae_magnitude']:.6f}",
            f"{m['rmse_magnitude']:.6f}",
        )
    console.print(table)

    # Osiguramo da izlazna mapa postoji pa spremimo sve rezultate u JSON.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[green]Saved -> {OUT}[/green]")


if __name__ == "__main__":
    main()
