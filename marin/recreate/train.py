# -*- coding: utf-8 -*-
"""Orkestrator: pokreće cijeli eksperiment i sprema usporedbu modela.

Ovo je glavna skripta koja sve povezuje. Učita podatke, izračuna perzistenciju
kao referencu, istrenira LSTM i GNN, ocijeni ih na testnom skupu, izmjeri
vrijeme, ispiše tablice i spremi rezultate (JSON, slike i predikcije). Pokreni
je sa `python train.py` iz mape marin/recreate/.
"""
import json
from pathlib import Path

import numpy as np
import torch

from data import make_dataset
from metrics import metrics, metrics_by_component, denorm
from baseline import persistence_predict
from lstm_model import PointLSTM
from gnn_model import STGAT
from engine import train, predict, time_inference

# Fiksiramo slučajne sjemenke da rezultati budu ponovljivi (isti svaki put).
torch.manual_seed(0)
np.random.seed(0)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "marin/results"
OUT.mkdir(parents=True, exist_ok=True)

# Konfiguracija po modelu: broj epoha i veličina serije. GNN je teži pa ima
# manje epoha i manju seriju.
CFG = dict(lstm=dict(epochs=25, bs=32), gnn=dict(epochs=16, bs=12))


def main():
    d = make_dataset()
    adj = torch.from_numpy(d["adjacency"])
    mean, std = d["mean"], d["std"]
    Xte, Yte = d["Xte"], d["Yte"]

    # Sve vraćamo u m/s (denorm) da su metrike u fizikalnim jedinicama.
    true = denorm(Yte, mean, std)                           # stvarne vrijednosti
    pers = denorm(persistence_predict(Xte), mean, std)      # predikcija perzistencije

    results, by_comp, timing = {}, {}, {}

    # Perzistencija: referenca bez treniranja. Ovdje je ref = pers, pa joj je
    # Skill po definiciji 0 (uspoređuje se sama sa sobom).
    results["Persistencija"] = metrics(pers, true, pers)
    by_comp["Persistencija"] = metrics_by_component(pers, true, pers)
    timing["Persistencija"] = dict(params=0, train_time=0.0, epochs_run=0,
                                   sec_per_epoch=0.0)

    # LSTM: istreniraj, predvidi na testu, vrati u m/s i ocijeni. Skill se uvijek
    # računa u odnosu na perzistenciju (pers) da vidimo je li model bolji od nje.
    lstm, info = train(PointLSTM(), d, adj, "PointLSTM", **CFG["lstm"])
    lp = denorm(predict(lstm, Xte, adj, CFG["lstm"]["bs"]), mean, std)
    results["LSTM (po točkama)"] = metrics(lp, true, pers)
    by_comp["LSTM (po točkama)"] = metrics_by_component(lp, true, pers)
    info["infer_time"] = time_inference(lstm, Xte, adj, CFG["lstm"]["bs"])
    timing["LSTM (po točkama)"] = info

    # GNN (ST-GAT): isti postupak kao za LSTM.
    gnn, info = train(STGAT(), d, adj, "ST-GAT (GNN)", **CFG["gnn"])
    gp = denorm(predict(gnn, Xte, adj, CFG["gnn"]["bs"]), mean, std)
    results["ST-GAT (GNN)"] = metrics(gp, true, pers)
    by_comp["ST-GAT (GNN)"] = metrics_by_component(gp, true, pers)
    info["infer_time"] = time_inference(gnn, Xte, adj, CFG["gnn"]["bs"])
    timing["ST-GAT (GNN)"] = info

    _print_tables(results, by_comp, timing, d, len(Xte))
    with open(OUT / "recreate_comparison.json", "w") as f:
        json.dump({"N_nodes": int(d["N"]), "test_days": int(len(Xte)),
                   "results": results, "by_component": by_comp,
                   "timing": timing}, f, indent=2, ensure_ascii=False)
    # Spremamo i same predikcije da se razdvajanja/analize mogu ponovno računati
    # bez ponovnog treniranja (treniranje je sporo, ovo je instant).
    np.savez_compressed(OUT / "recreate_preds.npz", true=true, pers=pers,
                        lstm=lp, gnn=gp)
    _fig_metrics(results)
    _fig_timing(results, timing)
    _fig_uv(by_comp)
    print(f"\nSpremljeno u {OUT}/recreate_comparison.json (+ .png, "
          f"recreate_timing.png, recreate_uv.png, recreate_preds.npz)")


def _print_tables(results, by_comp, timing, d, n_test):
    """Ispiši dvije tablice u terminal: ukupne metrike i razdvajanje po komponenti."""
    print("\n" + "=" * 70)
    print(f"USPOREDBA  (testni skup, N={d['N']} čvorova, {n_test} dana)")
    print("=" * 70)
    print(f"{'Model':22s}{'R²':>8s}{'RMSE':>9s}{'MAE':>9s}{'Skill':>9s}"
          f"{'Train[s]':>10s}")
    print("-" * 70)
    for k in results:
        m, t = results[k], timing[k]
        print(f"{k:22s}{m['R2']:>8.3f}{m['RMSE']:>9.4f}{m['MAE']:>9.4f}"
              f"{m['Skill']:>9.3f}{t['train_time']:>10.1f}")
    print("-" * 70)
    print("RMSE/MAE u m/s; Skill>0 = bolje od persistencije.")

    print("\nRAZDVAJANJE PO KOMPONENTI (R² / RMSE)")
    print("-" * 70)
    print(f"{'Model':22s}{'R² U':>9s}{'R² V':>9s}{'RMSE U':>10s}{'RMSE V':>10s}")
    for k in by_comp:
        u, v = by_comp[k]["U"], by_comp[k]["V"]
        print(f"{k:22s}{u['R2']:>9.3f}{v['R2']:>9.3f}"
              f"{u['RMSE']:>10.4f}{v['RMSE']:>10.4f}")
    print("-" * 70)
    print("U = zonalna (istok-zapad), V = meridionalna (sjever-jug).")


def _fig_metrics(results):
    """Spremi stupčasti grafikon R² i RMSE po modelu (recreate_comparison.png)."""
    import matplotlib
    matplotlib.use("Agg")              # Agg radi bez ekrana (sprema ravno u datoteku)
    import matplotlib.pyplot as plt
    models = list(results)
    r2 = [results[m]["R2"] for m in models]
    rmse = [results[m]["RMSE"] for m in models]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    c = ["#9ecae1", "#2c6fbb", "#1b7837"]
    a1.bar(models, r2, color=c); a1.set_title("R² (testni skup)"); a1.set_ylim(0, 1)
    a1.tick_params(axis="x", rotation=20)
    for i, v in enumerate(r2):
        a1.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    a2.bar(models, rmse, color=c); a2.set_title("RMSE [m/s]")
    a2.tick_params(axis="x", rotation=20)
    for i, v in enumerate(rmse):
        a2.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "recreate_comparison.png", dpi=150)


def _fig_uv(by_comp):
    """Grupirani stupci: R² po komponenti (U vs V) za svaki model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    models = list(by_comp)
    r2u = [by_comp[m]["U"]["R2"] for m in models]
    r2v = [by_comp[m]["V"]["R2"] for m in models]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    b1 = ax.bar(x - w / 2, r2u, w, label="U (zonalna)", color="#2c6fbb")
    b2 = ax.bar(x + w / 2, r2v, w, label="V (meridionalna)", color="#7fb0e0")
    ax.set_ylabel("R² (testni skup)"); ax.set_ylim(0, 1)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
    ax.set_title("Točnost po komponenti struje")
    ax.legend()
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "recreate_uv.png", dpi=150)


def _fig_timing(results, timing):
    """Trošak (vrijeme treniranja) naspram dobitka (Skill nad persistencijom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # uzimamo samo naučene modele (perzistencija ima train_time = 0)
    learned = [m for m in results if timing[m]["train_time"] > 0]
    tt = [timing[m]["train_time"] for m in learned]
    skill = [results[m]["Skill"] for m in learned]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cols = {"LSTM (po točkama)": "#2c6fbb", "ST-GAT (GNN)": "#1b7837"}
    for m, x, y in zip(learned, tt, skill):
        ax.scatter(x, y, s=140, color=cols.get(m, "#888"), zorder=3)
        ax.annotate(f"  {m}\n  R²={results[m]['R2']:.3f}", (x, y), va="center")
    ax.set_xlabel("Vrijeme treniranja [s]")
    ax.set_ylabel("Skill score (nad persistencijom)")
    ax.set_title("Trošak vremena naspram dobitka točnosti")
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout(); fig.savefig(OUT / "recreate_timing.png", dpi=150)


if __name__ == "__main__":
    main()
