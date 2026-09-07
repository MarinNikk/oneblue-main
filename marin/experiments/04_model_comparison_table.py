# -*- coding: utf-8 -*-
"""Bogatija tablica usporedbe modela: uz R2 dodaje i RMSE, MAE i Skill.

Tablica na slajdu obrane pokazuje samo R2 (u) i R2 (v). Sve ostale mjere
(RMSE, MAE, Skill nad perzistencijom) već su bile izračunate u istom pokretanju,
samo se nisu prikazale. Ova skripta ih izvlači iz gotove datoteke rezultata
marin/results/recreate_comparison.json i slaže ih u jednu preglednu tablicu.

Važno: brojevi ovdje odnose se na tri modela koja su vrednovana pod POTPUNO
istim uvjetima (isti test skup, ista maska, isti split), pa su izravno usporedivi:
perzistencija, LSTM po točkama i ST-GAT (graf mreža). Za RBF interpolaciju,
RBF-LSTM i ARIMA imamo samo R2 iz zasebnih pokretanja (drugačiji zadatak ili
split), pa se navode odvojeno, kao referenca, bez RMSE/MAE.

Izlaz (u marin/results/):
  - model_comparison_full_metrics.md   (markdown tablica)
  - model_comparison_full_metrics.png  (ista tablica kao slika, za slajd)

Pokretanje:
  cd marin/experiments
  ../../.venv/bin/python 04_model_comparison_table.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # crtamo u datoteku, bez otvaranja prozora
import matplotlib.pyplot as plt

# Putanje: rezultati su u marin/results/, neovisno odakle se skripta pokrece.
RESULTS = Path(__file__).resolve().parents[1] / "results"
SRC_JSON = RESULTS / "recreate_comparison.json"
OUT_MD = RESULTS / "model_comparison_full_metrics.md"
OUT_PNG = RESULTS / "model_comparison_full_metrics.png"

# Lijepa hrvatska imena redaka i redoslijed kakav zelimo u tablici.
# Kljucevi lijevo moraju tocno odgovarati imenima u JSON-u.
MODELS = [
    ("Persistencija", "Perzistencija"),
    ("LSTM (po točkama)", "LSTM (po točkama)"),
    ("ST-GAT (GNN)", "ST-GAT (graf mreža)"),
]

# Referentni modeli za koje imamo samo R2 (iz zasebnih pokretanja na kodu kolega
# i biljeznicama). RMSE/MAE za njih nije zasebno racunat, pa stoje odvojeno da se
# ne mijesaju s izravno usporedivim brojkama iznad.
REFERENCE = [
    ("RBF interpolacija", "≈ 1,0", "≈ 1,0", "prostorna interpolacija (lakši zadatak, nije izravno usporediva s predikcijom)"),
    ("RBF-LSTM (rekonstrukcija)", "≈ 0,5", "≈ 0,5", "koeficijenti se dobro predvide, ali rekonstrukcija polja gubi točnost"),
    ("ARIMA (jedna točka)", "≈ 0,68", "-", "klasična vremenska serija po jednoj točki"),
]


def hr(x, dec):
    """Broj u hrvatskom zapisu: fiksan broj decimala i zarez umjesto tocke."""
    return f"{x:.{dec}f}".replace(".", ",")


def hr_signed(x, dec):
    """Kao hr(), ali s vodecim + za pozitivne (za Skill, gdje predznak nosi znacenje)."""
    return f"{x:+.{dec}f}".replace(".", ",")


def load_rows():
    """Iz JSON-a slozi retke tablice s po-komponentnim i ukupnim mjerama.

    Vraca listu rjecnika, jedan po modelu, s vec formatiranim (tekst) poljima
    spremnim za ispis u markdown i u sliku.
    """
    data = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    by_comp = data["by_component"]       # R2/RMSE/MAE/Skill zasebno za u i v
    overall = data["results"]            # ukupni Skill (preko obje komponente)

    rows = []
    for key, label in MODELS:
        u = by_comp[key]["U"]
        v = by_comp[key]["V"]
        rows.append({
            "model": label,
            "r2_u": hr(u["R2"], 3),   "r2_v": hr(v["R2"], 3),
            "rmse_u": hr(u["RMSE"], 4), "rmse_v": hr(v["RMSE"], 4),
            "mae_u": hr(u["MAE"], 4),   "mae_v": hr(v["MAE"], 4),
            "skill": hr_signed(overall[key]["Skill"], 3),
        })
    return data, rows


def write_md(data, rows):
    """Zapise markdown tablicu (glavna + referentna) u OUT_MD."""
    n_nodes = data["N_nodes"]
    n_days = data["test_days"]
    lines = []
    lines.append("# Usporedba modela: sve mjere (test skup)\n")
    lines.append(f"Izvor: `marin/results/recreate_comparison.json`  |  "
                 f"{n_nodes} oceanskih točaka  |  {n_days} test dana  |  "
                 f"RMSE i MAE su u m/s.\n")
    lines.append("Sva tri modela vrednovana su pod istim uvjetima (isti test "
                 "skup, maska i split), pa su izravno usporediva.\n")
    # Zaglavlje glavne tablice.
    lines.append("| Model | R2 (u) | R2 (v) | RMSE (u) | RMSE (v) | MAE (u) | MAE (v) | Skill |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['model']} | {r['r2_u']} | {r['r2_v']} | "
                     f"{r['rmse_u']} | {r['rmse_v']} | {r['mae_u']} | {r['mae_v']} | {r['skill']} |")
    lines.append("")
    lines.append("Skill je poboljšanje nad perzistencijom (0 znači jednako kao "
                 "perzistencija, pozitivno je bolje).\n")
    # Referentni modeli: samo R2, jasno odvojeni.
    lines.append("## Referentni modeli (samo R2, iz zasebnih pokretanja)\n")
    lines.append("Za ove modele RMSE i MAE nisu zasebno računati u ovom pokretanju.\n")
    lines.append("| Model | R2 (u) | R2 (v) | Napomena |")
    lines.append("|---|---|---|---|")
    for name, r2u, r2v, note in REFERENCE:
        lines.append(f"| {name} | {r2u} | {r2v} | {note} |")
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Zapisano: {OUT_MD}")


def write_png(data, rows):
    """Nacrta glavnu tablicu kao sliku (drop-in za slajd)."""
    header = ["Model", "R² (u)", "R² (v)", "RMSE (u)", "RMSE (v)", "MAE (u)", "MAE (v)", "Skill"]
    cells = [[r["model"], r["r2_u"], r["r2_v"], r["rmse_u"], r["rmse_v"],
              r["mae_u"], r["mae_v"], r["skill"]] for r in rows]
    # Prvi stupac (ime modela) treba biti siri jer su imena duga; ostali su uski.
    col_widths = [0.26] + [0.1057] * 7

    fig, ax = plt.subplots(figsize=(13, 2.2))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=header, colWidths=col_widths,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    # Ime modela poravnaj lijevo (citljivije od centriranog dugackog teksta).
    for row in range(1, len(rows) + 1):
        tbl[row, 0].set_text_props(ha="left")
        tbl[row, 0].PAD = 0.04

    # Bojanje: tamno-tirkizno zaglavlje (boja kao na slajdovima), naizmjenicni redci.
    teal = "#0e5e6f"
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cfd8dc")
        if row == 0:
            cell.set_facecolor(teal)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f2f6f7")
        if col == 0 and row > 0:
            cell.set_text_props(fontweight="bold")   # ime modela podebljano

    n_nodes = data["N_nodes"]
    n_days = data["test_days"]
    ax.set_title("Usporedba modela predikcije: sve mjere (test skup)",
                 fontsize=13, fontweight="bold", pad=14)
    fig.text(0.5, 0.02,
             f"{n_nodes} oceanskih točaka, {n_days} test dana; RMSE i MAE u m/s; "
             f"Skill je poboljšanje nad perzistencijom",
             ha="center", fontsize=8, color="#555")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {OUT_PNG}")


def main():
    data, rows = load_rows()
    write_md(data, rows)
    write_png(data, rows)


if __name__ == "__main__":
    main()
