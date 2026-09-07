# -*- coding: utf-8 -*-
"""
HARNESS: naučeni modeli (LSTM, ConvLSTM, RBF-LSTM, GNN) sa SVE 4 mjere na ISTOM
splitu/maski/regijama/sezonama kao temeljni modeli (02_baseline_full_metrics.py).

Cilj: sve iste brojke i metrike za sve modele, sve sezone,
sve dijelove Jadrana; usporedba svih naučenih modela s perzistencijom.

STATUS: harness spreman; nedostaju checkpointi (.pth) pa svaki `predict_*` treba
pretrenirati / učitati težine. Pokreni na GPU-u. NE trenira automatski.

Kako popuniti:
  1. Za svaki model implementiraj funkciju koja vraća predikciju polja na TEST danima
     u obliku [n_test, 2, LAT, LON] (u,v), poravnatu s `field[test_idx]` iz baseline skripte.
  2. Metrike i razbijanje po sezoni/regiji preuzmi iz zajedničkih pomoćnih funkcija dolje
     (identične onima u 02_baseline_full_metrics.py) da brojke budu dosljedne.
  3. Za višednevnu GNN predikciju (1..10) koristi autoregresivni `predict_gnn(horizon=h)`
     i usporedi sa perzistencijom na istom horizontu (Igorov kod: src/baselines/...).
"""
from pathlib import Path
import numpy as np
import xarray as xr

# Skripta se pokrece kao datoteka, pa korijen repozitorija racunamo rucno
# (dva nivoa iznad ove datoteke). UO = komponenta u (istok-zapad), VO = v (sjever-jug).
ROOT = Path(__file__).resolve().parents[2]
UO = ROOT / "data/raw/adriatic_currents_uo_detided_2024_2026.nc"
VO = ROOT / "data/raw/adriatic_currents_vo_detided_2024_2026.nc"

# ---- zajedničke pomoćne funkcije (drži identičnima s 02_baseline_full_metrics.py) ----
def load_field():
    """Ucitaj struje i koordinate i vrati polje [vrijeme, 2, sirina, duzina].

    Druga dimenzija (2) su komponente u i v slozene zajedno. Vraca torku
    (field, lat, lon, time). Mora biti identicno onome u 02 skripti da su
    brojke usporedive.
    """
    u = xr.open_dataset(UO)["uo_detided"].values.astype(np.float32)
    v = xr.open_dataset(VO)["vo_detided"].values.astype(np.float32)
    lat = xr.open_dataset(UO)["latitude"].values
    lon = xr.open_dataset(UO)["longitude"].values
    time = xr.open_dataset(UO)["time"].values
    return np.stack([u, v], 1), lat, lon, time  # [T,2,LAT,LON]

def metrics(pred, true, ref_persist):
    """Izracunaj sve 4 mjere (R2, RMSE, MAE, Skill) na spljostenim nizovima.

    Istovjetna funkciji iz 02 skripte. ref_persist je perzistencijska
    predikcija koja sluzi kao referenca za vjestinu (skill); ako je None,
    skill je 0. Maskiramo NaN (kopno) prije racunanja. R2 koristimo jer su
    vrijednosti struja blizu nule, pa su postotne mjere nestabilne.
    """
    p = pred.reshape(-1).astype(np.float64); t = true.reshape(-1).astype(np.float64)
    m = np.isfinite(p) & np.isfinite(t); p, t = p[m], t[m]
    mse = np.mean((p - t) ** 2)
    out = dict(R2=float(1 - np.sum((t-p)**2)/np.sum((t-t.mean())**2)),
               RMSE=float(np.sqrt(mse)), MAE=float(np.mean(np.abs(p-t))))
    if ref_persist is None:
        out["Skill"] = 0.0
    else:
        # Skill = 1 - greska modela / greska perzistencije (pozitivno = bolji).
        r = ref_persist.reshape(-1).astype(np.float64)[m]
        out["Skill"] = float(1 - mse/np.mean((r-t)**2))
    return out

# regije/sezone: kopiraj `region`, `seasons`, `italian_side` iz 02_baseline_full_metrics.py

# ---- mjesta za popuniti (svaka vraća [n_test,2,LAT,LON] na TEST danima) ----
# Svaka predict_* funkcija mora vratiti predikciju poravnatu s field[test_idx],
# da metrike i razbijanje po sezoni/regiji budu jednaki kao kod temeljnih modela.
def predict_persistence(field, test_idx):
    """Perzistencija: predikcija za dan t je stvarno polje prethodnog dana (t-1).

    Sluzi kao referentni (najjednostavniji) model i kao osnova za skill.
    """
    return field[test_idx - 1]

def predict_lstm(field, test_idx):
    """Predikcija LSTM modela na test danima (jos nije implementirano)."""
    raise NotImplementedError("Učitaj PointByPointLSTM checkpoint i vrati predikciju na test danima")

def predict_convlstm(field, test_idx):
    """Predikcija ConvLSTM modela na test danima (jos nije implementirano)."""
    raise NotImplementedError("Treba i 6-kanalni ulaz (u,v,wind_u,wind_v,ssh,temp), vidi src/model/adriatic_dataset.py")

def predict_rbf_lstm(field, test_idx):
    """Predikcija RBF-LSTM modela na test danima (jos nije implementirano)."""
    raise NotImplementedError("Vrati i latentni (koeficijenti) i izlazni (polje) prostor -> sve 3 mjere za oba (#304)")

def predict_gnn(field, test_idx, horizon=1):
    """Predikcija GNN (ST-GAT) modela; horizon>1 ide autoregresivno (jos nije implementirano)."""
    raise NotImplementedError("ST-GAT; za horizon>1 autoregresivno (#322/#324)")

if __name__ == "__main__":
    # Ovo je samo kostur (harness): popuni predict_* funkcije pa pokreni na GPU-u.
    print("Harness. Popuni predict_* funkcije i pokreni na GPU-u.")
    print("Za dosljednost koristi isti split/masku/regije kao 02_baseline_full_metrics.py.")
