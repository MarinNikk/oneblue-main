# -*- coding: utf-8 -*-
"""Pokazatelji uspješnosti, zajednički za sve modele.

Ovdje su funkcije koje mjere koliko je predikcija struja dobra: R², RMSE, MAE
i Skill score u odnosu na perzistenciju. Svi modeli (perzistencija, LSTM, GNN)
ocjenjuju se istim mjerilima pa su rezultati izravno usporedivi. Mjerenje se
radi u fizikalnim jedinicama (m/s), zato se predikcije prvo vraćaju iz
normaliziranog prostora funkcijom denorm().
"""
import numpy as np


def denorm(x, mean, std):
    """Vrati vrijednosti iz normaliziranog natrag u fizikalni prostor (m/s).

    Obrnuto od normalizacije iz data.py: množimo standardnom devijacijom i
    dodajemo prosjek. Metrike računamo u m/s da brojke imaju fizikalni smisao.
    """
    return x * std + mean


def metrics(pred, true, ref):
    """Izračunaj skup pokazatelja za jednu predikciju.

    Prima tri niza u m/s (mogu biti bilo kojeg oblika koji završava s 2, tj.
    komponentama u i v):
      - pred: predikcija modela,
      - true: stvarne izmjerene vrijednosti,
      - ref:  referentna predikcija (perzistencija) za Skill score.

    Vraća rječnik:
      - R2:    udio objašnjene varijance (1 = savršeno, 0 = kao prosjek),
      - RMSE:  korijen srednje kvadratne pogreške (m/s),
      - MAE:   srednja apsolutna pogreška (m/s),
      - Skill: 1 - MSE(model)/MSE(perzistencija). Skill > 0 znači da je model
               bolji od perzistencije, 0 je jednako dobar, < 0 gori.
    """
    # sve spljoštimo u 1D da metrike računamo preko svih točaka i komponenti odjednom
    p, t, r = pred.reshape(-1), true.reshape(-1), ref.reshape(-1)
    mse = np.mean((p - t) ** 2)
    # R²: koliko model smanjuje pogrešku u odnosu na puko predviđanje prosjeka
    r2 = 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p - t)))
    # Skill: usporedba s perzistencijom. Dijelimo MSE modela s MSE perzistencije,
    # pa je pozitivno samo ako model pogađa bolje od "sutra = danas".
    skill = float(1 - mse / np.mean((r - t) ** 2))
    return dict(R2=float(r2), RMSE=rmse, MAE=mae, Skill=skill)


def metrics_by_component(pred, true, ref):
    """Iste pokazatelje izračunaj odvojeno za U i V komponentu struje.

    U je zonalna (istok-zapad), V meridionalna (sjever-jug) komponenta. Dijeljenjem
    po komponenti vidi se pogađa li model jednu smjer bolje od druge. Vraća
    rječnik oblika {"U": {...metrike...}, "V": {...metrike...}}.
    """
    # ...,i bira i-tu komponentu (0 = U, 1 = V) iz zadnje osi niza
    return {name: metrics(pred[..., i], true[..., i], ref[..., i])
            for i, name in enumerate(("U", "V"))}
