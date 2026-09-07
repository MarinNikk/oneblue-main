# -*- coding: utf-8 -*-
"""Zajednički alat za treniranje, predikciju i mjerenje vremena.

Ovdje su funkcije koje rade isto za svaki naučeni model (LSTM i GNN): petlja
treniranja s ranim zaustavljanjem, predikcija po serijama i mjerenje brzine
zaključivanja. Time se modeli treniraju i mjere na potpuno isti način, pa je
usporedba poštena. Ove funkcije poziva orkestrator train.py.
"""
import time

import numpy as np
import torch
import torch.nn as nn


def batches(n, bs, shuffle=True):
    """Generator indeksa serija (mini-batcheva) preko n primjera.

    Vraća komade po bs indeksa. Kad je shuffle=True (treniranje), redoslijed se
    izmiješa svaku epohu da model ne uči redoslijed primjera napamet.
    """
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def predict(model, X, adj, bs):
    """Pokreni model nad cijelim skupom X i vrati predikcije kao numpy (n, N, 2).

    Radi u serijama od bs primjera da stane u memoriju. model.eval() i
    torch.no_grad() isključuju treniranje (npr. dropout) i računanje gradijenata
    jer ovdje samo predviđamo, ne učimo.
    """
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs])
            outs.append(model(xb, adj).numpy())
    return np.concatenate(outs, 0)


def time_inference(model, X, adj, bs, repeats=3):
    """Izmjeri prosječno vrijeme jedne predikcije nad cijelim skupom (u sekundama).

    Prvo pokrene predikciju jednom "za zagrijavanje" (da se izbjegnu troškovi
    prvog pokretanja), pa onda mjeri prosjek kroz repeats ponavljanja.
    """
    predict(model, X, adj, bs)            # zagrijavanje (rezultat se ne mjeri)
    t0 = time.time()
    for _ in range(repeats):
        predict(model, X, adj, bs)
    return (time.time() - t0) / repeats


def train(model, d, adj, name, epochs, bs, lr=1e-3, patience=4, verbose=True):
    """Treniraj model uz rano zaustavljanje i vrati (model, info).

    Parametri:
      - model:    neistrenirani model (PointLSTM ili STGAT),
      - d:        rječnik podataka iz make_dataset(),
      - adj:      matrica susjedstva (tensor), prosljeđuje se modelu,
      - name:     naziv modela za ispis,
      - epochs:   najveći broj epoha,
      - bs:       veličina serije,
      - lr:       stopa učenja,
      - patience: koliko epoha bez poboljšanja na validaciji prije zaustavljanja.

    Koristi MSE gubitak i Adam optimizator. Rano zaustavljanje prati gubitak na
    validaciji: ako se patience epoha zaredom ne popravi, učenje staje i vraća
    se najbolje viđeno stanje modela (sprječava prenaučenost). info je rječnik
    s brojem parametara, vremenom treniranja, brojem odrađenih epoha i najboljim
    validacijskim gubitkom.
    """
    Xtr, Ytr, Xva, Yva = d["Xtr"], d["Ytr"], d["Xva"], d["Yva"]
    # weight_decay je blaga L2 regularizacija koja drži težine malima
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss()
    # pamtimo najbolji rezultat na validaciji i pripadno stanje modela
    best_val, best_state, wait = float("inf"), None, 0
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[{name}] parametara: {n_params:,}")
    per_epoch, t_start, ran = [], time.time(), 0
    for ep in range(1, epochs + 1):
        model.train()                      # uključi način treniranja (dropout aktivan)
        t0 = time.time()
        tot = 0.0
        for bi in batches(len(Xtr), bs):
            xb = torch.from_numpy(Xtr[bi])
            yb = torch.from_numpy(Ytr[bi])
            opt.zero_grad()                # očisti stare gradijente
            loss = lossf(model(xb, adj), yb)
            loss.backward()                # izračunaj gradijente (backpropagation)
            opt.step()                     # ažuriraj težine
            tot += loss.item() * len(bi)   # zbroj gubitka, pomnožen veličinom serije
        # na kraju svake epohe provjeri model na validacijskom skupu
        vp = predict(model, Xva, adj, bs)
        vloss = float(np.mean((vp - Yva) ** 2))
        per_epoch.append(time.time() - t0)
        ran = ep
        if verbose:
            print(f"  ep {ep:02d}  train {tot/len(Xtr):.4f}  val {vloss:.4f}  "
                  f"({per_epoch[-1]:.1f}s)")
        # ako se validacija popravila (s malom marginom), spremi ovo stanje kao najbolje
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            # nema poboljšanja: brojimo strpljenje i stajemo kad ga potrošimo
            wait += 1
            if wait >= patience:
                if verbose:
                    print("  rano zaustavljanje")
                break
    if best_state:
        model.load_state_dict(best_state)  # vrati najbolje stanje, ne zadnje
    info = dict(
        params=int(n_params),
        train_time=float(time.time() - t_start),
        epochs_run=int(ran),
        sec_per_epoch=float(np.mean(per_epoch)),
        best_val=float(best_val),
    )
    return model, info
