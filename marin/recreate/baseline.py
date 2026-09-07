# -*- coding: utf-8 -*-
"""Perzistencija: najjednostavnija referentna predikcija.

Perzistencija pretpostavlja da će sutra biti isto kao danas, pa za predikciju
samo prepisuje zadnji dan ulaznog prozora. Nema učenja ni parametara. Služi kao
mjerilo protiv kojeg se ocjenjuju pravi modeli (LSTM, GNN): ako neki model ne
uspije nadmašiti perzistenciju, onda zapravo nije naučio ništa korisno.
"""


def persistence_predict(X):
    """Vrati zadnji dan svakog ulaznog prozora kao predikciju sljedećeg dana.

    Prima X oblika (n, seq, N, 2): n prozora, svaki od seq dana, N morskih
    čvorova i 2 komponente (u, v). Vraća (n, N, 2), tj. X[:, -1] je zadnji
    (najnoviji) dan svakog prozora, što je predikcija po principu "sutra = danas".
    """
    return X[:, -1]
