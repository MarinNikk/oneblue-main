# -*- coding: utf-8 -*-
"""PointLSTM: sekvencijski model koji uči vremensku dinamiku po točkama.

Ovo je prvi pravi (naučeni) model u usporedbi. Jedan isti LSTM primjenjuje se
na svaku morsku točku zasebno, pa model uči samo kako se struja u pojedinoj
točki mijenja kroz vrijeme (7 dana unatrag), bez gledanja susjednih točaka.
Radi s residual pristupom: ne predviđa apsolutnu struju nego deltu (korekciju)
u odnosu na zadnji dan, jer je promjena iz dana u dan obično mala i lakše ju
je naučiti nego cijelo polje iznova.
"""
import torch
import torch.nn as nn


class PointLSTM(nn.Module):
    """LSTM dijeljen preko svih morskih točaka (uči samo vremensku dinamiku).

    Parametri konstruktora:
      - in_dim:   broj ulaznih značajki po danu (2: komponente u i v),
      - hidden:   veličina skrivenog stanja LSTM-a,
      - layers:   broj slojeva LSTM-a,
      - out_dim:  broj izlaznih vrijednosti po točki (2: u i v),
      - residual: ako je True, izlaz je zadnji dan + naučena delta (residual pristup).
    """

    def __init__(self, in_dim=2, hidden=48, layers=2, out_dim=2, residual=True):
        super().__init__()
        self.residual = residual
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True,
                            dropout=0.1 if layers > 1 else 0.0)
        # linearni sloj ("glava") pretvara skriveno stanje u predikciju (u, v)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, adj=None):
        """Propusti ulaz kroz model. adj se ignorira (LSTM ne koristi graf).

        x je oblika (B, seq, N, 2): B prozora u seriji, seq dana, N točaka,
        2 komponente. Vraća predikciju oblika (B, N, 2) za sljedeći dan.
        """
        B, S, N, Fin = x.shape                              # x: (B, seq, N, 2)
        base = x[:, -1, :, :2]                              # zadnji dan = perzistencija (osnova za residual)
        # spojimo serije i točke u jednu os (B*N) pa LSTM obrađuje svaku točku
        # kao zasebnu vremensku sekvenciju od S dana
        xf = x.permute(0, 2, 1, 3).reshape(B * N, S, Fin)
        out, _ = self.lstm(xf)
        # uzimamo samo izlaz zadnjeg vremenskog koraka i vraćamo oblik (B, N, 2)
        delta = self.head(out[:, -1, :]).view(B, N, -1)     # naučena korekcija (delta)
        # residual: predikcija = zadnji dan + delta; inače model predviđa izravno
        return base + delta if self.residual else delta
