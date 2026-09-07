# -*- coding: utf-8 -*-
"""ST-GAT: prostorno-vremenski model koji more tretira kao graf.

Za razliku od LSTM-a, ovaj model gleda i susjedne točke, ne samo vremensku
dinamiku jedne točke. More je graf (čvorovi = morske točke, bridovi = susjedstvo
iz data.py). Prvo graf-pažnja (GAT) miješa informaciju među prostornim susjedima
za svaki dan, a zatim GRU agregira te dane kroz vrijeme. Sve je ručno pisano,
bez biblioteke torch_geometric. Kao i LSTM, koristi residual pristup (predviđa
deltu u odnosu na zadnji dan).

ST u nazivu znači Spatio-Temporal (prostorno-vremenski), GAT je Graph Attention
Network, tj. graf s mehanizmom pažnje.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Jedan sloj graf-pažnje (jedna glava), ograničen na susjedstvo.

    Mehanizam pažnje svakom susjedu dodjeljuje težinu (koliko ga treba "slušati"),
    a te težine model uči sam. Ograničenje na susjedstvo znači da točka gleda
    samo svoje susjede iz grafa, ne cijelo more. Parametri: in_dim (ulazne
    značajke po čvoru), out_dim (izlazne značajke po čvoru).
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)   # linearna projekcija značajki čvora
        # a_src i a_dst su naučeni vektori pažnje (za izvorni i odredišni čvor brida)
        self.a_src = nn.Parameter(torch.zeros(out_dim))
        self.a_dst = nn.Parameter(torch.zeros(out_dim))
        # inicijalizacija težina (Xavier drži varijancu stabilnom na početku učenja)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.a_src, std=0.1)
        nn.init.normal_(self.a_dst, std=0.1)

    def forward(self, h, adj_mask):
        """Izračunaj nove značajke čvorova spajanjem susjeda preko pažnje.

        h je (B, N, in_dim) značajke čvorova, adj_mask je (N, N) bool matrica
        susjedstva. Vraća (B, N, out_dim).
        """
        Wh = self.W(h)                                     # (B, N, out) projicirane značajke
        # rezultat pažnje za svaki par (izvor, odredište) prije maskiranja
        e_src = (Wh * self.a_src).sum(-1, keepdim=True)
        e_dst = (Wh * self.a_dst).sum(-1, keepdim=True)
        e = F.leaky_relu(e_src + e_dst.transpose(1, 2), 0.2)
        # maskiranje: paru koji NISU susjedi dajemo -beskonačno da nakon softmaxa
        # dobiju težinu 0, tj. čvor ne smije gledati nesusjede
        e = e.masked_fill(~adj_mask.unsqueeze(0), float("-inf"))
        alpha = torch.softmax(e, dim=-1)                   # normalizirane težine pažnje (zbroj = 1)
        # bmm: svaki čvor dobije težinsku sumu značajki svojih susjeda
        return F.elu(torch.bmm(alpha, Wh))


class STGAT(nn.Module):
    """Prostorno-vremenski model: dva GAT sloja po prostoru + GRU po vremenu.

    Parametri: in_dim (ulazne komponente, 2), gat_dim (širina GAT slojeva),
    hidden (skriveno stanje GRU-a), out_dim (izlaz, 2: u i v), residual
    (residual pristup, kao kod LSTM-a). Vraća predikciju polja struja za
    sljedeći dan.
    """

    def __init__(self, in_dim=2, gat_dim=48, hidden=64, out_dim=2, residual=True):
        super().__init__()
        self.residual = residual
        self.gat1 = GATLayer(in_dim, gat_dim)              # prostorni sloj 1
        self.gat2 = GATLayer(gat_dim, gat_dim)             # prostorni sloj 2 (dublje susjedstvo)
        self.gru = nn.GRU(gat_dim, hidden, batch_first=True)   # vremenska agregacija
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, adj):
        """x: (B, seq, N, 2), adj: (N, N) susjedstvo. Vraća (B, N, 2)."""
        B, S, N, Fin = x.shape                              # x: (B, seq, N, 2)
        base = x[:, -1, :, :2]                              # zadnji dan = osnova za residual (perzistencija)
        spat = []
        # Za SVAKI dan u prozoru odvojeno propustimo stanje mora kroz dva GAT
        # sloja pa svaka točka skupi informaciju od (susjeda susjeda) svojih susjeda.
        for t in range(S):                                  # graf-pažnja po koraku (danu)
            h = self.gat2(self.gat1(x[:, t], adj), adj)
            spat.append(h)
        spat = torch.stack(spat, dim=1)                     # (B, S, N, gat_dim)
        # presložimo tako da GRU obrađuje svaku točku kao sekvenciju od S dana
        seq = spat.permute(0, 2, 1, 3).reshape(B * N, S, -1)
        out, _ = self.gru(seq)                              # agregacija po vremenu
        # izlaz zadnjeg koraka -> delta (korekcija) po točki
        delta = self.head(out[:, -1, :]).view(B, N, -1)
        return base + delta if self.residual else delta
