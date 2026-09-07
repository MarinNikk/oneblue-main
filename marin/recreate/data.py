# -*- coding: utf-8 -*-
"""Priprema podataka za predikciju morskih struja Jadrana.

Ovdje se sve sirove NetCDF datoteke (polja struja po danima) pretvaraju u
tenzore spremne za treniranje modela. Postupak uključuje: učitavanje i
prorjeđivanje mreže, izdvajanje samo morskih čvorova (maskiranje kopna),
gradnju grafa susjedstva, kronološku podjelu na train/val/test, normalizaciju
i tvorbu kliznih prozora od 7 dana. Ostale datoteke (modeli, engine, train)
oslanjaju se na rječnik koji vraća funkcija make_dataset().
"""
from pathlib import Path
import numpy as np
import xarray as xr

# Putanje do sirovih podataka. parents[2] se penje dvije razine iznad ove
# datoteke (marin/recreate/ -> marin/ -> korijen projekta), pa lako dođemo
# do mape data/raw bez obzira odakle se skripta pokreće.
ROOT = Path(__file__).resolve().parents[2]
UO = ROOT / "data/raw/adriatic_currents_uo_detided_2024_2026.nc"  # zonalna (u) komponenta
VO = ROOT / "data/raw/adriatic_currents_vo_detided_2024_2026.nc"  # meridionalna (v) komponenta

SEQ_LEN = 7          # ulazni prozor: model gleda 7 uzastopnih dana i predviđa sljedeći
STRIDE_GRID = 3      # prorjeđivanje mreže: uzimamo svaku 3. točku po širini i visini
SPLIT = (0.70, 0.15, 0.15)   # omjer train / validacija / test


def load_grid():
    """Učitaj polja struja i prorijedi mrežu.

    Vraća četvorku (U, V, lat, lon):
      - U, V: nizovi oblika (vrijeme, redak, stupac) sa zonalnom i meridionalnom
        komponentom brzine struje (m/s),
      - lat, lon: pripadne koordinate mreže.

    Prorjeđivanje (::STRIDE_GRID po obje prostorne osi) smanjuje broj točaka
    otprilike 9 puta. To ubrzava treniranje i smanjuje memoriju, a krupne
    značajke strujnog polja ostaju sačuvane.
    """
    U = xr.open_dataset(UO)["uo_detided"].values[:, ::STRIDE_GRID, ::STRIDE_GRID]
    V = xr.open_dataset(VO)["vo_detided"].values[:, ::STRIDE_GRID, ::STRIDE_GRID]
    uo = xr.open_dataset(UO)["uo_detided"]
    # koordinate moramo prorijediti istim korakom da odgovaraju prorijeđenim poljima
    lat = uo.latitude.values[::STRIDE_GRID]
    lon = uo.longitude.values[::STRIDE_GRID]
    return U, V, lat, lon


def ocean_mask(U):
    """Maska mora: True tamo gdje je točka valjana u SVIM vremenskim trenucima.

    Na mreži se kopno i područja izvan mjerenja zapisuju kao NaN. Točka je
    more samo ako nijednom kroz cijeli niz nije NaN (axis=0 je vremenska os).
    Time maskiramo kopno pa modeli uče i mjere se isključivo nad morem.
    """
    return ~np.isnan(U).any(axis=0)


def build_adjacency(mask):
    """Izgradi graf susjedstva nad morskim čvorovima.

    Prima masku mora (2D bool polje). Vraća:
      - A: (N, N) bool matricu susjedstva, gdje je N broj morskih čvorova,
      - nodes: (N, 2) niz s (redak, stupac) pozicijom svakog čvora na mreži.

    Svaki čvor spaja se sa svojih najviše 8 susjeda (lijevo/desno, gore/dolje
    i dijagonale) te sam sa sobom (petlja, A[n, n] = True). Taj graf koristi
    GNN model da bi morska točka mogla "gledati" svoje okolne točke.
    """
    H, W = mask.shape
    # idx preslikava poziciju na mreži (i, j) u redni broj čvora n; -1 znači kopno
    idx = -np.ones((H, W), dtype=int)
    nodes = np.argwhere(mask)          # koordinate svih morskih točaka
    for n, (i, j) in enumerate(nodes):
        idx[i, j] = n
    N = len(nodes)
    A = np.zeros((N, N), dtype=bool)
    for n, (i, j) in enumerate(nodes):
        A[n, n] = True                 # petlja: čvor je sam sebi susjed
        # prođi kroz 3x3 okolinu oko (i, j)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                # susjed se dodaje samo ako je unutar mreže i ako je more
                if 0 <= ii < H and 0 <= jj < W and mask[ii, jj]:
                    A[n, idx[ii, jj]] = True
    return A, nodes


def make_dataset():
    """Složi cijeli skup podataka i vrati rječnik tenzora za treniranje.

    Ovo je glavna ulazna točka modula. Redom: učita i prorijedi mrežu, napravi
    masku mora i graf susjedstva, izdvoji samo morske čvorove, podijeli podatke
    kronološki, normalizira ih i izreže klizne prozore.

    Vraća rječnik s ključevima:
      Xtr/Ytr, Xva/Yva, Xte/Yte  -> ulazi i ciljevi za train/val/test,
      adjacency                  -> matrica susjedstva grafa,
      mean, std                  -> statistike normalizacije (po komponenti),
      N, nodes                   -> broj i pozicije morskih čvorova,
      lat, lon, mask             -> geometrija mreže i maska mora.
    """
    U, V, lat, lon = load_grid()
    mask = ocean_mask(U)
    A, nodes = build_adjacency(mask)
    N = len(nodes)

    # iz 2D polja izvlačimo samo morske točke u 1D niz čvorova
    rows, cols = nodes[:, 0], nodes[:, 1]
    Un = U[:, rows, cols]
    Vn = V[:, rows, cols]
    # spojimo u i v u zadnju os: svaka točka i svaki dan ima 2 značajke (u, v)
    data = np.stack([Un, Vn], axis=-1).astype(np.float32)   # (T, N, 2)
    T = data.shape[0]

    # Kronološka podjela: prvih 70% dana je train, sljedećih 15% val, zadnjih
    # 15% test. Dijelimo po vremenu (a ne nasumično) jer je ovo predikcija
    # budućnosti, pa model nikad ne smije "vidjeti" kasnije dane dok uči;
    # time izbjegavamo curenje informacija iz budućnosti u prošlost.
    n_tr = int(T * SPLIT[0])
    n_va = int(T * (SPLIT[0] + SPLIT[1]))

    # Normalizacija (oduzmi prosjek, podijeli standardnom devijacijom) dovodi u
    # i v na sličan raspon pa treniranje bolje konvergira. Statistike računamo
    # SAMO iz treninga da test ostane nepoznat (opet: bez curenja podataka).
    mean = data[:n_tr].reshape(-1, 2).mean(0)
    std = data[:n_tr].reshape(-1, 2).std(0)
    norm = (data - mean) / std

    def windows(t0, t1):
        """Izreži klizne prozore čiji ciljani (zadnji) dan pada u [t0, t1).

        Za svaki dan t uzimamo prethodnih SEQ_LEN dana kao ulaz X, a sam dan t
        kao cilj Y. Time model iz 7 uzastopnih dana predviđa sljedeći dan.
        """
        X, Y = [], []
        # krećemo od max(SEQ_LEN, t0) da uvijek ima punih 7 dana ispred prozora
        for t in range(max(SEQ_LEN, t0), t1):
            X.append(norm[t - SEQ_LEN:t])   # ulaz: dani t-7 .. t-1
            Y.append(norm[t])               # cilj: dan t
        return np.asarray(X), np.asarray(Y)

    Xtr, Ytr = windows(0, n_tr)
    Xva, Yva = windows(n_tr, n_va)
    Xte, Yte = windows(n_va, T)

    return dict(
        Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte,
        adjacency=A, mean=mean, std=std, N=N, nodes=nodes,
        lat=lat, lon=lon, mask=mask,
    )


if __name__ == "__main__":
    # Brza provjera: ispiši osnovne dimenzije skupa kad se datoteka pokrene sama.
    d = make_dataset()
    print(f"čvorova (N): {d['N']}")
    print(f"bridova: {int(d['adjacency'].sum())} (uklj. petlje)")
    print(f"train/val/test prozori: {len(d['Xtr'])}/{len(d['Xva'])}/{len(d['Xte'])}")
    print(f"X oblik: {d['Xtr'].shape}  Y oblik: {d['Ytr'].shape}")
