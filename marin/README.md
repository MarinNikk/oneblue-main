# Predikcija morskih struja Jadrana

Radni prostor s kodom za prikupljanje, obradu i predikciju polja morskih struja
Jadrana. Podaci dolaze iz Copernicus Marine servisa (komponente brzine u i v na
mreži cijelog Jadrana), a cilj je procijeniti buduće stanje struja i usporediti
više pristupa pod istim, poštenim uvjetima vrednovanja.

Kod ovdje uvozi zajedničke dijelove iz `src/` u korijenu projekta (učitavanje
podataka, temeljni modeli) i dodaje vlastite implementacije, eksperimente i
primjenu. Mapa `src/` i sirovi podaci u `data/raw/` se ne diraju.

## Struktura

```
marin/
  recreate/            # vlastite reimplementacije modela predikcije (od nule)
  experiments/         # samostalne skripte za vrednovanje temeljnih modela
  copernicus_vs_insitu/  # provjera Copernicus podataka naspram terenskih mjerenja
  ros/                 # objava polja struja na ROS 2 (RViz2 vizualizacija)
  results/             # izlazi: metrike (JSON) i slike (PNG) iz eksperimenata
```

### recreate/
Čiste, vlastite reimplementacije modela za predikciju jednodnevnog polja struja,
jedna datoteka po ulozi:
| Datoteka | Uloga |
|---|---|
| `data.py` | učitavanje NetCDF-a, prorjeđivanje mreže, maska mora, kronološka podjela, normalizacija, 7-dnevni prozori |
| `metrics.py` | mjere pogreške R2, RMSE, MAE i vještina (skill) nad perzistencijom, zasebno za u i v |
| `baseline.py` | perzistencija ("sutra = danas"), mjerilo koje napredni model mora nadmašiti |
| `lstm_model.py` | LSTM po točkama (PointLSTM), uči samo vremensku dinamiku |
| `gnn_model.py` | graf mreža s pažnjom (ST-GAT), more kao graf povezanih točaka |
| `engine.py` | treniranje s ranim zaustavljanjem, predikcija, mjerenje vremena |
| `train.py` | orkestrator: pokreće sve modele i crta usporedne grafove |

Pokretanje (traje oko 20 min na procesoru):
```bash
cd marin/recreate
../../.venv/bin/python train.py
```

### experiments/
Skripte koje vrednuju temeljne (netrenirane) modele na stvarnim podacima.
| Datoteka | Što radi |
|---|---|
| `01_baseline_reproduction.py` | reprodukcija n-dnevnih prosjeka, provjera da je perzistencija najjača |
| `02_baseline_full_metrics.py` | sve 4 mjere po godišnjim dobima i regijama (sjever/centar/jug), s opcijom uklanjanja talijanske strane |
| `03_learned_full_metrics.py` | kostur za isto vrednovanje naučenih modela (dijelovi još nisu dovršeni) |
| `04_model_comparison_table.py` | iz `recreate_comparison.json` složi tablicu usporedbe modela sa svim mjerama (R2, RMSE, MAE, Skill), kao markdown i kao sliku |

```bash
cd marin/experiments
../../.venv/bin/python 01_baseline_reproduction.py
```

### copernicus_vs_insitu/
Provjera koliko su Copernicus modelski podaci pouzdani u odnosu na neovisna
obalna mjerenja (temperatura i salinitet). Detalji u `copernicus_vs_insitu/README.md`.

### ros/
Objava polja struja na dvije standardne ROS 2 teme, spremne za prikaz u RViz2.
Detalji u `ros/README.md`.

## Okruženje

Python 3.13 virtualno okruženje (`.venv`) u korijenu projekta, napravljeno alatom
`uv`. Torch je verzija samo za procesor (bez GPU-a na ovom računalu).

```bash
cd /home/mnikolic/Documents/oneblue-main
source .venv/bin/activate          # ili prefiks .venv/bin/ ispred naredbi
```

## Podaci

Podaci nisu u repozitoriju (izuzeti iz gita). Kredencijali za Copernicus su u
`secrets.yaml`. Preuzimanje u `data/raw/`:

```bash
python -m src.main copernicus test-download     # provjera kredencijala
python -m src.main copernicus download-specific <dataset_id> <variable>
```

Većina modela očekuje detajdirane datoteke struja:
`data/raw/adriatic_currents_uo_detided_*.nc` i `..._vo_detided_*.nc`.

## Rezultati

Izlazi eksperimenata su u `results/`: metrike kao JSON i markdown tablice te
slike (usporedbe modela, sezonske i regionalne analize, validacija Copernicusa).
