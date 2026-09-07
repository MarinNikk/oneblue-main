# Copernicus naspram in-situ mjerenja (temperatura i salinitet)

Usporedba Copernicus reanalize s terenskim mjerenjima na dvije obalne regije.
Sve je u `marin/`, originalni podaci u `data/raw/` i kod u `src/` nisu dirani.

## Zašto temperatura i salinitet (a ne struje)
Naši Copernicus NetCDF-ovi za Jadran sadrže **samo struje (u/v)**, a dostupna
mjerenja (Cipar, Katalonska obala) **nemaju izmjerene struje**, imaju
temperaturu, salinitet, kisik, pH, klorofil, nutrijente. Jedine varijable
zajedničke objema stranama su **temperatura i salinitet**, pa se usporedba radi
preko njih (i za druge Copernicus varijable, ne za jadranske struje).

## Podaci
- **Mjerenja:** `DFMR measurements.xlsx` (Cipar, 2006-2025) i
  `20260213_Annex Catalan Coast.xlsx` (Katalonija, 1994-2025); površinski uzorci.
- **Copernicus:** reanaliza **MEDSEA_MULTIYEAR_PHY_006_004**, mjesečno, površinski
  sloj (~1 m), izrezano na obje regije, 1994-2022. Varijable `thetao` (temp), `so` (sal).

## Postupak
1. `insitu.py`: učita i očisti oba skupa u zajednički oblik (UTM u lat/lon za
   Kataloniju, ukloni fizikalno nemoguće vrijednosti, zadrži površinu).
2. `download.py`: preuzme Copernicus mjesečnu temp/sal za granice svake regije.
3. `compare.py`: svaki uzorak spoji s najbližom Copernicus ćelijom i mjesecom
   (kopno = NaN se izbaci), izračuna bias/RMSE/MAE/r i nacrta slike.

## Rezultati (1994-2022, površina)

| Regija / varijabla | N | bias | RMSE | r |
|---|---|---|---|---|
| Cipar · temperatura | 248 | +0,06 | 1,81 °C | 0,89 |
| Katalonija · temperatura | 7368 | -0,38 | 1,73 °C | 0,94 |
| Cipar · salinitet | 234 | +1,02 | 1,50 psu | 0,05 |
| Katalonija · salinitet | 6547 | +0,99 | 2,49 psu | 0,19 |

## Zaključak
- **Temperatura: vrlo dobro slaganje** u obje regije (r 0,89-0,94, bias ~0,
  RMSE < 1,9 °C). Sezonski ciklus se gotovo preklapa, reanaliza pouzdano
  reproducira površinsku temperaturu istočnog i zapadnog Sredozemlja.
- **Salinitet: sustavan pozitivan bias (~+1 psu) i slaba korelacija.** Model
  drži gotovo konstantnu otvorenomorsku vrijednost (~37,5-39,5 psu), dok
  mjerenja pokazuju niži i promjenjiviji priobalni salinitet. Uzrok: mreža od
  4,2 km **ne razlučuje priobalno osvježavanje** (riječni/obalni dotok slatke
  vode), pa promašuje lokalne epizode niskog saliniteta.
- **Pouka:** za priobalnu primjenu Copernicus je pouzdan za temperaturu, ali za
  salinitet blizu obale treba oprez / lokalna korekcija.

## Pokretanje
```bash
cd marin/copernicus_vs_insitu
../../.venv/bin/copernicusmarine login     # jednom (kredencijali)
../../.venv/bin/python download.py         # preuzme 4 NetCDF-a (~27 MB)
../../.venv/bin/python compare.py          # tablica + slike u marin/results/
```
Izlaz u `marin/results/`: `copernicus_vs_insitu.json`, `copernicus_scatter.png`,
`copernicus_seasonal.png`, `copernicus_matched.parquet`.
