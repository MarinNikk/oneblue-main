# Usporedba modela: sve mjere (test skup)

Izvor: `marin/results/recreate_comparison.json`  |  1210 oceanskih točaka  |  115 test dana  |  RMSE i MAE su u m/s.

Sva tri modela vrednovana su pod istim uvjetima (isti test skup, maska i split), pa su izravno usporediva.

| Model | R2 (u) | R2 (v) | RMSE (u) | RMSE (v) | MAE (u) | MAE (v) | Skill |
|---|---|---|---|---|---|---|---|
| Perzistencija | 0,825 | 0,763 | 0,0380 | 0,0469 | 0,0260 | 0,0316 | +0,000 |
| LSTM (po točkama) | 0,842 | 0,793 | 0,0361 | 0,0438 | 0,0247 | 0,0297 | +0,113 |
| ST-GAT (graf mreža) | 0,840 | 0,796 | 0,0364 | 0,0435 | 0,0248 | 0,0295 | +0,115 |

Skill je poboljšanje nad perzistencijom (0 znači jednako kao perzistencija, pozitivno je bolje).

## Referentni modeli (samo R2, iz zasebnih pokretanja)

Za ove modele RMSE i MAE nisu zasebno računati u ovom pokretanju.

| Model | R2 (u) | R2 (v) | Napomena |
|---|---|---|---|
| RBF interpolacija | ≈ 1,0 | ≈ 1,0 | prostorna interpolacija (lakši zadatak, nije izravno usporediva s predikcijom) |
| RBF-LSTM (rekonstrukcija) | ≈ 0,5 | ≈ 0,5 | koeficijenti se dobro predvide, ali rekonstrukcija polja gubi točnost |
| ARIMA (jedna točka) | ≈ 0,68 | - | klasična vremenska serija po jednoj točki |
