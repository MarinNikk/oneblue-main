# -*- coding: utf-8 -*-
"""
Ujednačeno vrednovanje TEMELJNIH modela (bez treniranja) sa SVE ČETIRI mjere
(R², RMSE, MAE i vještina nad perzistencijom) za:
  - svaki model: perzistencija (1-dnevni), n-dnevni prosjek (n=2,3,5,7)
  - svako godišnje doba: zima (DJF), proljeće (MAM), ljeto (JJA), jesen (SON)
  - svaki dio Jadrana: sjeverni (>44,5° N), srednji (42-44,5°), južni (<42°)
  - varijantu domene: cijeli Jadran  i  bez talijanske strane (opcijski)

Cilj: izračunati sve metrike za sve scenarije (i za najjednostavnije modele).
NE trenira ništa -> jeftino i ponovljivo. Rezultat: JSON + gotove markdown tablice.

Glavni nalaz: perzistencija (1-dnevni) je najjaca; svaki duzi n-dnevni prosjek ima
NEGATIVNU vjestinu (skill) nad perzistencijom, tj. gori je od "sutra = danas".
Sjeverni Jadran ima najmanju apsolutnu pogresku (MAE), ali ujedno i najnizi R2,
jer je ondje strujanje slabije i manje promjenjivo (malo varijance koju model
moze "objasniti").

Pokretanje:
    ../../.venv/bin/python 02_baseline_full_metrics.py
"""
import json
from pathlib import Path

import numpy as np
import xarray as xr

# Skripta se pokrece kao datoteka, pa rucno racunamo korijen repozitorija
# (dva nivoa iznad: experiments -> marin -> korijen) da bismo slozili apsolutne
# putanje do ulaznih NetCDF datoteka. UO = komponenta u (istok-zapad),
# VO = komponenta v (sjever-jug).
ROOT = Path(__file__).resolve().parents[2]
UO = ROOT / "data/raw/adriatic_currents_uo_detided_2024_2026.nc"
VO = ROOT / "data/raw/adriatic_currents_vo_detided_2024_2026.nc"
OUT = ROOT / "marin/results"
OUT.mkdir(parents=True, exist_ok=True)

# --- "Italija" = mala kocka dolje lijevo (Tirensko/Jonsko more zapadno od
# talijanskog poluotoka, odvojeno od jadranske dijagonale). Dogovoreno s Marinom.
# Taj jugozapadni kut nije Jadran, pa ga mozemo izbaciti iz analize da nam strane
# vode ne iskrivljuju brojke.
def italian_side(lat, lon):
    """Vrati booleovu masku (True tamo gdje je talijanska/zapadna strana).

    Prima 2D polja geografske sirine (lat) i duzine (lon); vraca niz istog
    oblika s True na tockama koje smatramo jugozapadnim kutom izvan Jadrana.
    """
    return (lat < 42.0) & (lon < 14.0)


def load():
    """Ucita obje komponente struje i pripadne koordinate iz NetCDF datoteka.

    Ne prima argumente. Vraca torku (u, v, lat, lon, time) gdje su u i v
    nizovi oblika [vrijeme, sirina, duzina] (float32), a lat/lon/time 1D
    koordinate.
    """
    uo = xr.open_dataset(UO)["uo_detided"]
    vo = xr.open_dataset(VO)["vo_detided"]
    lat = uo["latitude"].values
    lon = uo["longitude"].values
    time = uo["time"].values
    u = uo.values.astype(np.float32)  # [T, LAT, LON]
    v = vo.values.astype(np.float32)
    return u, v, lat, lon, time


def metrics(pred, true, ref_persist):
    """Izracunaj sve 4 mjere pogreske na spljostenim (u+v) nizovima.

    pred = predikcija modela, true = stvarna vrijednost, ref_persist =
    perzistencijska predikcija koja sluzi kao referenca za vjestinu (skill);
    ako je None (slucaj same perzistencije), skill je 0.
    Vraca rjecnik {R2, RMSE, MAE, Skill}.
    """
    # reshape(-1) spljosti sve dimenzije (vrijeme, komponente, tocke) u jedan
    # dugacak 1D niz; racunamo u float64 radi tocnosti.
    p = pred.reshape(-1).astype(np.float64)
    t = true.reshape(-1).astype(np.float64)
    # Maskiramo samo konacne vrijednosti: NaN su tocke kopna ili neispravni dani.
    m = np.isfinite(p) & np.isfinite(t)
    p, t = p[m], t[m]
    mse = np.mean((p - t) ** 2)
    # R2: koliki dio varijance podataka model objasni. 1 je savrseno, 0 znaci
    # "jednako dobro kao pogoditi prosjek", a negativno znaci gore od prosjeka.
    # Koristimo R2 jer su vrijednosti struja blizu nule, pa bi postotne mjere
    # (npr. SMAPE) dijelile s gotovo nulom i postajale nestabilne/besmislene.
    r2 = 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p - t)))
    if ref_persist is None:
        skill = 0.0
    else:
        # Vjestina (skill) nad perzistencijom: 1 - (greska modela / greska
        # perzistencije). Pozitivno = bolji od perzistencije, negativno = gori.
        r = ref_persist.reshape(-1).astype(np.float64)[m]
        mse_ref = np.mean((r - t) ** 2)
        skill = float(1 - mse / mse_ref) if mse_ref > 0 else float("nan")
    return dict(R2=float(r2), RMSE=rmse, MAE=mae, Skill=skill)


def season_of(dt64):
    """Vrati godisnje doba (Zima/Proljece/Ljeto/Jesen) za zadani datum.

    Prima jedan numpy datetime64; iz njega izvlaci mjesec (1-12) i preslika ga
    na sezonu. Zima je DJF (prosinac, sijecanj, veljaca), itd.
    """
    m = dt64.astype("datetime64[M]").astype(int) % 12 + 1
    return {12: "Zima", 1: "Zima", 2: "Zima", 3: "Proljeće", 4: "Proljeće", 5: "Proljeće",
            6: "Ljeto", 7: "Ljeto", 8: "Ljeto", 9: "Jesen", 10: "Jesen", 11: "Jesen"}[int(m)]


def main():
    """Glavni tijek: ucitaj podatke, izgradi predikcije temeljnih modela,
    izracunaj sve 4 mjere po domeni/sezoni/regiji i zapisi JSON + markdown
    tablice. Ne prima argumente i ne vraca nista (nuspojava su datoteke i ispis).
    """
    u, v, lat, lon, time = load()
    T, LAT, LON = u.shape
    # Od 1D koordinata napravimo 2D mrezu (svaka tocka polja dobije svoju
    # sirinu i duzinu), da kasnije mozemo maskirati po poziciji.
    LOND, LATD = np.meshgrid(lon, lat)  # [LAT, LON]

    # Maska mora (ocean): tocka je more samo ako je konacna u SVIM danima i u
    # obje komponente. Tako izbacimo kopno (NaN) i nepouzdane tocke. To je ono
    # sto zovemo "maskiranje kopna".
    ocean = np.isfinite(u).all(0) & np.isfinite(v).all(0)  # [LAT, LON]

    # Podjela na regije po geografskoj sirini: sjever, centar, jug Jadrana.
    region = np.full((LAT, LON), "", dtype=object)
    region[LATD > 44.5] = "Sjeverni"
    region[(LATD <= 44.5) & (LATD >= 42.0)] = "Srednji"
    region[LATD < 42.0] = "Južni"

    italy = italian_side(LATD, LOND)  # [LAT, LON]

    # Slozimo u i v u jedno polje s dimenzijom komponente: [vrijeme, 2, sirina, duzina].
    field = np.stack([u, v], axis=1)  # [T, 2, LAT, LON]
    seasons = np.array([season_of(t) for t in time])

    # Temeljni modeli: za svaki ciljni dan t predvidamo iz proslosti.
    # Broj uz model je velicina prozora n (perzistencija = prozor 1).
    models = {"Perzistencija (1 dan)": 1, "2-dnevni prosjek": 2, "3-dnevni prosjek": 3,
              "5-dnevni prosjek": 5, "7-dnevni prosjek": 7}

    def build_pred(n):
        """Izgradi predikciju n-dnevnog prosjeka za cijelo polje.

        pred[t] = prosjek polja od dana t-n do t-1 (proslih n dana). Valjano je
        tek za t >= n (prije toga nema dovoljno proslosti, ostaje NaN).
        """
        # Kumulativna suma nam daje brz izracun prosjeka prozora: razlika dviju
        # kumulativnih suma je suma tog intervala, pa podijelimo s n.
        pred = np.full_like(field, np.nan)
        cs = np.cumsum(field, axis=0)
        for t in range(n, T):
            pred[t] = (cs[t - 1] - (cs[t - n - 1] if t - n - 1 >= 0 else 0)) / n
        return pred

    # Perzistencija je referenca za vjestinu (skill) svih ostalih modela.
    persist = build_pred(1)

    def subset_mask(domain, season, reg):
        """Sastavi prostornu i vremensku masku za trazeni podskup podataka.

        domain: "cijeli" ili "bez_italije"; season: "sve" ili ime sezone;
        reg: "sve" ili ime regije. Vraca (sm, tmask): prostornu masku tocaka
        [sirina, duzina] i vremensku masku dana [vrijeme].
        """
        sm = ocean.copy()
        if domain == "bez_italije":
            # Izbacimo jugozapadni kut (vidi italian_side).
            sm = sm & (~italy)
        if reg != "sve":
            sm = sm & (region == reg)
        # vremenska maska: ili svi dani, ili samo dani trazene sezone
        if season == "sve":
            tmask = np.ones(T, bool)
        else:
            tmask = seasons == season
        return sm, tmask

    # Cetverostruka petlja prolazi sve kombinacije: domena x model x sezona x
    # regija i za svaku racuna sve 4 mjere. To je "sezonska i regionalna analiza".
    results = {}
    for domain in ("cijeli", "bez_italije"):
        results[domain] = {}
        for mname, n in models.items():
            pred = persist if n == 1 else build_pred(n)
            results[domain][mname] = {}
            for season in ("sve", "Zima", "Proljeće", "Ljeto", "Jesen"):
                results[domain][mname][season] = {}
                for reg in ("sve", "Sjeverni", "Srednji", "Južni"):
                    sm, tmask = subset_mask(domain, season, reg)
                    # Uzimamo samo dane gdje predikcija postoji (t >= n).
                    tvalid = np.arange(T) >= n
                    tsel = tmask & tvalid
                    # Ako nema dana ili tocaka u ovom podskupu, preskoci.
                    if tsel.sum() == 0 or sm.sum() == 0:
                        continue
                    # Izreze: odaberi valjane dane (tsel) pa prostorne tocke (sm).
                    pr = pred[tsel][:, :, sm]
                    tr = field[tsel][:, :, sm]
                    # Za perzistenciju nema reference (skill=0); za ostale je
                    # referenca perzistencija na istom podskupu.
                    rf = None if n == 1 else persist[tsel][:, :, sm]
                    results[domain][mname][season][reg] = metrics(pr, tr, rf)

    # Spremimo sve brojke u JSON (ensure_ascii=False da ostanu nasa slova c, z, s).
    (OUT / "baseline_full_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # ---- markdown tablice ----
    def fmt(d):
        """Formatiraj jedan rjecnik mjera u redak tablice: R2 | RMSE | MAE | Skill."""
        return f"{d['R2']:.3f} | {d['RMSE']:.4f} | {d['MAE']:.4f} | {d['Skill']:+.3f}"

    # Gradimo citljive markdown tablice red po red u listu `lines`.
    lines = []
    lines.append(f"Podaci: {UO.name} / {VO.name}  |  {T} dana  |  oceanskih točaka (cijeli): {int(ocean.sum())}, bez Italije: {int((ocean & ~italy).sum())}\n")
    for domain in ("cijeli", "bez_italije"):
        lines.append(f"\n# DOMENA: {'cijeli Jadran' if domain=='cijeli' else 'bez talijanske strane (granicu potvrditi)'}\n")
        # Tablica A: model x (ukupno + regije), season='sve'
        lines.append("## Sve četiri mjere po modelu i regiji (cijela godina) [R² | RMSE | MAE | Skill]\n")
        lines.append("| Model | Ukupno | Sjeverni | Srednji | Južni |")
        lines.append("|---|---|---|---|---|")
        for m in models:
            row = [m]
            for reg in ("sve", "Sjeverni", "Srednji", "Južni"):
                d = results[domain][m]["sve"].get(reg)
                row.append(fmt(d) if d else "-")
            lines.append("| " + " | ".join(row) + " |")
        # Tablica B: model x sezona (ukupna regija)
        lines.append("\n## Sve četiri mjere po modelu i godišnjem dobu (cijeli Jadran) [R² | RMSE | MAE | Skill]\n")
        lines.append("| Model | Zima | Proljeće | Ljeto | Jesen |")
        lines.append("|---|---|---|---|---|")
        for m in models:
            row = [m]
            for s in ("Zima", "Proljeće", "Ljeto", "Jesen"):
                d = results[domain][m][s].get("sve")
                row.append(fmt(d) if d else "-")
            lines.append("| " + " | ".join(row) + " |")
    (OUT / "baseline_full_metrics_tables.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\n[OK] ->", OUT / "baseline_full_metrics.json")
    print("[OK] ->", OUT / "baseline_full_metrics_tables.md")


if __name__ == "__main__":
    main()
