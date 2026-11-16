"""Chargement des trois jeux de données (téléchargés une fois, puis lus depuis data/)."""
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA = Path("data")
HF_ROWS = "https://datasets-server.huggingface.co/rows"


def _lignes_hf(dataset, config, split, n):
    """Récupère les n premières lignes d'un dataset Hugging Face, sans tout télécharger."""
    lignes = []
    while len(lignes) < n:
        r = requests.get(HF_ROWS, params={"dataset": dataset, "config": config, "split": split,
                                          "offset": len(lignes), "length": min(100, n - len(lignes))}, timeout=60)
        r.raise_for_status()
        lignes += [x["row"] for x in r.json()["rows"]]
    return lignes


def charger_sms_spam():
    """SMS Spam Collection (UCI) : 5 574 SMS étiquetés 'ham' (normal) ou 'spam'."""
    f = DATA / "sms_spam.tsv"
    if not f.exists():
        DATA.mkdir(exist_ok=True)
        r = requests.get("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip", timeout=60)
        f.write_bytes(zipfile.ZipFile(io.BytesIO(r.content)).read("SMSSpamCollection"))
    return pd.read_csv(f, sep="\t", header=None, names=["label", "texte"], quoting=3)


def charger_wmt14(n=100):
    """WMT14 anglais-français (jeu de test newstest2014) : phrases + traduction humaine de référence."""
    f = DATA / f"wmt14_en_fr_test_{n}.json"
    if not f.exists():
        DATA.mkdir(exist_ok=True)
        lignes = _lignes_hf("wmt/wmt14", "fr-en", "test", n)
        f.write_text(json.dumps([r["translation"] for r in lignes], ensure_ascii=False, indent=1), encoding="utf-8")
    return pd.DataFrame(json.loads(f.read_text(encoding="utf-8")))


def charger_cnn_dailymail(n=30):
    """CNN/DailyMail (jeu de test) : articles de presse + résumé écrit par un journaliste."""
    f = DATA / f"cnn_dailymail_test_{n}.json"
    if not f.exists():
        DATA.mkdir(exist_ok=True)
        lignes = _lignes_hf("abisee/cnn_dailymail", "3.0.0", "test", n)
        f.write_text(json.dumps([{"article": r["article"], "reference": r["highlights"]} for r in lignes],
                                ensure_ascii=False, indent=1), encoding="utf-8")
    return pd.DataFrame(json.loads(f.read_text(encoding="utf-8")))
