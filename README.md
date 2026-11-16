# Métriques d'évaluation pour les tâches de NLP

Projet du Master Data & IA (INPT) : présentation des métriques d'évaluation de trois tâches de traitement du langage naturel, avec une démonstration sur de **vrais systèmes** et de **vraies données**.

| Tâche | Système évalué | Données | Métriques | Résultat |
|---|---|---|---|---|
| Classification (détection de spam) | TF-IDF + Régression logistique (entraînée dans le notebook) | SMS Spam Collection, 5 574 SMS | Accuracy, Précision, Rappel, F1 | F1 = 0,896 |
| Traduction anglais → français | MarianMT `opus-mt-en-fr` (Transformer, ONNX int8) | WMT14 newstest2014, 100 phrases | BLEU | BLEU = 38,6 |
| Résumé automatique | Lead-3, TextRank, Aléatoire | CNN/DailyMail, 30 articles | ROUGE-1, ROUGE-2, ROUGE-L | Lead-3 : ROUGE-1 = 29,3 |

## Structure du dépôt

```
projet_nlp/
├── demo_metriques_nlp.ipynb   # le notebook de démonstration (déjà exécuté, résultats visibles)
├── src/
│   ├── donnees.py             # téléchargement et cache des 3 jeux de données
│   └── traduction.py          # traducteur MarianMT (ONNX) : encodeur + décodeur glouton
├── data/                      # données en cache (petits fichiers, la démo marche hors ligne)
├── figures/                   # graphiques produits par le notebook
├── results/metriques.json     # tous les chiffres de la démo
├── presentation/              # diapositives (.pptx) et script oral
└── requirements.txt
```

## Installation et exécution

Testé avec Python 3.12 sous Ubuntu (WSL).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute --inplace demo_metriques_nlp.ipynb   # ou ouvrir le notebook dans VS Code / Jupyter
```

- Au premier lancement, le modèle de traduction (environ 110 Mo) est téléchargé dans `models/`.
- Les données et les traductions sont déjà dans `data/` : la démo tourne donc sans internet. Pour tout recalculer, supprimer les fichiers de `data/`.
- Durée : environ 10 secondes avec le cache, environ 2 minutes sans (traduction de 100 phrases sur CPU).

## Documentation du code

### `src/donnees.py`

| Fonction | Rôle |
|---|---|
| `charger_sms_spam()` | Télécharge le fichier SMS Spam Collection (UCI) et le renvoie en `DataFrame` (`label`, `texte`). |
| `charger_wmt14(n=100)` | Récupère les `n` premières paires de phrases (`en`, `fr`) du test WMT14 via l'API *datasets-server* de Hugging Face, sans télécharger tout le dataset. |
| `charger_cnn_dailymail(n=30)` | Récupère `n` articles du test CNN/DailyMail avec leur résumé de référence (`article`, `reference`). |

Chaque fonction écrit un fichier dans `data/` et le relit aux appels suivants.

### `src/traduction.py`

Classe `TraducteurMarian` :

1. Télécharge le modèle `Xenova/opus-mt-en-fr`, une version ONNX quantifiée en int8 du modèle `Helsinki-NLP/opus-mt-en-fr`.
2. Découpe la phrase anglaise en sous-mots avec SentencePiece (`source.spm`) et les convertit en identifiants (`vocab.json`).
3. Fait passer la phrase dans l'**encodeur** du Transformer.
4. Le **décodeur** génère la phrase française jeton par jeton. À chaque étape, on prend le jeton le plus probable (décodage glouton) jusqu'au jeton de fin.
5. Reconstruit le texte français avec `target.spm`.

```python
from src.traduction import TraducteurMarian
TraducteurMarian().traduire("The cat is sleeping on the sofa.")   # 'Le chat dort sur le canapé.'
```

### `demo_metriques_nlp.ipynb`

**1. Classification**
- Découpage stratifié 80/20 (`random_state=42`), puis entraînement de `TfidfVectorizer` + `LogisticRegression`.
- Calcul **à la main** de VP, FP, FN, VN et des 4 métriques, puis vérification avec `sklearn.metrics`.
- Comparaison avec un classifieur naïf « toujours normal » : son accuracy est de 86,6 %, mais son rappel est de 0. Cela montre le piège de l'accuracy.
- Effet du seuil de décision (0,2 / 0,3 / 0,5 / 0,7) sur la précision et le rappel.

**2. Traduction**
- Exemple simple de BLEU (`sacrebleu.sentence_bleu`) : phrase identique, un mot changé, synonymes, hors sujet.
- Traduction de 100 phrases WMT14 avec MarianMT.
- BLEU du corpus (`sacrebleu.corpus_bleu`) : précisions 1 à 4-grammes et pénalité de brièveté.
- BLEU phrase par phrase : histogramme, meilleures et moins bonnes phrases.

**3. Résumé**
- Exemple simple de ROUGE (`rouge_score`, avec stemming).
- Trois systèmes extractifs de 3 phrases : Aléatoire, Lead-3 et TextRank (`sumy`).
- F1 moyen de ROUGE-1, ROUGE-2 et ROUGE-L sur 30 articles, avec un graphique et un exemple commenté.

**4. Bilan** : tableau récapitulatif et export de `results/metriques.json`.

## Sources

- SMS Spam Collection : Almeida, Gómez Hidalgo, Yamakami (2011), UCI Machine Learning Repository.
- WMT14 : *Findings of the 2014 Workshop on Statistical Machine Translation*.
- CNN/DailyMail : Hermann et al. (2015), See et al. (2017).
- BLEU : Papineni et al. (2002). ROUGE : Lin (2004). TextRank : Mihalcea & Tarau (2004).
- MarianMT / OPUS-MT : Tiedemann & Thottingal (2020).
