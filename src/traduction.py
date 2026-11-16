"""Traduction anglais -> français avec MarianMT (Helsinki-NLP/opus-mt-en-fr).

On utilise la version compressée (ONNX, quantifiée en int8) publiée par Xenova :
environ 110 Mo au lieu de 300 Mo, et pas besoin de PyTorch.

Utilisation :
    from src.traduction import TraducteurMarian
    t = TraducteurMarian()
    t.traduire("The cat is sleeping.")   # -> "Le chat dort."
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import hf_hub_download

REPO = "Xenova/opus-mt-en-fr"
FICHIERS = [
    "config.json",
    "vocab.json",
    "source.spm",
    "target.spm",
    "onnx/encoder_model_quantized.onnx",
    "onnx/decoder_model_quantized.onnx",
]


class TraducteurMarian:
    def __init__(self, dossier="models/opus-mt-en-fr"):
        # 1) Téléchargement (une seule fois, ensuite le cache local est utilisé)
        p = {f: hf_hub_download(REPO, f, local_dir=dossier) for f in FICHIERS}

        # 2) Configuration et vocabulaire
        self.cfg = json.loads(Path(p["config.json"]).read_text())
        self.vocab = json.loads(Path(p["vocab.json"]).read_text())
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

        # 3) Tokenizers SentencePiece : un pour l'anglais, un pour le français
        self.sp_src = spm.SentencePieceProcessor(model_file=p["source.spm"])
        self.sp_tgt = spm.SentencePieceProcessor(model_file=p["target.spm"])

        # 4) Le réseau de neurones : un encodeur et un décodeur (Transformer)
        self.encodeur = ort.InferenceSession(p["onnx/encoder_model_quantized.onnx"])
        self.decodeur = ort.InferenceSession(p["onnx/decoder_model_quantized.onnx"])

    def traduire(self, phrase, max_tokens=256):
        eos, pad = self.cfg["eos_token_id"], self.cfg["pad_token_id"]

        # Phrase anglaise -> morceaux de mots -> numéros (ids)
        morceaux = self.sp_src.encode(phrase, out_type=str)
        ids = [self.vocab.get(m, self.vocab["<unk>"]) for m in morceaux] + [eos]
        entree = np.array([ids], dtype=np.int64)
        masque = np.ones_like(entree)

        # L'encodeur lit toute la phrase anglaise
        etats = self.encodeur.run(None, {"input_ids": entree, "attention_mask": masque})[0]

        # Le décodeur écrit la phrase française mot par mot (décodage glouton :
        # à chaque étape on prend le mot le plus probable)
        sortie = [self.cfg["decoder_start_token_id"]]
        for _ in range(max_tokens):
            logits = self.decodeur.run(None, {
                "input_ids": np.array([sortie], dtype=np.int64),
                "encoder_hidden_states": etats,
                "encoder_attention_mask": masque,
            })[0]
            logits[0, -1, pad] = -1e9          # on interdit le jeton de remplissage
            suivant = int(logits[0, -1].argmax())
            if suivant == eos:
                break
            sortie.append(suivant)

        # Numéros -> morceaux -> phrase française
        return self.sp_tgt.decode_pieces([self.inv_vocab[i] for i in sortie[1:]])
