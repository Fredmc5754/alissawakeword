# -*- coding: utf-8 -*-
# train.py
# Entraîne le modèle OpenWakeWord pour "Alissa"

import os
import glob
import numpy as np
import soundfile as sf
from openwakeword.train import train_model

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 16000
MODEL_NAME  = "alissa"

def load_wavs(pattern: str) -> list:
    files = glob.glob(pattern)
    audios = []
    for f in files:
        try:
            audio, sr = sf.read(f, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            audios.append(audio)
        except Exception as e:
            print(f"[WARN] Impossible de charger {f}: {e}")
    return audios

print("[TRAIN] Chargement des samples...")
positive_samples = load_wavs("samples/positive/*.wav")
negative_samples = load_wavs("samples/negative/*.wav")

print(f"[TRAIN] {len(positive_samples)} positifs / {len(negative_samples)} négatifs")

if len(positive_samples) < 10:
    raise RuntimeError(f"Pas assez de samples positifs ({len(positive_samples)} < 10)")

print("[TRAIN] Entraînement en cours...")

train_model(
    model_name=MODEL_NAME,
    positive_samples=positive_samples,
    negative_samples=negative_samples,
    output_dir=OUTPUT_DIR,
    target_accuracy=0.85,
    max_steps=5000,
    sample_rate=SAMPLE_RATE,
)

output_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.onnx")
if os.path.isfile(output_path):
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[TRAIN] Modele sauvegarde : {output_path} ({size_kb:.0f} Ko)")
else:
    raise RuntimeError(f"Modele non trouve apres entrainement : {output_path}")
