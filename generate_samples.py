# -*- coding: utf-8 -*-
# generate_samples.py
# Génère des samples audio synthétiques de "Alissa" pour l'entraînement OWW
# Utilise gTTS (Google TTS) avec variations de vitesse/pitch pour la diversité

import os
import numpy as np
import soundfile as sf
from gtts import gTTS
from io import BytesIO
import librosa

OUTPUT_DIR = "samples/positive"
NEGATIVE_DIR = "samples/negative"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(NEGATIVE_DIR, exist_ok=True)

SAMPLE_RATE = 16000

# Variantes orthographiques et phonétiques d'"Alissa"
# Plus de variantes = meilleure généralisation du modèle
VARIANTS = [
    "Alissa",
    "Alyssa", 
    "Alisa",
    "Alysa",
    "Élissa",
    "Alissa,",      # avec pause naturelle
    "Alissa ?",     # intonation montante
    "Alissa !",     # intonation forte
    "Hey Alissa",
    "Dis Alissa",
    "Ok Alissa",
]

# Langues TTS pour varier les accents
LANGS = ["fr", "fr-ca", "fr-be"]

def generate_tts(text: str, lang: str = "fr") -> np.ndarray:
    """Génère un sample audio via gTTS et retourne float32 16kHz."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio, sr = librosa.load(buf, sr=SAMPLE_RATE, mono=True)
        return audio
    except Exception as e:
        print(f"  [WARN] gTTS failed ({text}, {lang}): {e}")
        return None


def augment(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list:
    """Génère des variantes augmentées d'un sample audio."""
    variants = [audio]  # original
    
    # Pitch shift ±2 demi-tons
    try:
        variants.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))
        variants.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2))
    except Exception:
        pass
    
    # Time stretch ±15%
    try:
        variants.append(librosa.effects.time_stretch(audio, rate=1.15))
        variants.append(librosa.effects.time_stretch(audio, rate=0.85))
    except Exception:
        pass
    
    # Ajout de bruit blanc léger
    noise = np.random.randn(len(audio)) * 0.005
    variants.append(audio + noise)
    
    return variants


count = 0
print(f"[GEN] Génération des samples positifs...")

for variant in VARIANTS:
    for lang in LANGS:
        audio = generate_tts(variant, lang)
        if audio is None:
            continue
        
        augmented = augment(audio)
        for i, aug in enumerate(augmented):
            # Normaliser
            aug = aug / (np.max(np.abs(aug)) + 1e-8)
            # Padder à 1.5s minimum
            target_len = int(SAMPLE_RATE * 1.5)
            if len(aug) < target_len:
                aug = np.pad(aug, (0, target_len - len(aug)))
            
            filename = f"{OUTPUT_DIR}/alissa_{count:04d}.wav"
            sf.write(filename, aug, SAMPLE_RATE)
            count += 1

print(f"[GEN] {count} samples positifs générés dans {OUTPUT_DIR}/")

# Samples négatifs : mots phonétiquement proches mais différents
NEGATIVE_WORDS = [
    "Alice", "Alicia", "Mélissa", "Clarissa", "Larissa",
    "Vanessa", "Theresa", "Amanda", "Sandra", "Lisa",
    "bonjour", "merci", "oui", "non", "voilà",
    "allumer", "éteindre", "température", "musique",
]

neg_count = 0
print(f"[GEN] Génération des samples négatifs...")

for word in NEGATIVE_WORDS:
    for lang in ["fr"]:
        audio = generate_tts(word, lang)
        if audio is None:
            continue
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        target_len = int(SAMPLE_RATE * 1.5)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        filename = f"{NEGATIVE_DIR}/neg_{neg_count:04d}.wav"
        sf.write(filename, audio, SAMPLE_RATE)
        neg_count += 1

print(f"[GEN] {neg_count} samples négatifs générés dans {NEGATIVE_DIR}/")
print(f"[GEN] Total : {count} positifs + {neg_count} négatifs")
