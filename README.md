# alissa-wakeword

Entraînement du modèle OpenWakeWord pour le wake word "Alissa" (assistant vocal personnel).

## Structure

```
.github/workflows/train.yml   → workflow GitHub Actions
generate_samples.py           → génération des samples synthétiques
train.py                      → entraînement du modèle
requirements.txt              → dépendances Python
```

## Utilisation

1. Pusher ce repo sur GitHub (compte : Fredmc5754)
2. GitHub Actions lance automatiquement l'entraînement
3. Récupérer `alissa.onnx` dans les artefacts du workflow
4. Copier dans `E:\alissa\wakewords\alissa.onnx`

## Relancer l'entraînement

Aller dans **Actions** → **Train Alissa Wake Word** → **Run workflow**

## Paramètres

Le modèle est entraîné sur des variantes synthétiques de "Alissa" générées via gTTS
avec augmentation (pitch shift, time stretch, bruit blanc) pour améliorer la robustesse.
