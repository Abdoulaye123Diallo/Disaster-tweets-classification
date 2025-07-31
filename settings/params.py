# settings/params.py

# Seed pour la reproductibilité
SEED = 42

# Paramètres des modèles
MODEL_PARAMS = {
  'TARGET_NAME': 'target',
 }

# Paramètres d'entraînement
TRAIN_PARAMS = {
    'test_size': 0.2,
    'random_state': SEED,
    'stratify': True
}