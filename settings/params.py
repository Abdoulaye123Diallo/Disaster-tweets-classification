# settings/params.py

# Seed pour la reproductibilité
SEED = 42

# Paramètres des modèles
MODEL_PARAMS = {
  'TARGET_NAME': 'target',
  'MIN_COMPLETION_RATE': 0.75,
  'MIN_PPS': 0.1,
  'DEFAULT_FEATURE_NAMES': ['Alley',
  'BsmtQual',
  'ExterQual',
  'Foundation',
  'FullBath',
  'GarageArea',
  'GarageCars',
  'GarageFinish',
  'GarageType',
  'GrLivArea',
  'KitchenQualMSSubClass',
  'Neighborhood',
  'OverallQual',
  'TotRmsAbvGrd',
  'building_age',
  'remodel_age',
  'garage_age'],
 'TEST_SIZE': 0.25
 }

# Paramètres d'entraînement
TRAIN_PARAMS = {
    'test_size': 0.2,
    'random_state': SEED,
    'stratify': True
}