#  Disaster Tweet Classification

Ce projet vise à développer un modèle de Machine Learning capable de **classer automatiquement des tweets** comme étant :
-  **Liés à une catastrophe (disaster)**  
-  **Non liés à une catastrophe (non-disaster)**

---

## 📁 Structure du projet

```bash
├── src/
│   └── endpoints/
│       └── app.py               # API FastAPI pour servir le modèle
│
├── notebooks/
│   └── exploration_train.ipynb # Analyse exploratoire et entraînement du modèle
│
├── models/
│   ├── model.pkl               # Pipeline ML (TF-IDF + Classifier)
│   ├── tfidf.pkl               # TF-IDF vectorizer
│   ├── le_location.pkl         # LabelEncoder pour la colonne 'location'
│   └── le_keyword.pkl          # LabelEncoder pour la colonne 'keyword'
│
├── data/
│   └── tweet/
│       ├── train.csv           # Données d'entraînement
│       └── test.csv            # Données de test
│
└── README.md                   # Ce fichier
