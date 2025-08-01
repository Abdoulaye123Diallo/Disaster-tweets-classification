#  Disaster Tweet Classification

Ce projet vise à développer un modèle de Machine Learning capable de **classer automatiquement des tweets** comme étant :
-  **Liés à une catastrophe (disaster)**  
-  **Non liés à une catastrophe (non-disaster)**




#### Membres du groupe : Abdoulaye Diallo, Aissatou Kany Djogope Mbodje, Mouhamad Samba S Traoré
---

## 📁 Structure du projet

```bash
├── src/
│   └── endpoints/
│       └── app.py               # API Flask pour servir le modèle
│
├── notebooks/
│   └── NLP-with-disaster-tweets-classifiaction.ipynb # Analyse exploratoire et entraînement du modèle
│
├── models/
│   ├── model.pkl               # Modele ML (Classifier)
│   ├── tfidf.pkl               # TF-IDF vectorizer
│   ├── le_location.pkl         # LabelEncoder pour la colonne 'location'
│   └── le_keyword.pkl          # LabelEncoder pour la colonne 'keyword'
│
├── data/
│   └── tweets/
│       ├── train.csv           # Données d'entraînement
│       └── test.csv            # Données de test
│
└── README.md                   # Ce fichier



