#  Disaster Tweet Classification

Ce projet vise à développer un modèle de Machine Learning capable de **classer automatiquement des tweets** comme étant :
-  **Liés à une catastrophe (disaster)**  
-  **Non liés à une catastrophe (non-disaster)**




#### Membres du groupe : Abdoulaye Diallo, Aissatou Kany Djogope Mbodje, Mouhamad Samba S Traoré
---

##  Endpoints de l'API

- `POST /predict`  
  ➤ Prédiction simple à partir d’un tweet, d’une localisation et d’un mot-clé.

- `POST /predict_batch`  
  ➤ Prédictions multiples (batch) à partir de plusieurs tweets, localisations et mots-clés.

- `GET /health`  
  ➤ Vérifie si le service est opérationnel (health check).

- `GET /info`  
  ➤ Fournit des informations sur l’API (description, version, etc.).

---

## 📦 Exemples d'utilisation avec `curl`

###  Prediction simple

curl -X POST https://disaster-tweets-classification.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Forest fire spreading rapidly", "location": "California", "keyword": "fire"}'


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



