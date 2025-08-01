from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re
import string
from scipy.sparse import hstack, csr_matrix
import os
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)
MODEL_TYPE = "custom"  # ou "pipeline"

# Variables globales pour stocker les modeles
model_pipeline = None
model = None
tfidf = None
le_loc = None
le_kw = None


def load_models():
    """Charge les modeles selon le type configuré"""
    global model_pipeline, model, tfidf, le_loc, le_kw
    
    if MODEL_TYPE == "pipeline": # not used
        try:
            model_pipeline = joblib.load('disaster_tweet_pipeline.pkl')
            print("Pipeline chargee avec succes")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement de la pipeline: {e}")
            return False
    
    elif MODEL_TYPE == "custom":
        try:
            #model = joblib.load('/Users/sambastraore/Downloads/dossier-mlops/model.pkl')
            #tfidf = joblib.load('/Users/sambastraore/Downloads/tfidf.pkl')
            #le_loc = joblib.load('/Users/sambastraore/Downloads/dossier-mlops/le_location.pkl')
            #le_kw = joblib.load('/Users/sambastraore/Downloads/dossier-mlops/le_keyword.pkl')
            #print("Modèle et objets personnalisés chargés avec succès") 



            model = joblib.load("././models/model.pkl")
            tfidf = joblib.load("././models/tfidf.pkl")
            le_loc = joblib.load("././models/le_location.pkl")
            le_kw = joblib.load("././models/le_keyword.pkl")


            return True
        except Exception as e:
            print(f"Erreur lors du chargement du modele ou des objets: {e}")
            return False
    
    return False

def clean_text(text):
    """Nettoie le texte d'entree"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+|[^a-z\s]', ' ', text)
    return ' '.join(text.split())

def safe_label_encode(encoder, value, default_value="unknown"):
    """Encode une valeur avec gestion des valeurs inconnues"""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        try:
            return encoder.transform([default_value])[0]
        except ValueError:
            # Si meme la valeur par défaut n existe pas, retourner 0
            return 0

def extract_features_from_json(json_input):
    """Extrait les features à partir du JSON d'entrée"""
    df = pd.DataFrame([json_input])
    
    # Texte nettoye
    df['clean_text'] = df['text'].apply(clean_text)
    # Features numeriques
    df['url_count'] = df['text'].str.count(r'http\S+|www\S+')
    df['text_length'] = df['text'].str.len()
    df['mention_count'] = df['text'].str.count(r'@\w+')
    df['question_count'] = df['text'].str.count(r'\?')
    df['caps_count'] = df['text'].str.count(r'[A-Z]')
    df['exclamation_count'] = df['text'].str.count(r'!')
    df['hashtag_count'] = df['text'].str.count(r'#\w+')
    df['word_count'] = df['text'].str.split().str.len()
    df['urgency_score'] = df['url_count'] + df['caps_count']
    df['social_score'] = df['mention_count'] + df['hashtag_count']

    # Encodage des colonnes categorielles avec gestion des valeurs inconnues
    df['location_encoded'] = safe_label_encode(le_loc, json_input.get('location', ''))
    df['keyword_encoded'] = safe_label_encode(le_kw, json_input.get('keyword', ''))

    return df

def predict_single_text(text, location=None, keyword=None):
    """Fonction unifiée pour prédire un seul texte"""
    if MODEL_TYPE == "pipeline":
        if model_pipeline is None:
            raise Exception("Pipeline non chargée")
        
        prediction = model_pipeline.predict([text])[0]
        prediction_proba = model_pipeline.predict_proba([text])[0]
        
        return {
            'prediction': int(prediction),
            'prediction_label': 'disaster' if prediction == 1 else 'not_disaster',
            'confidence': {
                'not_disaster': float(prediction_proba[0]),
                'disaster': float(prediction_proba[1])
            },
            'confidence_score': float(max(prediction_proba))
        }
    
    elif MODEL_TYPE == "custom":
        if model is None or tfidf is None:
            raise Exception("Modele custom non charge")
        
        # Creer un objet JSON pour extract_features_from_json
        json_input = {
            'text': text,
            'location': location or '',
            'keyword': keyword or ''
        }
        
        # Extraction des features
        df = extract_features_from_json(json_input)

        #print("test")
        # TF-IDF sur texte nettoyé
        #print(type(df['clean_text']))
        X_text = tfidf.transform(df['clean_text'])
    
        #print(X_text)

        # Features numériques
        num_cols = ['url_count', 'text_length', 'mention_count', 'question_count',
                    'caps_count', 'exclamation_count', 'hashtag_count', 'word_count',
                    'urgency_score', 'social_score', 'location_encoded', 'keyword_encoded']
        X_num = csr_matrix(df[num_cols].values)

        # Fusion
        #X_final = hstack([X_text, X_num])

        # Prediction
        #prediction = model.predict(X_final)[0]
        prediction = model.predict(X_text)[0]
        #proba = model.predict_proba(X_final)[0] if hasattr(model, 'predict_proba') else [0, 0]
        proba = model.predict_proba(X_text)[0]

        return {
            'prediction': int(prediction),
            'prediction_label': 'disaster' if prediction == 1 else 'not_disaster',
            'confidence_score': float(proba[1]) if len(proba) > 1 else None
        }

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédit si un tweet décrit une catastrophe ou non
    ---
    tags:
      - Prédictions
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              example: "Fire in downtown area"
            location:
              type: string
              example: "New York"
            keyword:
              type: string
              example: "fire"
    responses:
      200:
        description: Résultat de la prédiction
        schema:
          type: object
          properties:
            prediction:
              type: integer
            prediction_label:
              type: string
            confidence_score:
              type: number
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Le champ "text" est requis',
                'example': {'text': 'Forest fire near downtown'} if MODEL_TYPE == "pipeline" 
                          else {'text': 'Fire downtown!!', 'location': 'New York', 'keyword': 'fire'}
            }), 400
        
        tweet_text = data['text']
        
        # Validation du texte
        if not isinstance(tweet_text, str) or len(tweet_text.strip()) == 0:
            return jsonify({'error': 'Le texte doit être une chaîne non vide'}), 400
        
        # Validation spécifique au mode custom
        if MODEL_TYPE == "custom":
            if 'location' not in data or 'keyword' not in data:
                return jsonify({
                    'error': 'Champs requis pour le mode custom : "text", "location", "keyword"',
                    'example': {
                        'text': 'Fire downtown!! Evacuate now!',
                        'location': 'New York',
                        'keyword': 'fire'
                    }
                }), 400
        
        # Prediction
        result = predict_single_text(
            tweet_text, 
            data.get('location'), 
            data.get('keyword')
        )
        
        # Ajouter le texte original au résultat
        result['text'] = tweet_text
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prediction: {str(e)}'}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Prédit plusieurs tweets en une seule requête
    ---
    tags:
      - Prédictions
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            texts:
              type: array
              items:
                type: string
              example: ["Fire downtown", "Nice weather"]
            locations:
              type: array
              items:
                type: string
              example: ["NY", "LA"]
            keywords:
              type: array
              items:
                type: string
              example: ["fire", "weather"]
    responses:
      200:
        description: Résultats des prédictions
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Le champ "texts" (liste) est requis',
                'example': {'texts': ['Forest fire near downtown', 'Beautiful sunset today']}
            }), 400
        
        texts = data['texts']
        locations = data.get('locations', [])
        keywords = data.get('keywords', [])
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'texts doit être une liste non vide'}), 400
        
        results = []
        
        for i, text in enumerate(texts):
            try:
                location = locations[i] if i < len(locations) else None
                keyword = keywords[i] if i < len(keywords) else None
                
                result = predict_single_text(text, location, keyword)
                result.update({
                    'index': i,
                    'text': text
                })
                results.append(result)
                
            except Exception as e:
                results.append({
                    'index': i,
                    'text': text,
                    'error': str(e)
                })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prediction batch: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Vérifie si le modèle est chargé et opérationnel
    ---
    tags:
      - Statut
    responses:
      200:
        description: Statut du modèle
    """
    model_loaded = False
    
    if MODEL_TYPE == "pipeline":
        model_loaded = model_pipeline is not None
    elif MODEL_TYPE == "custom":
        model_loaded = all([model is not None, tfidf is not None, le_loc is not None, le_kw is not None])
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_type': MODEL_TYPE,
        'model_status': 'loaded' if model_loaded else 'not_loaded'
    })

@app.route('/info', methods=['GET'])
def get_info():
    """
    Informations générales sur l’API
    ---
    tags:
      - Info
    responses:
      200:
        description: Informations sur l’API
    """
    return jsonify({
        'model_type': MODEL_TYPE,
        'endpoints': {
            'POST /predict': 'Prediction d\'un seul tweet',
            'POST /predict_batch': 'Prediction multiple',
            'GET /health': 'Vérification de santé',
            'GET /info': 'Informations sur l\'API'
        },
        'required_fields': {
            'pipeline': ['text'],
            'custom': ['text', 'location', 'keyword']
        }
    })

# ============================================================================
# LANCEMENT DE L'APPLICATION
# ============================================================================

if __name__ == '__main__':
    print(f"Démarrage de l API en mode: {MODEL_TYPE}")
    
    # Chargement des modèles
    if load_models():
        print("Modeles chargés avec succes")
    else:
        print("Erreur lors du chargement des modeles")
    
    print("\nEndpoints disponibles:")
    print("  POST /predict - Prediction d'un seul tweet")
    print("  POST /predict_batch - Prediction multiple")
    print("  GET /health - Verification de sante")
    print("  GET /info - Informations sur l API")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

# ============================================================================
# EXEMPLES D'UTILISATION
# ============================================================================

"""
# Test avec curl:

# Mode Pipeline - Prediction simple
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Forest fire spreading rapidly near downtown area"}'

# Mode Custom - Prediction simple
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Forest fire spreading rapidly", "location": "California", "keyword": "fire"}'

# Prediction multiple (custom)
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Earthquake hits the city", "Beautiful sunny day"], 
       "locations": ["Tokyo", "Paris"], 
       "keywords": ["earthquake", "weather"]}'

# Vérification de sante
curl http://localhost:5000/health

# Informations sur l API
curl http://localhost:5000/info

# Test avec Python requests:
import requests

# Prediction simple (custom)
response = requests.post('http://localhost:5000/predict', 
                        json={
                            'text': 'Wildfire emergency evacuation',
                            'location': 'California',
                            'keyword': 'wildfire'
                        })
print(response.json())

# Prediction multiple
response = requests.post('http://localhost:5000/predict_batch', 
                        json={
                            'texts': ['Flood warning issued', 'Going to the beach'],
                            'locations': ['Houston', 'Miami'],
                            'keywords': ['flood', 'beach']
                        })
print(response.json())
"""