import pytest
import pandas as pd
import numpy as np
import re
from typing import Optional
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]  
DATA_PATH = ROOT_DIR / 'data' / 'tweets' / 'train.csv'


# Configuration du dataset de test
@pytest.fixture
def df() -> pd.DataFrame:
    """
    Fixture qui charge le dataset disaster tweets.
    """
    return pd.read_csv(DATA_PATH)
    
    # Dataset factice 
    #return pd.DataFrame({
    #    'id': [1, 2, 3, 4, 5],
    #    'keyword': ['earthquake', 'fire', None, 'flood', 'storm'],
    #    'location': ['California', None, 'New York', 'Texas', 'Florida'],
    #    'text': [
    #        'Earthquake hits California!',
    #        'Fire spreading rapidly',
    #        'Just a normal day',
    #        'Flood warning issued',
    #        'Storm approaching coast'
    #    ],
    #    'target': [1, 1, 0, 1, 1]
    #})

class TestDatasetStructure:
    """Tests de structure du dataset"""
    
    def test_required_columns_present(self, df):
        required_columns = ['id', 'keyword', 'location', 'text', 'target']
        missing_columns = set(required_columns) - set(df.columns)
        assert len(missing_columns) == 0, f"Colonnes manquantes: {missing_columns}"
    
    def test_dataset_not_empty(self, df):
        assert len(df) > 0, "Le dataset est vide"
        assert len(df) >= 100, f"Dataset trop petit: {len(df)} lignes (minimum recommandé: 100)"
    
    def test_column_types(self, df):
        assert pd.api.types.is_numeric_dtype(df['id']) or df['id'].dtype == 'object', \
            f"Type de 'id' incorrect: {df['id'].dtype}"
        
        assert pd.api.types.is_numeric_dtype(df['target']), \
            f"Type de 'target' incorrect: {df['target'].dtype}"
        
        assert df['text'].dtype in ['object', 'string'], \
            f"Type de 'text' incorrect: {df['text'].dtype}"

class TestDataQuality:
    """Tests de qualité des données"""
    
    def test_no_missing_critical_columns(self, df):
        critical_columns = ['id', 'text', 'target']
        for col in critical_columns:
            missing_count = df[col].isnull().sum()
            assert missing_count == 0, f"Colonne critique '{col}' a {missing_count} valeurs manquantes"
    
    def test_no_duplicate_ids(self, df):
        duplicate_ids = df['id'].duplicated().sum()
        assert duplicate_ids == 0, f"{duplicate_ids} IDs dupliqués trouvés"
    
    def test_no_empty_texts(self, df):
        empty_texts = (df['text'].str.strip() == '').sum()
        assert empty_texts == 0, f"{empty_texts} textes vides trouvés"
    
    def test_reasonable_duplicate_texts(self, df):
        """Vérifier que le taux de textes dupliqués reste raisonnable"""
        duplicate_texts = df['text'].duplicated().sum()
        duplicate_rate = duplicate_texts / len(df) * 100
        assert duplicate_rate < 20, f"Taux de textes dupliqués trop élevé: {duplicate_rate:.1f}%"

class TestTargetVariable:
    """Tests sur la variable cible"""
    
    def test_target_binary_values(self, df):
        unique_values = set(df['target'].unique())
        expected_values = {0, 1}
        assert unique_values == expected_values, \
            f"Valeurs inattendues dans target: {unique_values - expected_values}"
    
    def test_class_balance_not_extreme(self, df):
        class_counts = df['target'].value_counts()
        if len(class_counts) == 2:
            imbalance_ratio = class_counts.max() / class_counts.min()
            assert imbalance_ratio <= 10, \
                f"Déséquilibre des classes extrême: {imbalance_ratio:.2f}:1"
    
    def test_both_classes_present(self, df):
        unique_targets = df['target'].nunique()
        assert unique_targets == 2, f"Nombre de classes incorrect: {unique_targets} (attendu: 2)"

class TestTextContent:
    """Tests sur le contenu textuel"""
    
    def test_text_length_reasonable(self, df):
        text_lengths = df['text'].str.len()
        
        # Textes trop courts (moins de 2 caracteres)
        too_short = (text_lengths < 2).sum()
        assert too_short == 0, f"{too_short} textes trop courts (< 2 caractères)"
        
        # Textes trop longs (plus de 500 caracteres, anormal pour des tweets)
        too_long = (text_lengths > 500).sum()
        assert too_long == 0, f"{too_long} textes anormalement longs (> 500 caractères)"
        
        # Longueur moyenne raisonnable
        mean_length = text_lengths.mean()
        assert 10 <= mean_length <= 300, \
            f"Longueur moyenne de texte suspecte: {mean_length:.1f} caractères"
    
    def test_text_encoding_clean(self, df):
        """Test 12: Vérifier l'absence de problèmes d'encodage"""
        encoding_issues = 0
        for text in df['text'].dropna():
            if isinstance(text, str):
                # Caracteres de remplacement d encodage
                if '�' in text or '\\x' in text or '\\u' in repr(text):
                    encoding_issues += 1
        
        encoding_rate = encoding_issues / len(df) * 100
        assert encoding_rate < 5, \
            f"Trop de problèmes d'encodage: {encoding_issues} textes ({encoding_rate:.1f}%)"
    
    def test_text_contains_meaningful_content(self, df):
        """Vérifier que les textes contiennent du contenu significatif"""
        # Textes avec seulement des espaces, chiffres ou caracteres speciaux
        meaningless_texts = 0
        for text in df['text'].dropna():
            if isinstance(text, str):
                # Retirer espaces, ponctuation, chiffres
                clean_text = re.sub(r'[^\w]', '', text)
                if len(clean_text) < 3:  # Moins de 3 caracteres alphabetiques
                    meaningless_texts += 1
        
        meaningless_rate = meaningless_texts / len(df) * 100
        assert meaningless_rate < 10, \
            f"Trop de textes sans contenu significatif: {meaningless_texts} ({meaningless_rate:.1f}%)"

class TestDataConsistency:
    """Tests de cohérence des données"""
    
    def test_keyword_format_consistency(self, df):
        """Vérifier la cohérence du format des mots-clés"""
        if 'keyword' in df.columns:
            suspicious_keywords = 0
            for keyword in df['keyword'].dropna():
                if isinstance(keyword, str):
                    # Mots-cles avec caracteres suspects ou trop courts
                    if len(keyword) < 2 or not re.match(r'^[a-zA-Z0-9\s%&-]+$', keyword):
                        suspicious_keywords += 1
            
            suspicious_rate = suspicious_keywords / df['keyword'].notna().sum() * 100 if df['keyword'].notna().sum() > 0 else 0
            assert suspicious_rate < 15, \
                f"Trop de mots-clés suspects: {suspicious_keywords} ({suspicious_rate:.1f}%)"
    

class TestDataTransformations:
    """Tests pour les transformations de donnees"""
    

    #def preprocess_text(self, text):
    #    """Fonction de preprocessing à tester"""
    #    if pd.isna(text) or not isinstance(text, str):
    #        return ""
        
    #    text = text.lower()
    #    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    #    text = re.sub(r'@\w+|#\w+', '', text)
    #    text = re.sub(r'\s+', ' ', text).strip()
    #    return text

    def preprocess_text(self, text):
    # Cas null ou vide
        if not text or pd.isna(text) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def test_lowercase_conversion(self):
        """Test conversion en minuscules"""
        text = "FOREST FIRE EMERGENCY"
        result = self.preprocess_text(text)
        assert result.islower(), "Le texte doit être converti en minuscules"
        assert "forest fire emergency" == result

    def test_url_removal(self):
        """Test suppression des URLs"""
        texts_with_urls = [
            "Fire emergency http://example.com",
            "Check this www.emergency.gov",
            "News https://news.com/fire-alert"
        ]
        
        for text in texts_with_urls:
            result = self.preprocess_text(text)
            assert "http" not in result and "www" not in result
            assert ".com" not in result and ".gov" not in result

    def test_mentions_hashtags_removal(self):
        """Test suppression des mentions et hashtags"""
        text = "Emergency @rescue_team #fire #help"
        result = self.preprocess_text(text)
        assert "@rescue_team" not in result
        assert "#fire" not in result
        assert "#help" not in result
        assert "emergency" in result

    def test_punctuation_removal(self):
        """Test suppression de la ponctuation"""
        text = "Emergency!!! Fire??? Help... Now!!!"
        result = self.preprocess_text(text)
        assert "!" not in result and "?" not in result and "." not in result
        assert "emergency fire help now" == result

    def test_whitespace_normalization(self):
        """Test normalisation des espaces"""
        text = "Emergency    fire     spreading    rapidly"
        result = self.preprocess_text(text)
        assert "  " not in result  
        assert result == "emergency fire spreading rapidly"

    def test_null_and_empty_handling(self):
        """Test gestion des valeurs nulles et vides"""
        test_cases = [None, "", "   ", "\t\n"]
        
        for case in test_cases:
            result = self.preprocess_text(case)
            assert result == "", f"Cas {case} doit retourner une chaîne vide"

    def test_preprocessing_consistency(self):
        """Test cohérence du preprocessing"""
        text = "Same TEXT input"
        result1 = self.preprocess_text(text)
        result2 = self.preprocess_text(text)
        assert result1 == result2, "Le preprocessing doit être déterministe"   

    
def run_all_tests(dataset_path: Optional[str] = None):
    """
    Fonction pour executer tous les tests
    
    """
    if dataset_path:
        pytest.main([__file__, '-v', f'--dataset-path={dataset_path}'])
    else:
        pytest.main([__file__, '-v'])

if __name__ == "__main__":
    # Exécution directe
    run_all_tests()