# src/make_dataset.py

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

def load_data(dataset_name="iris", test_size=0.2, random_state=42):
    """
    Charge et prépare les données pour l'entraînement
    
    Args:
        dataset_name (str): nom du dataset ('iris', 'wine', 'boston')
        test_size (float): proportion du test set
        random_state (int): seed pour la reproductibilité
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    if dataset_name == "iris":
        data = load_iris()
    elif dataset_name == "wine":
        data = load_wine()
    elif dataset_name == "house_prices":
        data = pd.read_csv("../data/train.csv")
        return data
    elif dataset_name == "tweets":
        data = pd.read_csv("../data/tweets/train.csv")
        return data
    else:
        raise ValueError(f"Dataset '{dataset_name}' non supporté")
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def load_csv_data(file_path, target_column, test_size=0.2, random_state=42):
    """
    Charge des données depuis un fichier CSV
    
    Args:
        file_path (str): chemin vers le fichier CSV
        target_column (str): nom de la colonne target
        test_size (float): proportion du test set
        random_state (int): seed pour la reproductibilité
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test