import json
import pickle
import pandas as pd
import streamlit as st
from config import MODEL_PATH, FEATURES_PATH

@st.cache_resource
def load_model_and_features():
    """
    Charge le modèle entraîné et la liste des colonnes attendues.
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    return model, feature_cols

def prepare_input(age, bmi, children, smoker, sex, region, feature_cols):
    """
    Prépare une ligne de données conforme aux colonnes utilisées par le modèle.
    """
    row = {col: 0 for col in feature_cols}

    row["age"] = age
    row["bmi"] = bmi
    row["children"] = children
    row["smoker_enc"] = 1 if smoker == "Oui" else 0
    row["sex_enc"] = 1 if sex == "Homme" else 0

    region_col = f"region_{region.lower()}"
    if region_col in row:
        row[region_col] = 1

    return pd.DataFrame([row])

def predict_charges(model, feature_cols, age, bmi, children, smoker, sex, region):
    """
    Retourne la prédiction du coût médical annuel.
    """
    X = prepare_input(age, bmi, children, smoker, sex, region, feature_cols)
    prediction = model.predict(X)[0]
    return max(0, float(prediction))