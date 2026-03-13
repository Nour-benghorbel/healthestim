import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "insurance_data.csv"
MODEL_PATH = BASE_DIR / "models" / "model_lr.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_cols.json"
BIAS_SMOKER_PATH = BASE_DIR / "models" / "bias_smoker.csv"
BIAS_REGION_PATH = BASE_DIR / "models" / "bias_region.csv"
BIAS_REPORT_PATH = BASE_DIR / "models" / "bias_report.csv"


def interpret_bias(error_value: float) -> str:
    """
    Retourne une interprétation simple de l'erreur moyenne par groupe.
    """
    if abs(error_value) < 100:
        return "Faible biais"
    elif error_value > 0:
        return "Légère sur-estimation"
    else:
        return "Sous-estimation"


def decode_region(row, region_cols):
    """
    Décode la région à partir des variables one-hot encodées.
    Comme drop_first=True, la région de référence est 'northeast'.
    """
    for col in region_cols:
        if row[col] == 1:
            return col.replace("region_", "")
    return "northeast"


def main():
    # Chargement des données
    df = pd.read_csv(DATA_PATH)

    # Encodage
    df["smoker_enc"] = df["smoker"].map({"yes": 1, "no": 0})
    df["sex_enc"] = df["sex"].map({"male": 1, "female": 0})

    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)

    X = pd.concat(
        [
            df[["age", "bmi", "children", "smoker_enc", "sex_enc"]],
            region_dummies
        ],
        axis=1
    )

    y = df["charges"]

    feature_cols = X.columns.tolist()

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entraînement du modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Métriques globales
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("R2 :", round(r2, 4))
    print("MAE:", round(mae, 2))

    # Sauvegarde du modèle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Sauvegarde des colonnes
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    # =========================
    # Analyse automatique des biais
    # =========================
    eval_df = X_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = y_pred
    eval_df["error"] = eval_df["y_pred"] - eval_df["y_true"]
    eval_df["abs_error"] = eval_df["error"].abs()

    # ----- Biais selon le statut fumeur -----
    bias_smoker = (
        eval_df.groupby("smoker_enc")["error"]
        .mean()
        .reset_index()
    )

    bias_smoker["Groupe"] = bias_smoker["smoker_enc"].map({
        0: "Non-fumeurs",
        1: "Fumeurs"
    })

    bias_smoker["Interprétation"] = bias_smoker["error"].apply(interpret_bias)

    bias_smoker = bias_smoker[["Groupe", "error", "Interprétation"]]
    bias_smoker.columns = ["Groupe", "Erreur moyenne (€)", "Interprétation"]

    # ----- Biais selon la région -----
    region_cols = [c for c in eval_df.columns if c.startswith("region_")]
    eval_df["region_name"] = eval_df.apply(lambda row: decode_region(row, region_cols), axis=1)

    bias_region = (
        eval_df.groupby("region_name")["error"]
        .mean()
        .reset_index()
    )

    bias_region["Groupe"] = bias_region["region_name"].map({
        "northeast": "Région northeast",
        "northwest": "Région northwest",
        "southeast": "Région southeast",
        "southwest": "Région southwest"
    })

    bias_region["Interprétation"] = bias_region["error"].apply(interpret_bias)

    bias_region = bias_region[["Groupe", "error", "Interprétation"]]
    bias_region.columns = ["Groupe", "Erreur moyenne (€)", "Interprétation"]

    # ----- Rapport global -----
    bias_report = pd.concat([bias_smoker, bias_region], ignore_index=True)

    # Arrondi pour lecture plus propre
    bias_smoker["Erreur moyenne (€)"] = bias_smoker["Erreur moyenne (€)"].round(2)
    bias_region["Erreur moyenne (€)"] = bias_region["Erreur moyenne (€)"].round(2)
    bias_report["Erreur moyenne (€)"] = bias_report["Erreur moyenne (€)"].round(2)

    # Sauvegarde des rapports
    bias_smoker.to_csv(BIAS_SMOKER_PATH, index=False, encoding="utf-8")
    bias_region.to_csv(BIAS_REGION_PATH, index=False, encoding="utf-8")
    bias_report.to_csv(BIAS_REPORT_PATH, index=False, encoding="utf-8")

    print("Modèle sauvegardé dans :", MODEL_PATH)
    print("Colonnes sauvegardées dans :", FEATURES_PATH)
    print("Rapport biais fumeur sauvegardé dans :", BIAS_SMOKER_PATH)
    print("Rapport biais région sauvegardé dans :", BIAS_REGION_PATH)
    print("Rapport global des biais sauvegardé dans :", BIAS_REPORT_PATH)


if __name__ == "__main__":
    main()