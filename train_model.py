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

def main():
    df = pd.read_csv(DATA_PATH)

    # Encodage
    df["smoker_enc"] = df["smoker"].map({"yes": 1, "no": 0})
    df["sex_enc"] = df["sex"].map({"male": 1, "female": 0})

    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)

    X = pd.concat([
        df[["age", "bmi", "children", "smoker_enc", "sex_enc"]],
        region_dummies
    ], axis=1)

    y = df["charges"]

    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("R2 :", round(r2_score(y_test, y_pred), 4))
    print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print("Modèle sauvegardé dans :", MODEL_PATH)
    print("Colonnes sauvegardées dans :", FEATURES_PATH)

if __name__ == "__main__":
    main()