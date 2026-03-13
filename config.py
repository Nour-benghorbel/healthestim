from pathlib import Path
import hashlib

# Dossier racine du projet
BASE_DIR = Path(__file__).resolve().parent

# Chemins vers les ressources
DATA_PATH = BASE_DIR / "data" / "insurance_data.csv"
MODEL_PATH = BASE_DIR / "models" / "model_lr.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_cols.json"
LOG_PATH = BASE_DIR / "app.log"
CSS_PATH = BASE_DIR / "assets" / "style.css"

# Comptes de démonstration
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "demo": hashlib.sha256("demo2024".encode()).hexdigest(),
}

# Métadonnées application
APP_TITLE = "HealthEstim"
APP_ICON = ""
LAYOUT = "wide"

# Variables sensibles exclues du modèle
EXCLUDED_COLUMNS = [
    "id_client",
    "nom",
    "prenom",
    "date_naissance",
    "email",
    "telephone",
    "numero_secu_sociale",
    "ville",
    "code_postal",
    "adresse_ip",
]