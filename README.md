# HealthEstim - Simulateur éthique de frais médicaux

## Description
HealthEstim est une application Streamlit permettant d'estimer les frais médicaux annuels
d'un futur client à partir de variables non directement identifiantes.

Le projet répond aux exigences suivantes :
- modèle de régression interprétable ;
- tableau de bord interactif ;
- formulaire de simulation ;
- authentification simple ;
- gestion des logs ;
- prise en compte de la conformité RGPD ;
- mesures d'accessibilité inspirées du RGAA / WCAG.

## Structure du projet
```bash
HEALTH-INSURTECH_PROJECT/
├── .streamlit/
│   └── config.toml
├── assets/
│   └── style.css
├── data/
│   └── insurance_data.csv
├── models/
│   ├── model_lr.pkl
│   └── feature_cols.json
├── pages/
│   ├── dashboard.py
│   ├── logs.py
│   ├── model_ethics.py
│   └── simulator.py
├── utils/
│   ├── auth.py
│   ├── logger.py
│   └── predict.py
├── app.py
├── config.py
├── README.md
└── requirements.txt


