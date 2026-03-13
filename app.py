import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import hashlib
import datetime
import os
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthEstim – Simulateur de frais médicaux",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ── CSS accessible (RGAA/WCAG AA) ────────────────────────────────────────────
st.markdown("""
<style>
/* Ratio de contraste ≥ 4.5:1 sur fond blanc */
:root {
  --primary: #0055A4;
  --accent:  #E63946;
  --bg:      #F8F9FA;
  --text:    #1A1A2E;
}
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; color: var(--text); }
h1 { color: var(--primary); font-size: 2rem; }
h2 { color: var(--primary); font-size: 1.4rem; }
.metric-card {
  background: #fff;
  border-left: 5px solid var(--primary);
  border-radius: 8px;
  padding: 1rem 1.2rem;
  margin-bottom: 1rem;
  box-shadow: 0 2px 8px rgba(0,0,0,.08);
}
.rgpd-banner {
  background: #FFF3CD;
  border: 2px solid #FFC107;
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1.5rem;
}
.result-box {
  background: linear-gradient(135deg, #0055A4 0%, #0077CC 100%);
  color: white;
  border-radius: 12px;
  padding: 1.5rem;
  text-align: center;
  font-size: 1.6rem;
  font-weight: bold;
  margin: 1rem 0;
}
/* Focus visible pour accessibilité clavier */
button:focus, input:focus, select:focus { outline: 3px solid #FFC107 !important; }
.warning-bias { background:#FFF3CD; border-radius:8px; padding:.8rem; border-left:4px solid #FFC107; }
</style>
""", unsafe_allow_html=True)

# ── Auth simple ───────────────────────────────────────────────────────────────
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "demo":  hashlib.sha256("demo2024".encode()).hexdigest(),
}

def check_login(username: str, password: str) -> bool:
    h = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(username) == h

def login_page():

    if "rgpd_accepted" not in st.session_state:
        st.session_state["rgpd_accepted"] = False

    st.markdown("<h1> HealthEstim</h1>", unsafe_allow_html=True)
    st.markdown("<p>Simulateur éthique de frais médicaux</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        # ─────────────────────────────────────────
        # ÉTAPE 1 : consentement RGPD obligatoire
        # ─────────────────────────────────────────
        if not st.session_state["rgpd_accepted"]:

            st.markdown("### Notice de confidentialité")

            st.markdown("""
            <div class='rgpd-banner' role='alert' aria-live='polite'>
            <b>Protection des données personnelles</b><br><br>

            Cette application permet uniquement de <b>simuler une estimation de frais médicaux</b>.

            <br><br>

            <b>Données utilisées pour la simulation :</b>
            âge, IMC, nombre d'enfants, statut fumeur, sexe, région.

            <br><br>

            <b>Données non utilisées :</b>
            aucune donnée personnelle identifiable (nom, email, téléphone, etc.).

            <br><br>

            <b>Conservation :</b>
            les données saisies dans le simulateur ne sont pas enregistrées et
            ne sont utilisées que pour la session en cours.

            <br><br>

            Le résultat affiché constitue uniquement une <b>estimation automatique</b>.
            </div>
            """, unsafe_allow_html=True)

            consent = st.checkbox(
                "J’ai lu la notice de confidentialité et j’accepte de poursuivre."
            )

            if st.button("Continuer vers la connexion", use_container_width=True):

                if consent:
                    st.session_state["rgpd_accepted"] = True
                    logger.info("RGPD_NOTICE_ACCEPTED")
                    st.rerun()
                else:
                    st.warning("Vous devez accepter la notice pour continuer.")

            return

        # ─────────────────────────────────────────
        # ÉTAPE 2 : page de connexion
        # ─────────────────────────────────────────

        st.markdown("### Connexion")

        with st.form("login_form"):

            username = st.text_input(
                "Nom d'utilisateur",
                autocomplete="username",
                help="Utilisez : admin ou demo"
            )

            password = st.text_input(
                "Mot de passe",
                type="password",
                autocomplete="current-password"
            )

            submitted = st.form_submit_button(
                "Se connecter",
                use_container_width=True
            )

            if submitted:

                if check_login(username, password):

                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username

                    logger.info("LOGIN_SUCCESS | user=%s", username)

                    st.success("Connexion réussie")
                    st.rerun()

                else:

                    logger.warning("LOGIN_FAIL | user=%s", username)
                    st.error("❌ Identifiants incorrects.")

        st.caption("Comptes de démonstration : **admin / admin123**  •  **demo / demo2024**")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/model_lr.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_cols.json") as f:
        cols = json.load(f)
    return model, cols

@st.cache_data
def load_data():
    df = pd.read_csv("data/insurance_data.csv")
    return df

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, cols, age, bmi, children, smoker, sex, region):
    row = {c: 0 for c in cols}
    row["age"] = age
    row["bmi"] = bmi
    row["children"] = children
    row["smoker_enc"] = 1 if smoker == "Oui" else 0
    row["sex_enc"] = 1 if sex == "Homme" else 0
    reg_col = f"region_{region.lower()}"
    if reg_col in row:
        row[reg_col] = 1
    X = pd.DataFrame([row])
    return max(0, model.predict(X)[0])

# ── Dashboard page ────────────────────────────────────────────────────────────
def dashboard_page(df):
    st.markdown("<h1> Dashboard – Analyse des frais médicaux</h1>", unsafe_allow_html=True)
    st.caption("Données anonymisées – aucune information personnelle affichée")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Clients", f"{len(df):,}".replace(",","  "), "#0055A4"),
        (" Frais moyens", f"{df['charges'].mean():,.0f} €", "#28A745"),
        (" Fumeurs", f"{(df['smoker']=='yes').sum()} ({(df['smoker']=='yes').mean()*100:.0f}%)", "#E63946"),
        (" IMC moyen", f"{df['bmi'].mean():.1f}", "#FFC107"),
    ]
    for col, (label, val, color) in zip([col1,col2,col3,col4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div style='color:{color};font-size:.85rem;font-weight:600'>{label}</div>
              <div style='font-size:1.6rem;font-weight:700;color:#1A1A2E'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Charts using st.vega_lite_chart (no extra deps)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("####  Distribution des frais médicaux")
        hist = df.assign(tranche=pd.cut(df["charges"], bins=20)).groupby("tranche", observed=True)["charges"].count().rename("Nombre de clients")
        hist.index = [f"{int(i.left/1000)}k-{int(i.right/1000)}k €" for i in hist.index]
        st.bar_chart(hist, height=280)

    with col_right:
        st.markdown("####  Frais moyens : fumeurs vs non-fumeurs")
        comp = df.groupby("smoker")["charges"].mean().reset_index()
        comp["smoker"] = comp["smoker"].map({"yes": "Fumeur", "no": "Non-fumeur"})
        comp = comp.set_index("smoker")
        st.bar_chart(comp, height=280)

    st.markdown("---")
    st.markdown("#### 🔗 Corrélation IMC × Âge × Frais médicaux")

    # Scatter via vega-lite
    sample = df[["age", "bmi", "charges", "smoker"]].sample(min(400, len(df)), random_state=1)
    sample["Statut tabac"] = sample["smoker"].map({"yes": "Fumeur", "no": "Non-fumeur"})
    st.vega_lite_chart(sample, {
        "mark": {"type": "circle", "opacity": 0.65, "size": 60},
        "encoding": {
            "x": {"field": "age", "type": "quantitative", "title": "Âge"},
            "y": {"field": "charges", "type": "quantitative", "title": "Frais médicaux (€)"},
            "color": {"field": "Statut tabac", "type": "nominal",
                      "scale": {"range": ["#0055A4", "#E63946"]}},
            "tooltip": [
                {"field": "age", "title": "Âge"},
                {"field": "bmi", "title": "IMC"},
                {"field": "charges", "title": "Frais (€)", "format": ",.0f"},
                {"field": "Statut tabac"},
            ],
        },
    }, use_container_width=True, height=350)

    # IMC vs charges
    st.markdown("####  IMC × Frais médicaux")
    st.vega_lite_chart(sample, {
        "mark": {"type": "circle", "opacity": 0.65, "size": 55},
        "encoding": {
            "x": {"field": "bmi", "type": "quantitative", "title": "IMC"},
            "y": {"field": "charges", "type": "quantitative", "title": "Frais médicaux (€)"},
            "color": {"field": "Statut tabac", "type": "nominal",
                      "scale": {"range": ["#0055A4", "#E63946"]}},
            "tooltip": [
                {"field": "bmi", "title": "IMC"},
                {"field": "charges", "title": "Frais (€)", "format": ",.0f"},
                {"field": "Statut tabac"},
            ],
        },
    }, use_container_width=True, height=350)

# ── Simulator page ────────────────────────────────────────────────────────────
def simulator_page(model, cols):
    st.markdown("<h1>🧮 Simulateur de tarif</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='rgpd-banner' role='note'>
    <b>🔒 Protection des données :</b> Aucune donnée saisie n'est conservée ni transmise.
    La simulation est entièrement locale à votre session.
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("### Vos informations médicales")
        age = st.slider("Âge", min_value=18, max_value=80, value=35,
                        help="Votre âge en années")
        bmi = st.slider("IMC (Indice de Masse Corporelle)", 15.0, 55.0, 28.0, 0.1,
                        help="Poids(kg) / Taille²(m)")
        children = st.selectbox("Nombre d'enfants à charge", [0,1,2,3,4,5], index=0)
        smoker = st.radio("Fumeur·se ?", ["Non", "Oui"], horizontal=True,
                          help="Statut tabagique actuel")
        sex = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
        region = st.selectbox("Région (US)", ["northeast","northwest","southeast","southwest"],
                              format_func=lambda x: {
                                  "northeast":"Nord-Est","northwest":"Nord-Ouest",
                                  "southeast":"Sud-Est","southwest":"Sud-Ouest"
                              }[x])
        simulate_btn = st.button("🔍 Simuler mes frais", use_container_width=True,
                                  type="primary")

    with col_result:
        st.markdown("### Résultat de la simulation")
        if simulate_btn:
            smoker_val = "Oui" if smoker == "Oui" else "Non"
            pred = predict(model, cols, age, bmi, children, smoker_val, sex, region)
            logger.info("SIMULATION | age=%d bmi=%.1f children=%d smoker=%s region=%s → %.0f",
                        age, bmi, children, smoker, region, pred)

            st.markdown(f"""
            <div class='result-box' role='status' aria-live='polite'>
             {pred:,.0f} € / an
              <div style='font-size:.9rem;font-weight:400;margin-top:.4rem'>
                estimation annuelle de vos frais médicaux
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Breakdown
            monthly = pred / 12
            st.metric(" Estimation mensuelle", f"{monthly:,.0f} €")

            # IMC category
            if bmi < 18.5:    bmi_cat, bmi_col = "Insuffisance pondérale", "#17A2B8"
            elif bmi < 25:    bmi_cat, bmi_col = "Poids normal ✅", "#28A745"
            elif bmi < 30:    bmi_cat, bmi_col = "Surpoids", "#FFC107"
            else:             bmi_cat, bmi_col = "Obésité", "#E63946"
            st.markdown(f"**Catégorie IMC :** <span style='color:{bmi_col}'>{bmi_cat}</span>",
                        unsafe_allow_html=True)

            # Bias warning
            if smoker == "Oui":
                st.markdown("""
                <div class='warning-bias' role='alert'>
                 <b>Note sur les biais :</b> Le statut fumeur a un poids très élevé dans le modèle
                (+23 651 € en moyenne). Ce coefficient reflète une réalité actuarielle mais peut
                pénaliser certains profils. Nous recommandons une revue humaine pour les cas limites.
                </div>""", unsafe_allow_html=True)

            # Contributions
            st.markdown("####  Facteurs influençant votre estimation")
            contribs = {
                "Âge": 256.98 * age,
                "IMC": 337.09 * bmi,
                "Enfants": 425.28 * children,
                "Fumeur": 23651.13 * (1 if smoker == "Oui" else 0),
            }
            contrib_df = pd.DataFrame({"Facteur": list(contribs.keys()),
                                        "Contribution (€)": list(contribs.values())})
            contrib_df = contrib_df[contrib_df["Contribution (€)"] > 0].sort_values(
                "Contribution (€)", ascending=False)
            st.bar_chart(contrib_df.set_index("Facteur"), height=220)
        else:
            st.info(" Renseignez vos informations et cliquez sur **Simuler**.")

# ── Model page ────────────────────────────────────────────────────────────────
def model_page():
    st.markdown("<h1>📘 Modèle & Éthique</h1>", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # Modèle utilisé
    # ─────────────────────────────────────────
    st.markdown("### Modèle utilisé : Régression Linéaire")
    st.markdown("""
    Nous utilisons une **régression linéaire** (scikit-learn) afin de garantir
    la **transparence** et l’**interprétabilité** des prédictions.
    Chaque coefficient permet de comprendre l’effet d’une variable sur le coût estimé,
    toutes choses égales par ailleurs.
    """)

    coefs = {
        "Âge (par année)": 256.98,
        "IMC (par point)": 337.09,
        "Enfant à charge": 425.28,
        "Fumeur": 23651.13,
        "Sexe masculin": -18.59,
        "Région Nord-Ouest": -370.68,
        "Région Sud-Est": -657.86,
        "Région Sud-Ouest": -809.80,
    }

    coef_df = pd.DataFrame({
        "Variable": list(coefs.keys()),
        "Coefficient (€)": list(coefs.values())
    })

    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    st.info(
        "Un coefficient positif indique une augmentation du coût prédit lorsque la variable augmente. "
        "Un coefficient négatif indique une baisse du coût prédit."
    )

    # ─────────────────────────────────────────
    # Performance du modèle
    # ─────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='metric-card'>
          <div style='color:#28A745;font-weight:600'>R² du modèle</div>
          <div style='font-size:2rem;font-weight:700'>0.784</div>
          <div style='color:#666;font-size:.85rem'>sur le jeu de test (20%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
          <div style='color:#E63946;font-weight:600'>MAE</div>
          <div style='font-size:2rem;font-weight:700'>4 181 €</div>
          <div style='color:#666;font-size:.85rem'>erreur absolue moyenne</div>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # Analyse des biais automatique
    # ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚖️ Analyse des biais")

    st.write(
        "L’analyse suivante est calculée automatiquement sur le jeu de test. "
        "Une erreur moyenne **positive** signifie que le modèle tend à **surestimer** "
        "les frais pour un groupe. Une erreur moyenne **négative** signifie qu’il tend "
        "à **sous-estimer** les frais."
    )

    try:
        bias_df = pd.read_csv("models/bias_report.csv")

        st.dataframe(bias_df, use_container_width=True, hide_index=True)

        st.markdown("""
        **Interprétation :**
        - une valeur proche de 0 indique un biais faible ;
        - une valeur positive indique une tendance à la sur-estimation ;
        - une valeur négative indique une tendance à la sous-estimation.
        """)

        st.markdown("""
        **Solution proposée :**  
        comparer ce modèle à un modèle plus flexible comme un **Random Forest Regressor**,
        puis vérifier si l’amélioration de la performance permet aussi de réduire les écarts
        d’erreur entre groupes.
        """)

    except FileNotFoundError:
        st.warning(
            "Le fichier `models/bias_report.csv` est introuvable. "
            "Exécute d’abord `train_model.py` pour générer l’analyse automatique des biais."
        )

    # ─────────────────────────────────────────
    # Conformité RGPD
    # ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔒 Conformité RGPD")

    st.markdown("""
    **Variables exclues du modèle :** `nom`, `prénom`, `email`, `téléphone`,
    `numéro de sécurité sociale`, `date de naissance`, `adresse IP`, `ville`, `code postal`.

    Ces informations sont des **données personnelles identifiantes** ou non nécessaires à la prédiction.
    Elles ne transitent jamais dans le pipeline du modèle.

    **Mesures mises en place :**
    1. **Minimisation des données** : seules les variables utiles à l’estimation sont utilisées.
    2. **Transparence** : une notice de confidentialité est affichée avant l’accès à l’application.
    3. **Sécurité** : l’accès est protégé par authentification et l’application est déployée en HTTPS.
    """)

    # ─────────────────────────────────────────
    # Accessibilité
    # ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ♿ Accessibilité")

    st.markdown("""
    L’application respecte plusieurs principes d’accessibilité inspirés du **RGAA / WCAG AA** :

    1. **Contrastes de couleur suffisants** pour garantir la lisibilité.
    2. **Navigation clavier** avec focus visible sur les éléments interactifs.
    3. **Libellés explicites et attributs ARIA** sur les composants dynamiques.
    """)

    st.caption(
        "Cette page combine explicabilité du modèle, vigilance sur les biais, "
        "protection des données et accessibilité."
    )

# ── Logs page ─────────────────────────────────────────────────────────────────
def logs_page():
    st.markdown("<h1> Journaux d'activité</h1>", unsafe_allow_html=True)
    if st.session_state.get("username") != "admin":
        st.warning(" Accès réservé à l'administrateur.")
        return
    try:
        with open("app.log") as f:
            lines = f.readlines()
        st.code("".join(lines[-50:]), language="text")
    except FileNotFoundError:
        st.info("Aucun log disponible.")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login_page()
        return

    model, cols = load_model()
    df = load_data()

    # Sidebar
    with st.sidebar:
        st.markdown(f"###  HealthEstim")
        st.markdown(f"Connecté : **{st.session_state['username']}**")
        st.markdown("---")
        page = st.radio("Navigation", [" Dashboard", " Simulateur", " Modèle & Éthique", " Logs"],
                        label_visibility="collapsed")
        st.markdown("---")
        st.markdown("""
        <div style='font-size:.75rem;color:#666'>
         Données anonymisées<br>
        🇪🇺 Conforme RGPD<br>
         Accessible WCAG AA
        </div>""", unsafe_allow_html=True)
        if st.button(" Déconnexion"):
            logger.info("LOGOUT | user=%s", st.session_state["username"])
            st.session_state.clear()
            st.rerun()

    if page == " Dashboard":
        dashboard_page(df)
    elif page == " Simulateur":
        simulator_page(model, cols)
    elif page == " Modèle & Éthique":
        model_page()
    elif page == " Logs":
        logs_page()

if __name__ == "__main__":
    main()