import pandas as pd
import streamlit as st
from config import EXCLUDED_COLUMNS

def render_model_ethics():
    """
    Affiche les informations sur le modèle, l'interprétabilité,
    l'éthique, l'accessibilité et la conformité RGPD.
    """
    st.title("Modèle et éthique")

    st.subheader("Modèle utilisé")
    st.write(
        "Le modèle choisi est une régression linéaire. "
        "Ce choix permet d'interpréter facilement chaque coefficient "
        "et de justifier l'effet de chaque variable sur la prédiction."
    )

    coefficients = {
        "Âge": 256.98,
        "IMC": 337.09,
        "Nombre d'enfants": 425.28,
        "Fumeur": 23651.13,
        "Sexe masculin": -18.59,
        "Région northwest": -370.68,
        "Région southeast": -657.86,
        "Région southwest": -809.80,
    }

    coef_df = pd.DataFrame({
        "Variable": list(coefficients.keys()),
        "Coefficient": list(coefficients.values())
    })

    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Performance du modèle")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("R²", "0.784")

    with col2:
        st.metric("MAE", "4 181 €")

    st.markdown("---")

    st.subheader("Analyse des biais")
    st.write(
        "Le modèle semble donner un poids très élevé au statut fumeur. "
        "Cela peut être cohérent sur le plan statistique, mais cela peut aussi conduire "
        "à une pénalisation importante de certains profils. Une analyse comparative par groupe "
        "permet d'identifier les écarts d'erreur."
    )

    bias_df = pd.DataFrame({
        "Groupe": ["Non-fumeurs", "Fumeurs", "Région northeast", "Région southeast"],
        "Erreur moyenne (€)": [-77, 87, 85, -275],
        "Interprétation": [
            "Faible biais",
            "Légère sur-estimation",
            "Légère sur-estimation",
            "Sous-estimation"
        ],
    })

    st.dataframe(bias_df, use_container_width=True, hide_index=True)

    st.write(
        "Solution proposée : recalibrer le modèle par sous-groupes, comparer avec un arbre de décision "
        "ou un Random Forest, puis vérifier si l'amélioration de la performance réduit aussi les écarts entre groupes."
    )

    st.markdown("---")

    st.subheader("Conformité RGPD")
    st.write(
        "Les données sensibles ou directement identifiantes sont exclues du pipeline de prédiction."
    )

    st.code("\n".join(EXCLUDED_COLUMNS), language="text")

    st.write(
        "Le traitement est limité aux variables strictement nécessaires à l'estimation des frais médicaux. "
        "Aucune donnée nominative n'est utilisée dans le modèle."
    )

    st.markdown("---")

    st.subheader("Mesures d'accessibilité")
    st.write("Trois mesures concrètes mises en place pour respecter l'accessibilité :")
    st.markdown(
        """
        1. Contrastes suffisants entre texte et fond.
        2. Navigation clavier avec focus visible.
        3. Libellés clairs et messages compréhensibles sur les formulaires.
        """
    )