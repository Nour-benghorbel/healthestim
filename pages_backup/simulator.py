import pandas as pd
import streamlit as st
from utils.predict import predict_charges

def render_simulator(model, feature_cols, logger):
    """
    Affiche le formulaire de simulation des frais médicaux.
    """
    st.title("Simulateur de tarif")

    st.info(
        "Les données saisies dans ce formulaire ne sont ni conservées ni partagées. "
        "La simulation est réalisée uniquement pour la session en cours."
    )

    col_form, col_result = st.columns(2)

    with col_form:
        age = st.slider("Âge", 18, 80, 35)
        bmi = st.slider("IMC", 15.0, 55.0, 28.0, 0.1)
        children = st.selectbox("Nombre d'enfants à charge", [0, 1, 2, 3, 4, 5], index=0)
        smoker = st.radio("Fumeur", ["Non", "Oui"], horizontal=True)
        sex = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
        region = st.selectbox(
            "Région",
            ["northeast", "northwest", "southeast", "southwest"]
        )

        simulate = st.button("Simuler", use_container_width=True)

    with col_result:
        st.subheader("Résultat")

        if simulate:
            prediction = predict_charges(
                model=model,
                feature_cols=feature_cols,
                age=age,
                bmi=bmi,
                children=children,
                smoker=smoker,
                sex=sex,
                region=region,
            )

            logger.info(
                "SIMULATION | age=%s | bmi=%s | children=%s | smoker=%s | sex=%s | region=%s | pred=%.2f",
                age, bmi, children, smoker, sex, region, prediction
            )

            st.success(f"Estimation annuelle : {prediction:,.0f} €")
            st.metric("Estimation mensuelle", f"{prediction / 12:,.0f} €")

            if bmi < 18.5:
                bmi_label = "Insuffisance pondérale"
            elif bmi < 25:
                bmi_label = "Poids normal"
            elif bmi < 30:
                bmi_label = "Surpoids"
            else:
                bmi_label = "Obésité"

            st.write(f"Catégorie IMC : {bmi_label}")

            contributions = {
                "Âge": 256.98 * age,
                "IMC": 337.09 * bmi,
                "Enfants": 425.28 * children,
                "Fumeur": 23651.13 * (1 if smoker == "Oui" else 0),
            }

            contrib_df = pd.DataFrame({
                "Facteur": list(contributions.keys()),
                "Contribution (€)": list(contributions.values())
            })

            contrib_df = contrib_df[contrib_df["Contribution (€)"] > 0]
            contrib_df = contrib_df.sort_values("Contribution (€)", ascending=False)

            st.subheader("Facteurs principaux de l'estimation")
            st.bar_chart(contrib_df.set_index("Facteur"))

            if smoker == "Oui":
                st.warning(
                    "Le statut fumeur a un effet très important dans le modèle. "
                    "Cette différence doit être analysée avec prudence dans la partie biais et équité."
                )
        else:
            st.write("Remplissez le formulaire puis cliquez sur Simuler.")