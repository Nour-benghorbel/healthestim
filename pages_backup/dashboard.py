import pandas as pd
import streamlit as st

def render_dashboard(df: pd.DataFrame):
    """
    Affiche le tableau de bord analytique.
    """
    st.title("Dashboard")
    st.caption("Données anonymisées utilisées pour l'analyse")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Nombre de clients", len(df))

    with col2:
        st.metric("Frais moyens", f"{df['charges'].mean():,.0f} €")

    with col3:
        smoker_rate = (df["smoker"] == "yes").mean() * 100
        st.metric("Taux de fumeurs", f"{smoker_rate:.1f} %")

    with col4:
        st.metric("IMC moyen", f"{df['bmi'].mean():.1f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribution des frais médicaux")
        hist = (
            df.assign(tranche=pd.cut(df["charges"], bins=20))
            .groupby("tranche", observed=True)["charges"]
            .count()
        )
        hist.index = [f"{int(i.left/1000)}k-{int(i.right/1000)}k" for i in hist.index]
        st.bar_chart(hist)

    with col_right:
        st.subheader("Frais moyens selon le statut fumeur")
        comp = df.groupby("smoker")["charges"].mean().reset_index()
        comp["smoker"] = comp["smoker"].map({"yes": "Fumeur", "no": "Non-fumeur"})
        st.bar_chart(comp.set_index("smoker"))

    st.markdown("---")

    st.subheader("Corrélation entre âge, IMC et frais médicaux")
    sample = df[["age", "bmi", "charges", "smoker"]].sample(min(400, len(df)), random_state=42)
    sample["Statut tabac"] = sample["smoker"].map({"yes": "Fumeur", "no": "Non-fumeur"})

    st.vega_lite_chart(
        sample,
        {
            "mark": {"type": "circle", "opacity": 0.65, "size": 70},
            "encoding": {
                "x": {"field": "age", "type": "quantitative", "title": "Âge"},
                "y": {"field": "charges", "type": "quantitative", "title": "Frais médicaux"},
                "color": {"field": "Statut tabac", "type": "nominal"},
                "tooltip": [
                    {"field": "age", "title": "Âge"},
                    {"field": "bmi", "title": "IMC"},
                    {"field": "charges", "title": "Charges", "format": ",.0f"},
                    {"field": "Statut tabac", "title": "Statut tabac"},
                ],
            },
        },
        use_container_width=True,
    )

    st.subheader("Relation entre IMC et frais médicaux")
    st.vega_lite_chart(
        sample,
        {
            "mark": {"type": "circle", "opacity": 0.65, "size": 70},
            "encoding": {
                "x": {"field": "bmi", "type": "quantitative", "title": "IMC"},
                "y": {"field": "charges", "type": "quantitative", "title": "Frais médicaux"},
                "color": {"field": "Statut tabac", "type": "nominal"},
                "tooltip": [
                    {"field": "bmi", "title": "IMC"},
                    {"field": "charges", "title": "Charges", "format": ",.0f"},
                    {"field": "Statut tabac", "title": "Statut tabac"},
                ],
            },
        },
        use_container_width=True,
    )