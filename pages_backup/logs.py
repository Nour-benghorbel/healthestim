import streamlit as st
from config import LOG_PATH

def render_logs(username: str):
    """
    Affiche les logs uniquement pour l'utilisateur administrateur.
    """
    st.title("Journaux d'activité")

    if username != "admin":
        st.warning("Accès réservé à l'administrateur.")
        return

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        last_lines = "".join(lines[-50:])
        st.code(last_lines if last_lines else "Aucun log disponible.", language="text")

    except FileNotFoundError:
        st.info("Aucun log disponible.")