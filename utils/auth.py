import hashlib
import streamlit as st
from config import USERS

def hash_password(password: str) -> str:
    """
    Hash un mot de passe avec SHA-256.
    """
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username: str, password: str) -> bool:
    """
    Vérifie si le couple identifiant / mot de passe est valide.
    """
    return USERS.get(username) == hash_password(password)

def init_session():
    """
    Initialise les variables de session nécessaires.
    """
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None

def logout():
    """
    Déconnecte l'utilisateur.
    """
    st.session_state.clear()