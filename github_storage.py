import json
import base64
import requests
import streamlit as st
from datetime import datetime


# ─────────────────────────────────────────────
# GITHUB STORAGE — Persistance des donnees
# Sauvegarde trades.json dans le repo GitHub
# ─────────────────────────────────────────────

def _get_github_config():
    """Recupere le token et repo depuis Streamlit Secrets."""
    try:
        token = st.secrets["github"]["token"]
        repo  = st.secrets["github"]["repo"]
        return token, repo
    except Exception:
        return None, None


def _get_file_sha(token, repo, filepath):
    """Recupere le SHA du fichier pour pouvoir le mettre a jour."""
    url     = f"https://api.github.com/repos/{repo}/contents/{filepath}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    resp    = requests.get(url, headers=headers, timeout=10)
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


def save_to_github(data, filepath="data/trades.json"):
    """
    Sauvegarde des donnees JSON dans le repo GitHub.
    """
    token, repo = _get_github_config()
    if not token or not repo:
        print("GitHub storage: token ou repo manquant dans Secrets")
        return False

    try:
        content_str = json.dumps(data, indent=2, default=str)
        content_b64 = base64.b64encode(content_str.encode()).decode()

        url     = f"https://api.github.com/repos/{repo}/contents/{filepath}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

        sha = _get_file_sha(token, repo, filepath)

        payload = {
            "message": f"Update {filepath} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "content": content_b64,
        }
        if sha:
            payload["sha"] = sha

        resp = requests.put(url, headers=headers, json=payload, timeout=15)
        if resp.status_code not in [200, 201]:
            print(f"GitHub save error: {resp.status_code} - {resp.text[:100]}")
            return False
        return True

    except Exception as e:
        print(f"GitHub save exception: {str(e)[:100]}")
        return False


def load_from_github(filepath="data/trades.json"):
    """Charge des donnees JSON depuis le repo GitHub."""
    token, repo = _get_github_config()
    if not token or not repo:
        return None

    try:
        url     = f"https://api.github.com/repos/{repo}/contents/{filepath}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        resp    = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            content_b64 = resp.json().get("content", "")
            content_str = base64.b64decode(content_b64).decode()
            return json.loads(content_str)
        return None

    except Exception:
        return None


def save_paper_trades_github(trades):
    """Sauvegarde les paper trades sur GitHub."""
    return save_to_github({"paper_trades": trades}, "data/paper_trades.json")


def load_paper_trades_github():
    """Charge les paper trades depuis GitHub."""
    data = load_from_github("data/paper_trades.json")
    if data:
        return data.get("paper_trades", [])
    return []


def save_journal_github(trades):
    """Sauvegarde le journal reel sur GitHub."""
    return save_to_github({"journal_trades": trades}, "data/journal_trades.json")


def load_journal_github():
    """Charge le journal reel depuis GitHub."""
    data = load_from_github("data/journal_trades.json")
    if data:
        return data.get("journal_trades", [])
    return []


def github_storage_available():
    """Verifie si le stockage GitHub est configure."""
    token, repo = _get_github_config()
    return bool(token and repo)
