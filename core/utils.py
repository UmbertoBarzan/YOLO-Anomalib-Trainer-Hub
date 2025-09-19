import yaml
from pathlib import Path
import os

#TODO: aggiungi qui qualsiasi funzione di utilità generale

def load_yaml(path: Path) -> dict:
    """Carica un file YAML e restituisce un dizionario."""
    if not path.exists():
        raise FileNotFoundError(f"File YAML non trovato: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: dict, path: Path):
    """Salva un dizionario come YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f)

def highlight(msg: str) -> str:
    """Aggiunge un evidenziatore ANSI arancione per il terminale."""
    return f"\033[48;5;208m{msg}\033[0m"

def ensure_dir(path: Path):
    """Crea la directory se non esiste."""
    path.mkdir(parents=True, exist_ok=True)
