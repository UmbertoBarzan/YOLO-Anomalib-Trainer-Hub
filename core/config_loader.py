import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path: Path = Path("configs/global.yaml")):
        self.config_path = config_path
        self.config_data = self._load_config()

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"File di configurazione non trovato: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_path(self, key: str) -> Path:
        """
        Ritorna un percorso specifico dalla sezione 'paths'
        """
        try:
            return Path(self.config_data["paths"][key])
        except KeyError:
            raise KeyError(f"Chiave non trovata in paths: '{key}'")

    def get(self, key: str, default=None):
        """
        Ritorna una qualsiasi altra chiave della configurazione
        """
        return self.config_data.get(key, default)
