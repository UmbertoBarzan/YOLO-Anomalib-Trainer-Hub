import logging
import os
import sys
import threading
from pathlib import Path
from contextlib import contextmanager

#TODO: Adattare se cambi la struttura directory in future
LOG_BASE_DIR = Path("logs")
LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Percorsi file log
SERVER_LOG_PATH = LOG_BASE_DIR / "server.log"
TRAINER_LOG_PATH = LOG_BASE_DIR / "trainer.log"
WEBSERVER_LOG_PATH = LOG_BASE_DIR / "webserver.log"
YOLO_LOG_PATH = LOG_BASE_DIR / "yolo_train.log"

# Colori ANSI per il terminale
COLORS = {
    'grey': '\033[90m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'bold_cyan': '\033[1;96m',
    'bold_yellow': '\033[1;93m',
    'bold_red': '\033[1;91m',
    'white': '\033[97m',
    'reset': '\033[0m',
    'highlight': '\033[48;5;208m'
}

# Livello log da env o default
log_level = getattr(logging, os.getenv("LOGGING_LEVEL", "INFO").upper(), logging.INFO)

# Formati colorati per terminale
FORMATS = {
    logging.INFO:    f"{COLORS['white']}%(asctime)s [{COLORS['bold_cyan']}INFO{COLORS['reset']}][%(name)s] %(message)s{COLORS['reset']}",
    logging.WARNING: f"{COLORS['yellow']}%(asctime)s [{COLORS['bold_yellow']}WARN{COLORS['reset']}][%(name)s] %(message)s{COLORS['reset']}",
    logging.ERROR:   f"{COLORS['red']}%(asctime)s [{COLORS['bold_red']}ERR {COLORS['reset']}][%(name)s] %(message)s{COLORS['reset']}",
}

# Formati puliti per file
FORMATS_FILE = {
    level: "%(asctime)s [%(levelname)s][%(name)s] %(message)s" for level in FORMATS
}

# Formatter terminale
class TerminalFormatter(logging.Formatter):
    def format(self, record):
        fmt = FORMATS.get(record.levelno, FORMATS[logging.INFO])
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)

# Formatter per i file log
class FileFormatter(logging.Formatter):
    def format(self, record):
        fmt = FORMATS_FILE.get(record.levelno, FORMATS_FILE[logging.INFO])
        msg = super().format(record)
        for code in COLORS.values():
            msg = msg.replace(code, "")
        return msg

# Global handler cache + lock
file_handlers = {}
file_lock = threading.Lock()

def get_file_handler(log_path: Path):
    with file_lock:
        if log_path not in file_handlers:
            handler = logging.FileHandler(log_path)
            handler.setLevel(log_level)
            handler.setFormatter(FileFormatter())
            file_handlers[log_path] = handler
        return file_handlers[log_path]

_configured_loggers = set()

def create_logger(name: str, log_file: Path = None):
    logger = logging.getLogger(name)
    if name in _configured_loggers:
        return logger

    logger.setLevel(log_level)
    logger.propagate = False
    logger.handlers.clear()

    # Aggiunge sempre console (eccetto per 'trainer')
    if name != "trainer":
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        console.setFormatter(TerminalFormatter())
        logger.addHandler(console)

    # File handler in base al tipo
    log_file = log_file or {
        "trainer": TRAINER_LOG_PATH,
        "webserver": WEBSERVER_LOG_PATH,
        "yolo": YOLO_LOG_PATH,
    }.get(name, SERVER_LOG_PATH)

    file_handler = get_file_handler(log_file)
    logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger

# Classe per silenziare log esterni e fare redirect
class LogManager:
    def __init__(self, log_path: Path = SERVER_LOG_PATH):
        self.log_path = log_path
        self.framework_loggers = [
            "anomalib", "ultralytics", "torch", "lightning", "pytorch_lightning"
        ]
        self._original = {}

    @contextmanager
    def redirect_logs(self):
        file = open(self.log_path, 'a')
        handler = logging.StreamHandler(file)
        handler.setLevel(log_level)
        handler.setFormatter(FileFormatter())

        root = logging.getLogger()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = file, file

        for name in self.framework_loggers:
            logger = logging.getLogger(name)
            self._original[name] = (logger.handlers[:], logger.level)
            logger.handlers = [handler]
            logger.setLevel(log_level)
            logger.propagate = False

        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            for name, (handlers, level) in self._original.items():
                logger = logging.getLogger(name)
                logger.handlers = handlers
                logger.setLevel(level)
