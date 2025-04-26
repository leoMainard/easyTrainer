from importlib.resources import files
from easytrainer.config.enums import ResourcePath

def load_stopwords():
    file = ResourcePath.STOPWORDS_FR.value
    stopwords_path = files("easytrainer.resources").joinpath(file)
    try:
        with stopwords_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            return set(lines) if not lines else lines
    except FileNotFoundError:
        print(f"Stopwords file {file} not found.")
        return set()

def extract_params(value, default=None):
    if isinstance(value, tuple):
        return value
    return (value, default)