import nltk
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.runtime_setup import configure_local_runtime

configure_local_runtime()

nltk.download('punkt')
nltk.download('stopwords')
