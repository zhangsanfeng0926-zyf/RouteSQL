import os
from pathlib import Path


def configure_local_runtime():
    root_dir = Path(__file__).resolve().parents[1]

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HUGGINGFACE_HUB_BASE_URL", os.environ["HF_ENDPOINT"])
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(root_dir / ".cache" / "sentence_transformers"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root_dir / ".cache" / "transformers"))

    nltk_dir = root_dir / "nltk_data"
    if nltk_dir.exists():
        os.environ.setdefault("NLTK_DATA", str(nltk_dir))
        try:
            import nltk

            nltk_path = str(nltk_dir)
            if nltk_path not in nltk.data.path:
                nltk.data.path.insert(0, nltk_path)
        except Exception:
            pass

    corenlp_new = root_dir / "third_party" / "stanford-corenlp-4.5.10"
    corenlp_legacy = root_dir / "third_party" / "stanford-corenlp-full-2018-10-05"
    if "CORENLP_HOME" not in os.environ:
        if corenlp_new.exists():
            os.environ["CORENLP_HOME"] = str(corenlp_new)
        elif corenlp_legacy.exists():
            os.environ["CORENLP_HOME"] = str(corenlp_legacy)
