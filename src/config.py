import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
INGREDIENTS_PATH = DATA_DIR / "ingredients.json"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
DOCUMENTS_DIR = KNOWLEDGE_BASE_DIR / "documents"
KNOWLEDGE_BASE_METADATA_PATH = KNOWLEDGE_BASE_DIR / "metadata.json"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RUNTIME_ARTIFACTS_DIR = ARTIFACTS_DIR / "runtime"


def load_env(env_path: Path | None = None) -> None:
    """Load environment variables from a .env file into os.environ without overriding existing ones."""
    target = env_path or (PROJECT_ROOT / ".env")
    if not target.is_file():
        return
    try:
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass


load_env()
