import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*huggingface_hub.*symlinks.*")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
INDEX_DIR = PROJECT_ROOT / "index"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5

# Faithfulness thresholds
ACCEPT_THRESHOLD = 0.6
REJECT_THRESHOLD = 0.4
SENTENCE_SUPPORT_THRESHOLD = 0.5
