from pathlib import Path
PRODUCTS_JSONL = Path(r"products_chunked.enriched.jsonl")
BM25_INDEX_DIR = Path(r"bm25_parent_index")

# QDRANT CONFIGURATION
QDRANT_URL = "XXXXX"
QDRANT_API_KEY = "XXXXX"
# Collection names
PRODUCT_COLLECTION = "product_bge"
FAQ_COLLECTION = "faq_bge"

# Embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# LLM configuration
LLM_MODEL_NAME = ""
LLM_LOAD_IN_4BIT = True
LLM_QUANT_TYPE = "nf4"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.9
LLM_REPETITION_PENALTY = 1.05

# RAG PARAMETERS
DEFAULT_TOPK = 8
DEFAULT_RRF_LAMBDA = 0.8
DEFAULT_PER_PARENT_CHUNKS = 2
DEFAULT_DENSE_TOPK_CHUNKS = 64
DEFAULT_BM25_TOPK_PARENTS = 50
DEFAULT_RRF_K = 60

# Rerank
USE_RERANK = True

# Filter parameters
STRICT_CATEGORY = False
BRAND_STRICT = False


# ANSWER POLICY
MIN_RESULTS_PRODUCT = 1
MIN_RESULTS_FAQ = 1
MIN_SCORE_PRODUCT = 0.0
MIN_SCORE_FAQ = 0.0

ALLOW_EXPAND_CHUNKS_DETAIL = False
EXPAND_CHUNKS_PER_PARENT_DETAIL = 2

TOKEN_BUDGET_EST_PRODUCT_BROWSE = 1800
TOKEN_BUDGET_EST_PRODUCT_DETAIL = 2000
TOKEN_BUDGET_EST_FAQ = 1500

PER_ITEM_CHAR_BUDGET_PRODUCT_BROWSE = 360
PER_ITEM_CHAR_BUDGET_PRODUCT_DETAIL = 700
PER_ITEM_CHAR_BUDGET_FAQ = 420


# UI CONFIGURATION (STREAMLIT)
PAGE_TITLE = "Tiki Chatbot RAG"
PAGE_ICON = "🛒"
LAYOUT = "wide"

# Theme colors
PRIMARY_COLOR = "#3b82f6"       # Blue 500
SECONDARY_COLOR = "#94a3b8"     # Slate 400
SUCCESS_COLOR = "#10b981"       # Emerald 500
WARNING_COLOR = "#f59e0b"       # Amber 500
DANGER_COLOR = "#ef4444"        # Red 500
INFO_COLOR = "#0ea5e9"          # Sky 500

# === DARK MODE COLORS -
BACKGROUND_COLOR = "#0f172a"
SECONDARY_BACKGROUND_COLOR = "#1e293b"
TEXT_COLOR = "#f1f5f9"
SECONDARY_TEXT_COLOR = "#cbd5e1"
BORDER_COLOR = "#334155"

# ADVANCED SETTINGS
# Cache settings
ENABLE_CACHING = True
CACHE_TTL = 3600  # seconds

# Performance
MAX_WORKERS = 4  # For parallel processing
TIMEOUT = 30  # seconds

# Feature flags
ENABLE_ANALYTICS = False
ENABLE_FEEDBACK = True
ENABLE_EXPORT = True



# VALIDATION
def validate_config():
    """Validate configuration and files"""
    errors = []

    # Check data files
    if not PRODUCTS_JSONL.exists():
        errors.append(f" Products JSONL not found: {PRODUCTS_JSONL}")
    else:
        print(f" Products JSONL found: {PRODUCTS_JSONL}")

    if not BM25_INDEX_DIR.exists():
        errors.append(f" BM25 index directory not found: {BM25_INDEX_DIR}")
    else:
        print(f" BM25 index directory found: {BM25_INDEX_DIR}")

    # Check required BM25 files
    required_bm25_files = ["parents.pkl", "docs_tokens.pkl", "meta.json", "params.json"]
    for fname in required_bm25_files:
        fpath = BM25_INDEX_DIR / fname
        if not fpath.exists():
            errors.append(f" BM25 file missing: {fpath}")
        else:
            print(f" BM25 file found: {fname}")

    # Check Qdrant credentials
    if not QDRANT_URL or not QDRANT_API_KEY:
        errors.append(" Qdrant credentials missing")
    else:
        print(" Qdrant credentials configured")

    if errors:
        print("\n" + "=" * 60)
        print(" CONFIGURATION ERRORS:")
        print("=" * 60)
        for error in errors:
            print(error)
        print("=" * 60)
        raise FileNotFoundError("\n".join(errors))

    print("\n" + "=" * 60)
    print(" CONFIGURATION VALIDATED SUCCESSFULLY!")
    print("=" * 60)
    return True


def print_config_summary():
    """Print configuration summary"""
    print(" CONFIGURATION SUMMARY")
    print(f" Products JSONL: {PRODUCTS_JSONL}")
    print(f" BM25 Index: {BM25_INDEX_DIR}")
    print(f" Qdrant URL: {QDRANT_URL[:50]}...")
    print(f" Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f" LLM Model: {LLM_MODEL_NAME}")
    print(f"️ Default Top-K: {DEFAULT_TOPK}")
    print(f"️ RRF Lambda: {DEFAULT_RRF_LAMBDA}")
    print(f"️ Temperature: {LLM_TEMPERATURE}")
    print(f" Page Title: {PAGE_TITLE}")
    print(f" Page Icon: {PAGE_ICON}")
    print("=" * 60 + "\n")



if __name__ == "__main__":
    # Run validation
    try:
        validate_config()
        print_config_summary()
        print(" Config file is ready")
    except Exception as e:
        print(f"\n Error: {e}")
        import sys
        sys.exit(1)