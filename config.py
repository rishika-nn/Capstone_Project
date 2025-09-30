"""
Configuration file for the text-driven video search system.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
FEATURES_DIR = DATA_DIR / "features"
INDEX_DIR = DATA_DIR / "index"

# Create directories if they don't exist
for dir_path in [DATA_DIR, VIDEOS_DIR, KEYFRAMES_DIR, FEATURES_DIR, INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Video processing parameters
FRAME_EXTRACTION_RATE = 2  # frames per second
MAX_KEYFRAMES_PER_SHOT = 5
SHOT_DETECTION_THRESHOLD = 0.3

# Deduplication parameters
PERCEPTUAL_HASH_THRESHOLD = 5  # Hamming distance threshold
SIMILARITY_THRESHOLD = 0.95

# Model parameters
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"

# FAISS parameters
FAISS_INDEX_TYPE = "IVF"  # or "HNSW"
FAISS_NLIST = 100
FAISS_NPROBE = 10

# Retrieval parameters
TOP_K_RESULTS = 20
RERANK_TOP_K = 50

# Query expansion parameters
QUERY_EXPANSION_MODEL = "gpt-3.5-turbo"  # or local model
MAX_EXPANDED_QUERIES = 5

# Clustering parameters
TEMPORAL_CLUSTER_THRESHOLD = 0.8
MAX_TEMPORAL_GAP = 300  # seconds

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "video_search.log"

# Create logs directory
LOG_FILE.parent.mkdir(exist_ok=True)
