# Text-Driven Video Search Pipeline

A comprehensive offline, keyframe-based retrieval pipeline for text-driven video search using computer vision and sentence transformers. This system enables searching through surveillance footage using natural language queries like "a black bag" or "person walking".

## Features

- **Frame Extraction**: Intelligent keyframe extraction with shot boundary detection and deduplication
- **Semantic Feature Extraction**: Computer vision features + Sentence Transformers for text encoding
- **Fast Retrieval**: FAISS-based indexing for efficient similarity search
- **Enhanced Search**: Novel enhancements including query expansion, temporal clustering, and multi-modal fusion
- **Object Grounding**: Optional caption-based object detection
- **Interactive Interface**: Command-line and interactive search modes

<!-- Removed legacy lightweight/full comparison to streamline README -->

## Architecture Overview

```
Videos → Frame Extraction → Feature Extraction → Indexing → Enhanced Retrieval
  ↓           ↓                    ↓              ↓           ↓
.mp4/.avi → Keyframes → BLIP Captions + Sentence Transformer Embeddings → FAISS Index → Search Results
```

## Installation

Prerequisites: Python 3.8+, optional CUDA GPU. Install deps:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Videos

Place your video files in the `data/videos/` directory. Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`

```
data/
└── videos/
    ├── camera1_2023-01-01.mp4
    ├── camera2_2023-01-01.mp4
    └── ...
```

### 2. Build the Search Index
```bash
python video_search_pipeline.py build
```

This will extract keyframes, generate captions + embeddings, and build a FAISS index.

### Run on Google Colab

1. Clone and install
```bash
!git clone https://github.com/rishika-nn/Capstone_Project capstone
%cd capstone
!pip install -r requirements.txt
```

2. Provide videos in `data/videos/`
- Upload via Colab left Files pane into `data/videos/`, or mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
```bash
!mkdir -p data
!rm -f data/videos
!ln -s "/content/drive/MyDrive/your_videos_folder" data/videos
!ls -lah data/videos
```

3. Build index
```bash
!python video_search_pipeline.py build --segment-captions --object-tags --force-rebuild
```

4. Search
```bash
!python video_search_pipeline.py search "a black bag" --max-results 20
```

Notes:
- The pipeline automatically falls back to a Flat FAISS index for small datasets (too few vectors for IVF training).
- If you previously had a partial build, clean outputs and rebuild:
```bash
!rm -rf data/keyframes data/features data/index
!python video_search_pipeline.py build --force-rebuild
```

### 3. Search for Frames
```bash
python video_search_pipeline.py search "a black bag"
```

### 4. Interactive Mode
```bash
python video_search_pipeline.py interactive
```

## Detailed Usage

### Command Line Interface

#### Build Index
```bash
# Basic build
python video_search_pipeline.py build

# Force rebuild existing index
python video_search_pipeline.py build --force-rebuild

# Specify custom videos directory
python video_search_pipeline.py build --videos-dir /path/to/videos
```

#### Search
```bash
# Basic search
python video_search_pipeline.py search "person with backpack"

# Search with more results
python video_search_pipeline.py search "black bag" --max-results 20

# Disable enhanced retrieval (faster, less accurate)
python video_search_pipeline.py search "car" --no-enhancements

# Save results to directory
python video_search_pipeline.py search "person walking" --save-results ./results
```

#### Statistics
```bash
python video_search_pipeline.py stats
```

### Interactive commands
- `search <query>`
- `stats`
- `quit`

## File Structure

```
├── config.py                    # Configuration settings
├── frame_extractor.py           # Frame extraction and deduplication
├── feature_extractor.py         # CV + Sentence Transformer feature extraction
├── retrieval_system.py          # FAISS indexing and retrieval
├── enhancements.py              # Novel enhancements (query expansion, clustering)
├── video_search_pipeline.py     # Main pipeline orchestration
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── data/                        # Data directory (created automatically)
    ├── videos/                  # Input video files
    ├── keyframes/               # Extracted keyframes
    ├── features/                # Extracted features and embeddings
    ├── index/                   # FAISS index files
    └── logs/                    # Log files

Required contents before build:
- `data/videos/` must contain at least one video file (.mp4/.avi/.mov/.mkv/.flv/.wmv)

Expected contents after build:
- `data/keyframes/*.jpg` and per-video metadata JSON
- `data/features/frame_features.json`, `embeddings.npy`, `feature_metadata.json`
- `data/index/faiss_index.bin`, `metadata.json`, `config.json`, `caption_index.json`
```

## Configuration

Edit `config.py` to customize the pipeline:

### Video Processing
```python
FRAME_EXTRACTION_RATE = 2        # frames per second
MAX_KEYFRAMES_PER_SHOT = 5       # max keyframes per shot
SHOT_DETECTION_THRESHOLD = 0.3   # shot boundary sensitivity
```

### Deduplication
```python
PERCEPTUAL_HASH_THRESHOLD = 5    # Hamming distance for duplicates
SIMILARITY_THRESHOLD = 0.95      # Feature similarity threshold
```

### Retrieval
```python
TOP_K_RESULTS = 20               # Default number of results
FAISS_INDEX_TYPE = "IVF"         # Index type: "IVF", "HNSW", or "Flat"
```

### Models
```python
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
```

## Advanced Features

### 1. Query Expansion

The system automatically expands queries with synonyms and contextual information:

- **Synonym Expansion**: "bag" → ["backpack", "briefcase", "purse"]
- **Contextual Expansion**: "black bag" → ["person with black bag", "scene showing black bag"]
- **Semantic Expansion**: Uses corpus captions to find related terms

### 2. Temporal Clustering

Groups temporally close and visually similar frames:

```python
# Enable temporal clustering in search
results = pipeline.search("black bag", use_clustering=True)
```

### 3. Multi-Modal Fusion

Combines visual and textual signals for better ranking:

- Visual similarity (approx via text similarity to captions)
- Textual similarity (caption matching)
- Word overlap analysis
- Grounding-based boosting

### 4. Object Grounding

Optional object detection for improved precision:

```python
# Enable object grounding
results = pipeline.search("black bag", use_grounding=True)
```

### 5. Domain Adaptation

Adapt embeddings to your specific environment:

```python
# Perform domain adaptation
adapted_embeddings = domain_adaptation.adapt_to_campus_environment(
    embeddings, captions
)
```

<!-- API usage examples removed to keep README concise -->

## Performance Tips (brief)
- Use GPU in Colab for faster feature extraction.
- For large datasets, reduce `FRAME_EXTRACTION_RATE` in `config.py`.

## Troubleshooting (quick)
- No videos found: ensure files exist in `data/videos/` or pass `--videos-dir`.
- Partial/failed build: `rm -rf data/keyframes data/features data/index` then rebuild with `--force-rebuild`.
- Small dataset IVF error: the system auto-falls back to Flat index.
- Search errors: recent updates pad query embeddings to index dim; rebuild if you changed feature shapes.

## Evaluation

### Metrics

The system provides several metrics for evaluation:

- **Retrieval Precision**: Fraction of relevant results in top-K
- **Caption Quality**: Measure via human eval or cosine to queries
- **Temporal Coherence**: Consistency across time
- **Grounding Accuracy**: Object detection precision

### Benchmarking

```python
# Evaluate on test queries
test_queries = [
    "a person with a black bag",
    "red car in parking lot",
    "person walking on sidewalk"
]

for query in test_queries:
    results = pipeline.search(query)
    # Evaluate results...
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this system in your research, please cite:

```bibtex
@article{text_driven_video_search,
  title={Text-Driven Video Search Using CV and Sentence Transformers},
  author={Your Name},
  journal={Your Conference},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [BLIP](https://github.com/salesforce/BLIP) for image captioning
- Sentence Transformers for efficient text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Transformers](https://huggingface.co/transformers/) for model integration

## Future Work

- [ ] Real-time video processing
- [ ] Multi-camera synchronization
- [ ] Advanced temporal reasoning
- [ ] Fine-grained object detection
- [ ] Web interface
- [ ] Mobile app integration
