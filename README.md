# Text-Driven Video Search Pipeline

A comprehensive offline, keyframe-based retrieval pipeline for text-driven video search using computer vision and sentence transformers. This system enables searching through surveillance footage using natural language queries like "a black bag" or "person walking".

## Features

- **Frame Extraction**: Intelligent keyframe extraction with shot boundary detection and deduplication
- **Semantic Feature Extraction**: Computer vision features + Sentence Transformers for text encoding
- **Fast Retrieval**: FAISS-based indexing for efficient similarity search
- **Enhanced Search**: Novel enhancements including query expansion, temporal clustering, and multi-modal fusion
- **Object Grounding**: Optional caption-based object detection
- **Interactive Interface**: Command-line and interactive search modes

## Lightweight vs Full Pipeline Comparison

| Feature | Lightweight | Full Pipeline |
|---------|-------------|---------------|
| **Model Size** | ~80MB (sentence transformer) | ~1GB (BLIP + sentence transformer) |
| **Memory Usage** | 2-4GB RAM | 6-8GB RAM |
| **Setup Time** | 2-3 minutes | 10-15 minutes |
| **Processing Speed** | Fast | Moderate |
| **Caption Quality** | Basic (CV-based) | High (BLIP) |
| **Search Accuracy** | Good | Excellent |
| **Dependencies** | Minimal | Heavy |
| **Best For** | Quick setup, basic search | Research, production |

### When to Use Lightweight Version:
- Quick prototyping and testing
- Limited computational resources
- Simple search requirements
- Educational purposes

### When to Use Full Pipeline:
- Production deployments
- Maximum search accuracy needed
- Rich caption generation required
- Research applications

## Architecture Overview

```
Videos → Frame Extraction → Feature Extraction → Indexing → Enhanced Retrieval
  ↓           ↓                    ↓              ↓           ↓
.mp4/.avi → Keyframes → BLIP Captions + Sentence Transformer Embeddings → FAISS Index → Search Results
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- At least 4GB RAM (8GB for full pipeline)

### Setup Options

#### Option 1: Lightweight Version (Recommended)
For faster setup and lower resource usage:

```bash
pip install -r requirements_lightweight.txt
```

This version uses:
- Sentence Transformers for text encoding
- Simple computer vision for image features (instead of BLIP)
- No heavy vision-language models

#### Option 2: Full Pipeline
For maximum accuracy with heavy models:

```bash
pip install -r requirements.txt
```

This includes BLIP and advanced features.

3. Download required models (done automatically on first run):
   - **Lightweight**: Sentence transformer model (~80MB)
   - **Full**: Sentence transformer model (~80MB)

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

#### Lightweight Version:
```bash
python lightweight_pipeline.py build
```

#### Full Pipeline:
```bash
python video_search_pipeline.py build
```

This will:
- Extract keyframes from all videos
- **Lightweight**: Generate simple captions + extract visual features + sentence transformer embeddings
- **Full**: Generate captions using BLIP + sentence transformer embeddings
- Build FAISS index for fast search

### 3. Search for Frames

#### Lightweight Version:
```bash
python lightweight_pipeline.py search "a black bag"
```

#### Full Pipeline:
```bash
python video_search_pipeline.py search "a black bag"
```

### 4. Interactive Mode

#### Lightweight Version:
```bash
python lightweight_pipeline.py interactive
```

#### Full Pipeline:
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

### Interactive Mode

Start interactive mode for multiple searches:

```bash
python video_search_pipeline.py interactive
```

Available commands:
- `search <query>` - Search for frames
- `stats` - Show pipeline statistics
- `quit` - Exit

Example session:
```
> search a person carrying a black bag
> stats
> search red car
> quit
```

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
# CLIP not used; we use a sentence transformer
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

## API Usage

### Python API

```python
from video_search_pipeline import VideoSearchPipeline

# Initialize pipeline
pipeline = VideoSearchPipeline()
pipeline.setup_pipeline()

# Build index
pipeline.build_index(Path("data/videos"))

# Load existing index
pipeline.load_index()

# Search
results = pipeline.search("a person with a black bag")

# Get thumbnails
thumbnails = pipeline.get_result_thumbnails(results)
```

### Individual Components

```python
from frame_extractor import extract_frames_from_videos
from feature_extractor import extract_features_from_frames
from retrieval_system import build_retrieval_system

# Extract frames
frames = extract_frames_from_videos(video_dir, output_dir)

# Extract features
features = extract_features_from_frames(frames, features_dir)

# Build retrieval system
retrieval_system = build_retrieval_system(features_file, index_dir)
```

## Performance Optimization

### GPU Acceleration

The system automatically uses GPU if available. To force CPU usage:

```bash
CUDA_VISIBLE_DEVICES="" python video_search_pipeline.py build
```

### Memory Optimization

For large video collections:

1. Reduce `FRAME_EXTRACTION_RATE` to extract fewer frames
2. Increase `PERCEPTUAL_HASH_THRESHOLD` for more aggressive deduplication
3. Use `FAISS_INDEX_TYPE = "IVF"` for faster search with large indices

### Batch Processing

The system processes videos in batches. Adjust batch sizes in the configuration:

```python
# In feature_extractor.py
batch_size = 8  # Reduce if memory issues
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size in feature extraction
   - Use CPU instead of GPU
   - Process videos in smaller batches

2. **Model Download Fails**
   - Check internet connection
   - Manually download models to `~/.cache/transformers/`

3. **Poor Search Results**
   - Enable enhanced retrieval
   - Try different query formulations
   - Check if captions are meaningful

4. **Slow Performance**
   - Use GPU acceleration
   - Reduce number of extracted frames
   - Use IVF index type for large datasets

### Debug Mode

Enable debug logging:

```python
# In config.py
LOG_LEVEL = "DEBUG"
```

### Log Files

Check logs in `data/logs/video_search.log` for detailed information.

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
