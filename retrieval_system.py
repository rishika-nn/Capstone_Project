"""
Indexing and retrieval system using FAISS for fast similarity search.
"""

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import torch
import torch.nn.functional as F

from config import (
    FAISS_INDEX_TYPE,
    FAISS_NLIST,
    FAISS_NPROBE,
    TOP_K_RESULTS,
    RERANK_TOP_K,
    INDEX_DIR
)
from feature_extractor import LightweightFeatureExtractor

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS-based index for fast similarity search."""
    
    def __init__(self, embedding_dim: int, index_type: str = FAISS_INDEX_TYPE):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata = []
        
    def build_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Array of embedding vectors
            metadata: List of metadata for each embedding
        """
        logger.info(f"Building FAISS index for {len(embeddings)} embeddings...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        if self.index_type == "IVF":
            # IVF (Inverted File) index
            quantizer = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, FAISS_NLIST)
            
            # Train the index
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            
        elif self.index_type == "HNSW":
            # HNSW (Hierarchical Navigable Small World) index
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efConstruction = 200
            
        else:
            # Flat index (exact search)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Set search parameters for IVF
        if self.index_type == "IVF":
            self.index.nprobe = FAISS_NPROBE
        
        self.metadata = metadata
        logger.info(f"Index built successfully with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                result = {
                    "index": int(idx),
                    "score": float(score),
                    "metadata": self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def save_index(self, index_path: Path):
        """Save index and metadata to disk."""
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path / "faiss_index.bin"))
        
        # Save metadata
        with open(index_path / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save index configuration
        config = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "num_vectors": self.index.ntotal
        }
        with open(index_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Index saved to {index_path}")
    
    def load_index(self, index_path: Path):
        """Load index and metadata from disk."""
        # Load FAISS index
        self.index = faiss.read_index(str(index_path / "faiss_index.bin"))
        
        # Load metadata
        with open(index_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load configuration
        with open(index_path / "config.json", 'r') as f:
            config = json.load(f)
            self.embedding_dim = config["embedding_dim"]
            self.index_type = config["index_type"]
        
        logger.info(f"Index loaded from {index_path} with {self.index.ntotal} vectors")


class VideoRetrievalSystem:
    """Main retrieval system for video search."""
    
    def __init__(self, feature_extractor: Optional[LightweightFeatureExtractor] = None):
        # Avoid NameError in some environments by not referencing old class names
        self.feature_extractor = feature_extractor or LightweightFeatureExtractor()
        self.faiss_index = None
        self.caption_index = {}  # Text-based index for caption search
        
    def build_retrieval_index(self, features: List[Dict], index_dir: Path):
        """
        Build complete retrieval index from extracted features.
        
        Args:
            features: List of feature dictionaries
            index_dir: Directory to save index files
        """
        logger.info("Building retrieval index...")
        
        # Extract embeddings and metadata
        embeddings = []
        metadata = []
        
        for feature in features:
            embeddings.append(feature["embedding"])
            
            # Create metadata entry
            meta = {
                "image_path": feature["image_path"],
                "caption": feature["caption"],
                "embedding_dim": feature["embedding_dim"]
            }
            
            # Add grounding information if available
            if "detected_objects" in feature:
                meta["detected_objects"] = feature["detected_objects"]
                meta["grounding_score"] = feature.get("grounding_score", 0.0)
            
            metadata.append(meta)
            
            # Build caption index
            caption_words = feature["caption"].lower().split()
            for word in caption_words:
                if word not in self.caption_index:
                    self.caption_index[word] = []
                self.caption_index[word].append(len(metadata) - 1)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Build FAISS index
        self.faiss_index = FAISSIndex(embeddings.shape[1])
        self.faiss_index.build_index(embeddings, metadata)
        
        # Save index
        self.faiss_index.save_index(index_dir)
        
        # Save caption index
        with open(index_dir / "caption_index.json", 'w') as f:
            json.dump(self.caption_index, f, indent=2)
        
        logger.info("Retrieval index built successfully")
    
    def load_retrieval_index(self, index_dir: Path):
        """Load retrieval index from disk."""
        # Load FAISS index
        self.faiss_index = FAISSIndex(0)  # Will be set by load_index
        self.faiss_index.load_index(index_dir)
        
        # Load caption index
        with open(index_dir / "caption_index.json", 'r') as f:
            self.caption_index = json.load(f)
        
        logger.info("Retrieval index loaded successfully")
    
    def search_by_text(self, query: str, k: int = TOP_K_RESULTS, 
                      use_caption_filtering: bool = True) -> List[Dict]:
        """
        Search for frames using text query.
        
        Args:
            query: Text query string
            k: Number of results to return
            use_caption_filtering: Whether to use caption-based filtering
            
        Returns:
            List of search results
        """
        if self.faiss_index is None:
            raise ValueError("Index not loaded. Call load_retrieval_index() first.")
        
        # Extract query embedding
        query_embedding = self.feature_extractor.extract_text_embedding(query)
        
        # Perform vector similarity search
        if use_caption_filtering:
            # Get more results for filtering
            results = self.faiss_index.search(query_embedding, RERANK_TOP_K)
            
            # Filter and rerank by caption relevance
            filtered_results = self._filter_by_caption_relevance(results, query)
            results = filtered_results[:k]
        else:
            results = self.faiss_index.search(query_embedding, k)
        
        return results
    
    def _filter_by_caption_relevance(self, results: List[Dict], 
                                   query: str) -> List[Dict]:
        """
        Filter results by caption relevance.
        
        Args:
            results: Initial search results
            query: Query text
            
        Returns:
            Filtered and reranked results
        """
        query_words = set(query.lower().split())
        
        filtered_results = []
        for result in results:
            caption_words = set(result["metadata"]["caption"].lower().split())
            
            # Compute word overlap
            word_overlap = len(query_words.intersection(caption_words))
            caption_score = word_overlap / len(query_words) if query_words else 0
            
            # Combine vector similarity and caption relevance
            combined_score = 0.7 * result["score"] + 0.3 * caption_score
            
            filtered_result = result.copy()
            filtered_result["score"] = combined_score
            filtered_result["caption_score"] = caption_score
            filtered_results.append(filtered_result)
        
        # Sort by combined score
        filtered_results.sort(key=lambda x: x["score"], reverse=True)
        return filtered_results
    
    def search_with_grounding(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Search with object grounding for better precision.
        
        Args:
            query: Text query
            k: Number of results to return
            
        Returns:
            Search results with grounding information
        """
        # Get initial results
        results = self.search_by_text(query, RERANK_TOP_K, use_caption_filtering=True)
        
        # Boost results with detected objects
        grounded_results = []
        for result in results:
            metadata = result["metadata"]
            
            # Check for grounding information
            if "detected_objects" in metadata:
                grounding_score = metadata.get("grounding_score", 0.0)
                result["score"] *= (1.0 + grounding_score * 0.5)  # Boost by grounding score
            
            grounded_results.append(result)
        
        # Sort by boosted scores
        grounded_results.sort(key=lambda x: x["score"], reverse=True)
        return grounded_results[:k]
    
    def get_frame_thumbnail(self, result: Dict, thumbnail_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Get thumbnail image for a search result.
        
        Args:
            result: Search result dictionary
            thumbnail_size: Size of thumbnail to return
            
        Returns:
            Thumbnail image as numpy array
        """
        from PIL import Image
        
        image_path = Path(result["metadata"]["image_path"])
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(thumbnail_size)
            return np.array(image)
        except Exception as e:
            logger.error(f"Failed to load thumbnail for {image_path}: {e}")
            return np.zeros((*thumbnail_size, 3), dtype=np.uint8)
    
    def remove_temporal_duplicates(self, results: List[Dict], 
                                 temporal_threshold: float = 5.0) -> List[Dict]:
        """
        Remove temporally close duplicate results.
        
        Args:
            results: Search results
            temporal_threshold: Maximum time difference in seconds for duplicates
            
        Returns:
            Filtered results with temporal duplicates removed
        """
        if not results:
            return results
        
        filtered_results = []
        seen_timestamps = []
        
        for result in results:
            # Extract timestamp from image path
            image_path = Path(result["metadata"]["image_path"])
            try:
                # Assuming timestamp is in filename format: video_frame_XXXXXX_timestamp.jpg
                timestamp_str = image_path.stem.split('_')[-1].replace('s', '')
                timestamp = float(timestamp_str)
            except:
                # If timestamp extraction fails, keep the result
                filtered_results.append(result)
                continue
            
            # Check if this timestamp is too close to any seen timestamp
            is_duplicate = False
            for seen_time in seen_timestamps:
                if abs(timestamp - seen_time) < temporal_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_results.append(result)
                seen_timestamps.append(timestamp)
        
        return filtered_results


def build_retrieval_system(features_file: Path, index_dir: Path) -> VideoRetrievalSystem:
    """
    Build and save retrieval system from features.
    
    Args:
        features_file: Path to features JSON file
        index_dir: Directory to save index
        
    Returns:
        Built retrieval system
    """
    # Load features
    with open(features_file, 'r') as f:
        features = json.load(f)
    
    # Build retrieval system
    retrieval_system = VideoRetrievalSystem()
    retrieval_system.build_retrieval_index(features, index_dir)
    
    return retrieval_system


def load_retrieval_system(index_dir: Path) -> VideoRetrievalSystem:
    """
    Load retrieval system from saved index.
    
    Args:
        index_dir: Directory containing saved index
        
    Returns:
        Loaded retrieval system
    """
    retrieval_system = VideoRetrievalSystem()
    retrieval_system.load_retrieval_index(index_dir)
    
    return retrieval_system


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python retrieval_system.py <features_file> <index_directory> [query]")
        sys.exit(1)
    
    features_file = Path(sys.argv[1])
    index_dir = Path(sys.argv[2])
    
    # Build retrieval system
    retrieval_system = build_retrieval_system(features_file, index_dir)
    
    # If query provided, perform search
    if len(sys.argv) > 3:
        query = " ".join(sys.argv[3:])
        results = retrieval_system.search_by_text(query)
        
        print(f"Search results for '{query}':")
        for i, result in enumerate(results[:5]):  # Show top 5
            print(f"{i+1}. Score: {result['score']:.3f}")
            print(f"   Caption: {result['metadata']['caption']}")
            print(f"   Path: {result['metadata']['image_path']}")
            print()

