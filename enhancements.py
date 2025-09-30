"""
Novel enhancements for the video search pipeline including multi-modal fusion,
temporal clustering, query expansion, and domain adaptation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image

from config import (
    TEMPORAL_CLUSTER_THRESHOLD,
    MAX_TEMPORAL_GAP,
    QUERY_EXPANSION_MODEL,
    MAX_EXPANDED_QUERIES
)

logger = logging.getLogger(__name__)


class QueryExpansion:
    """Query expansion using language models and semantic similarity."""
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        
    def expand_query_with_synonyms(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query string
            
        Returns:
            List of expanded queries
        """
        # Simple synonym mapping (in practice, use WordNet or pre-trained embeddings)
        synonym_dict = {
            "bag": ["backpack", "briefcase", "purse", "handbag", "suitcase"],
            "person": ["man", "woman", "people", "individual", "human"],
            "car": ["vehicle", "automobile", "truck", "van", "sedan"],
            "black": ["dark", "dark-colored", "ebony", "charcoal"],
            "white": ["light", "pale", "ivory", "cream"],
            "red": ["crimson", "scarlet", "burgundy", "maroon"],
            "blue": ["navy", "azure", "cobalt", "royal"],
            "walking": ["walking", "moving", "strolling", "walking around"],
            "running": ["running", "jogging", "sprinting", "rushing"],
            "standing": ["standing", "stationary", "still", "motionless"]
        }
        
        expanded_queries = [query]
        query_lower = query.lower()
        
        for word, synonyms in synonym_dict.items():
            if word in query_lower:
                for synonym in synonyms[:2]:  # Limit to 2 synonyms per word
                    expanded_query = query_lower.replace(word, synonym)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
        
        return expanded_queries[:MAX_EXPANDED_QUERIES]
    
    def expand_query_with_context(self, query: str) -> List[str]:
        """
        Expand query by adding contextual information.
        
        Args:
            query: Original query string
            
        Returns:
            List of contextually expanded queries
        """
        context_templates = [
            "a person with {query}",
            "someone carrying {query}",
            "a scene showing {query}",
            "an image containing {query}",
            "{query} in the background",
            "{query} in the foreground"
        ]
        
        expanded_queries = [query]
        for template in context_templates:
            expanded_query = template.format(query=query)
            expanded_queries.append(expanded_query)
        
        return expanded_queries[:MAX_EXPANDED_QUERIES]
    
    def expand_query_semantic(self, query: str, corpus_captions: List[str], 
                            top_k: int = 5) -> List[str]:
        """
        Expand query using semantic similarity with corpus captions.
        
        Args:
            query: Original query string
            corpus_captions: List of captions from the video corpus
            top_k: Number of similar captions to use for expansion
            
        Returns:
            List of semantically expanded queries
        """
        query_embedding = self.feature_extractor.extract_text_embedding(query)
        
        # Compute similarities with all captions
        similarities = []
        for caption in corpus_captions:
            caption_embedding = self.feature_extractor.extract_text_embedding(caption)
            similarity = np.dot(query_embedding, caption_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(caption_embedding)
            )
            similarities.append((caption, similarity))
        
        # Sort by similarity and get top captions
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_captions = [caption for caption, _ in similarities[:top_k]]
        
        # Extract key phrases from top captions
        expanded_queries = [query]
        for caption in top_captions:
            # Simple phrase extraction (in practice, use NLP tools)
            words = caption.split()
            if len(words) >= 3:
                # Extract noun phrases or important words
                key_phrase = " ".join(words[:3])  # Simple approach
                if key_phrase not in expanded_queries:
                    expanded_queries.append(key_phrase)
        
        return expanded_queries[:MAX_EXPANDED_QUERIES]


class TemporalClustering:
    """Temporal clustering for grouping similar frames across time."""
    
    def __init__(self, similarity_threshold: float = TEMPORAL_CLUSTER_THRESHOLD):
        self.similarity_threshold = similarity_threshold
    
    def cluster_temporal_neighbors(self, results: List[Dict]) -> List[List[Dict]]:
        """
        Cluster search results by temporal proximity and visual similarity.
        
        Args:
            results: List of search results with metadata
            
        Returns:
            List of temporal clusters
        """
        if not results:
            return []
        
        # Extract timestamps and embeddings
        timestamps = []
        embeddings = []
        valid_results = []
        
        for result in results:
            try:
                # Extract timestamp from image path
                image_path = Path(result["metadata"]["image_path"])
                timestamp_str = image_path.stem.split('_')[-1].replace('s', '')
                timestamp = float(timestamp_str)
                
                # Get embedding from metadata (if available)
                if "embedding" in result["metadata"]:
                    embedding = np.array(result["metadata"]["embedding"])
                else:
                    # Use score as simple embedding
                    embedding = np.array([result["score"]])
                
                timestamps.append(timestamp)
                embeddings.append(embedding)
                valid_results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to process result: {e}")
                continue
        
        if not embeddings:
            return [results]
        
        embeddings = np.array(embeddings)
        
        # Combine temporal and visual features
        temporal_features = self._extract_temporal_features(timestamps)
        combined_features = np.hstack([embeddings, temporal_features])
        
        # Cluster using DBSCAN
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,
            min_samples=2,
            metric='cosine'
        )
        cluster_labels = clustering.fit_predict(combined_features)
        
        # Group results by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_results[i])
        
        # Add noise points as individual clusters
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise
                clusters[f"noise_{i}"] = [valid_results[i]]
        
        return list(clusters.values())
    
    def _extract_temporal_features(self, timestamps: List[float]) -> np.ndarray:
        """Extract temporal features for clustering."""
        features = []
        
        for i, timestamp in enumerate(timestamps):
            # Temporal features
            temporal_feature = [
                timestamp / 3600,  # Hour of day
                timestamp % 3600 / 60,  # Minute of hour
                np.sin(timestamp * 2 * np.pi / 86400),  # Daily cycle
                np.cos(timestamp * 2 * np.pi / 86400)
            ]
            
            # Add relative temporal features
            if i > 0:
                time_diff = timestamp - timestamps[i-1]
                temporal_feature.extend([
                    time_diff,
                    min(time_diff, MAX_TEMPORAL_GAP) / MAX_TEMPORAL_GAP
                ])
            else:
                temporal_feature.extend([0, 0])
            
            features.append(temporal_feature)
        
        return np.array(features)
    
    def select_representative_frames(self, clusters: List[List[Dict]]) -> List[Dict]:
        """
        Select representative frame from each temporal cluster.
        
        Args:
            clusters: List of temporal clusters
            
        Returns:
            List of representative frames
        """
        representatives = []
        
        for cluster in clusters:
            if len(cluster) == 1:
                representatives.append(cluster[0])
            else:
                # Select frame with highest score
                best_frame = max(cluster, key=lambda x: x["score"])
                representatives.append(best_frame)
        
        return representatives


class MultiModalFusion:
    """Multi-modal fusion for combining visual and textual signals."""
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def fuse_visual_textual_signals(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Fuse visual and textual signals for better ranking.
        
        Args:
            results: Initial search results
            query: Query text
            
        Returns:
            Results with fused scores
        """
        query_embedding = self.feature_extractor.extract_text_embedding(query)
        
        enhanced_results = []
        for result in results:
            # Visual similarity not available (no image embeddings in metadata).
            # Use caption/text similarity only, plus original score and word overlap.
            visual_similarity = 0.0
            
            # Get textual similarity (caption-based)
            caption = result["metadata"]["caption"]
            caption_embedding = self.feature_extractor.extract_text_embedding(caption)
            textual_similarity = np.dot(query_embedding, caption_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(caption_embedding)
            )
            
            # Compute word overlap
            query_words = set(query.lower().split())
            caption_words = set(caption.lower().split())
            word_overlap = len(query_words.intersection(caption_words)) / len(query_words)
            
            # Fuse scores
            fused_score = (
                0.55 * textual_similarity +
                0.35 * word_overlap +
                0.10 * result["score"]  # Original score
            )
            
            enhanced_result = result.copy()
            enhanced_result["score"] = fused_score
            enhanced_result["visual_similarity"] = visual_similarity
            enhanced_result["textual_similarity"] = textual_similarity
            enhanced_result["word_overlap"] = word_overlap
            
            enhanced_results.append(enhanced_result)
        
        # Sort by fused score
        enhanced_results.sort(key=lambda x: x["score"], reverse=True)
        return enhanced_results
    
    def apply_object_grounding(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Apply object grounding to boost relevant results.
        
        Args:
            results: Search results
            query: Query text
            
        Returns:
            Results with grounding-based boosting
        """
        # Extract object classes from query
        object_classes = self._extract_object_classes(query)
        
        grounded_results = []
        for result in results:
            # Check for detected objects
            detected_objects = result["metadata"].get("detected_objects", [])
            
            # Compute grounding score
            grounding_score = 0.0
            for obj in detected_objects:
                if obj["class"] in object_classes:
                    grounding_score = max(grounding_score, obj["score"])
            
            # Boost score based on grounding
            boosted_score = result["score"] * (1.0 + grounding_score * 0.5)
            
            grounded_result = result.copy()
            grounded_result["score"] = boosted_score
            grounded_result["grounding_boost"] = grounding_score
            
            grounded_results.append(grounded_result)
        
        # Sort by boosted score
        grounded_results.sort(key=lambda x: x["score"], reverse=True)
        return grounded_results
    
    def _extract_object_classes(self, query: str) -> List[str]:
        """Extract object classes from query text."""
        object_classes = [
            "person", "bag", "backpack", "briefcase", "car", "bicycle", "motorcycle",
            "dog", "cat", "bird", "chair", "table", "laptop", "phone", "book",
            "hat", "shirt", "pants", "shoes", "glasses", "watch"
        ]
        
        query_lower = query.lower()
        found_objects = [obj for obj in object_classes if obj in query_lower]
        
        return found_objects


class DomainAdaptation:
    """Domain adaptation for improving embeddings on specific data."""
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def perform_contrastive_adaptation(self, embeddings: np.ndarray, 
                                     captions: List[str], 
                                     num_epochs: int = 10) -> np.ndarray:
        """
        Perform self-supervised contrastive learning on embeddings.
        
        Args:
            embeddings: Initial embeddings
            captions: Corresponding captions
            num_epochs: Number of training epochs
            
        Returns:
            Adapted embeddings
        """
        logger.info("Performing contrastive adaptation...")
        
        # Simple contrastive learning implementation
        # In practice, you would use a proper contrastive learning framework
        
        adapted_embeddings = embeddings.copy()
        
        for epoch in range(num_epochs):
            # Create positive pairs (similar captions)
            positive_pairs = self._create_positive_pairs(captions)
            
            # Update embeddings based on positive pairs
            for i, j in positive_pairs:
                if i < len(adapted_embeddings) and j < len(adapted_embeddings):
                    # Pull similar embeddings closer
                    sim = cosine_similarity([adapted_embeddings[i]], [adapted_embeddings[j]])[0, 0]
                    if sim < 0.9:  # Not too similar already
                        direction = adapted_embeddings[j] - adapted_embeddings[i]
                        adapted_embeddings[i] += 0.01 * direction
                        adapted_embeddings[j] -= 0.01 * direction
        
        logger.info("Contrastive adaptation completed")
        return adapted_embeddings
    
    def _create_positive_pairs(self, captions: List[str]) -> List[Tuple[int, int]]:
        """Create positive pairs based on caption similarity."""
        positive_pairs = []
        
        for i, caption1 in enumerate(captions):
            for j, caption2 in enumerate(captions[i+1:], i+1):
                # Simple similarity based on word overlap
                words1 = set(caption1.lower().split())
                words2 = set(caption2.lower().split())
                overlap = len(words1.intersection(words2))
                
                if overlap >= 2:  # At least 2 common words
                    positive_pairs.append((i, j))
        
        return positive_pairs[:100]  # Limit number of pairs
    
    def adapt_to_campus_environment(self, embeddings: np.ndarray, 
                                  captions: List[str]) -> np.ndarray:
        """
        Adapt embeddings specifically for campus environment.
        
        Args:
            embeddings: Initial embeddings
            captions: Corresponding captions
            
        Returns:
            Campus-adapted embeddings
        """
        # Campus-specific keywords
        campus_keywords = [
            "student", "campus", "building", "classroom", "library", "cafeteria",
            "parking", "walkway", "bench", "tree", "grass", "sidewalk"
        ]
        
        # Boost embeddings that contain campus keywords
        adapted_embeddings = embeddings.copy()
        
        for i, caption in enumerate(captions):
            caption_lower = caption.lower()
            campus_score = sum(1 for keyword in campus_keywords if keyword in caption_lower)
            
            if campus_score > 0:
                # Boost embedding magnitude
                adapted_embeddings[i] *= (1.0 + campus_score * 0.1)
        
        # Normalize embeddings
        norms = np.linalg.norm(adapted_embeddings, axis=1, keepdims=True)
        adapted_embeddings = adapted_embeddings / (norms + 1e-8)
        
        return adapted_embeddings


class EnhancedRetrieval:
    """Combined enhancement system for improved video search."""
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.query_expansion = QueryExpansion(feature_extractor)
        self.temporal_clustering = TemporalClustering()
        self.multimodal_fusion = MultiModalFusion(feature_extractor)
        self.domain_adaptation = DomainAdaptation(feature_extractor)
    
    def enhanced_search(self, retrieval_system, query: str, 
                       use_expansion: bool = True,
                       use_clustering: bool = True,
                       use_fusion: bool = True,
                       use_grounding: bool = False) -> List[Dict]:
        """
        Perform enhanced search with all improvements.
        
        Args:
            retrieval_system: Base retrieval system
            query: Search query
            use_expansion: Whether to use query expansion
            use_clustering: Whether to use temporal clustering
            use_fusion: Whether to use multi-modal fusion
            use_grounding: Whether to use object grounding
            
        Returns:
            Enhanced search results
        """
        # Step 1: Query expansion
        if use_expansion:
            expanded_queries = self.query_expansion.expand_query_with_synonyms(query)
            expanded_queries.extend(self.query_expansion.expand_query_with_context(query))
        else:
            expanded_queries = [query]
        
        # Step 2: Search with expanded queries
        all_results = []
        for expanded_query in expanded_queries:
            if use_grounding:
                results = retrieval_system.search_with_grounding(expanded_query, RERANK_TOP_K)
            else:
                results = retrieval_system.search_by_text(expanded_query, RERANK_TOP_K)
            all_results.extend(results)
        
        # Step 3: Remove duplicates and merge results
        unique_results = self._merge_duplicate_results(all_results)
        
        # Step 4: Multi-modal fusion
        if use_fusion:
            unique_results = self.multimodal_fusion.fuse_visual_textual_signals(unique_results, query)
        
        # Step 5: Object grounding
        if use_grounding:
            unique_results = self.multimodal_fusion.apply_object_grounding(unique_results, query)
        
        # Step 6: Temporal clustering
        if use_clustering:
            clusters = self.temporal_clustering.cluster_temporal_neighbors(unique_results)
            unique_results = self.temporal_clustering.select_representative_frames(clusters)
        
        # Step 7: Final ranking and filtering
        final_results = unique_results[:TOP_K_RESULTS]
        
        return final_results
    
    def _merge_duplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Merge duplicate results from expanded queries."""
        seen_paths = set()
        unique_results = []
        
        for result in results:
            image_path = result["metadata"]["image_path"]
            if image_path not in seen_paths:
                seen_paths.add(image_path)
                unique_results.append(result)
        
        return unique_results


if __name__ == "__main__":
    # Example usage
    from feature_extractor import SemanticFeatureExtractor
    
    # Initialize enhancement system
    feature_extractor = SemanticFeatureExtractor()
    enhancement_system = EnhancedRetrieval(feature_extractor)
    
    # Example query expansion
    query = "a black bag"
    expanded = enhancement_system.query_expansion.expand_query_with_synonyms(query)
    print(f"Expanded queries: {expanded}")
    
    # Example temporal clustering
    sample_results = [
        {"score": 0.9, "metadata": {"image_path": "video1_frame_000001_10.0s.jpg", "caption": "a person with a bag"}},
        {"score": 0.8, "metadata": {"image_path": "video1_frame_000002_10.5s.jpg", "caption": "a person carrying a bag"}},
        {"score": 0.7, "metadata": {"image_path": "video1_frame_000010_15.0s.jpg", "caption": "a different scene"}}
    ]
    
    clusters = enhancement_system.temporal_clustering.cluster_temporal_neighbors(sample_results)
    print(f"Temporal clusters: {len(clusters)}")

