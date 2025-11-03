"""
Video Frame Search Engine - Main Application
Production-ready video semantic search system with BLIP and Pinecone
"""

import os
import logging
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Import all modules
from video_search_config import Config
from frame_extractor import VideoFrameExtractor, FrameData
from caption_generator import BlipCaptionGenerator, CaptionedFrame
from embedding_generator import TextEmbeddingGenerator, EmbeddedFrame
from pinecone_manager import PineconeManager, SearchResult
from object_caption_pipeline import ObjectCaptionPipeline, ObjectCaption
from motion_analyzer import MotionAnalyzer
from temporal_bootstrapping import TemporalBootstrapper, BoostedResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_search_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoSearchEngine:
    """
    Complete video search engine integrating all components
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the video search engine
        
        Args:
            config: Configuration object (uses default if None)
        """
        # Use provided config or default
        self.config = config or Config()
        
        # Validate configuration
        try:
            self.config.validate()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        # Initialize components (lazy loading)
        self.frame_extractor = None
        self.caption_generator = None
        self.embedding_generator = None
        self.pinecone_manager = None
        self.object_pipeline = None  # Object-focused captioning pipeline
        self.motion_analyzer = None  # Motion analyzer for adaptive windows
        self.temporal_bootstrapper = None  # Temporal bootstrapping module
        
        # Processing state
        self.current_video = None
        self.processed_frames = []
        self.processing_stats = {}
        self.video_paths = {}  # Track video paths for motion analysis
        
        logger.info("Video Search Engine initialized")
    
    def _initialize_components(self):
        """Initialize all components if not already initialized"""
        if not self.frame_extractor:
            self.frame_extractor = VideoFrameExtractor(
                similarity_threshold=self.config.FRAME_SIMILARITY_THRESHOLD,
                max_frames=self.config.MAX_FRAMES_PER_VIDEO,
                resize_width=self.config.FRAME_RESIZE_WIDTH
            )
            logger.info("Frame extractor initialized")
        
        if not self.caption_generator:
            self.caption_generator = BlipCaptionGenerator(
                model_name=self.config.BLIP_MODEL,
                batch_size=self.config.BLIP_BATCH_SIZE,
                use_gpu=self.config.USE_GPU,
                generate_multiple_captions=getattr(self.config, 'GENERATE_MULTIPLE_CAPTIONS', False),
                captions_per_frame=getattr(self.config, 'CAPTIONS_PER_FRAME', 3)
            )
            logger.info("Caption generator initialized with multi-caption support")
        
        if not self.embedding_generator:
            self.embedding_generator = TextEmbeddingGenerator(
                model_name=self.config.EMBEDDING_MODEL,
                batch_size=self.config.EMBEDDING_BATCH_SIZE,
                use_gpu=self.config.USE_GPU,
                normalize=True
            )
            logger.info("Embedding generator initialized")
        
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
            logger.info("Pinecone manager initialized")
    
    def _initialize_temporal_bootstrapping(self):
        """Initialize temporal bootstrapping components if not already initialized"""
        if not self.motion_analyzer:
            self.motion_analyzer = MotionAnalyzer(
                use_gpu=self.config.USE_GPU
            )
            logger.info("Motion analyzer initialized")
        
        if not self.temporal_bootstrapper:
            self.temporal_bootstrapper = TemporalBootstrapper(
                motion_analyzer=self.motion_analyzer,
                base_boost_factor=self.config.TEMPORAL_BOOST_FACTOR,
                confidence_weight=self.config.CONFIDENCE_WEIGHT,
                min_confidence_threshold=self.config.MIN_CONFIDENCE_THRESHOLD,
                min_window_seconds=self.config.MIN_WINDOW_SECONDS,
                max_window_seconds=self.config.MAX_WINDOW_SECONDS
            )
            logger.info("Temporal bootstrapper initialized")
    
    def process_video(self, 
                     video_path: str,
                     video_name: Optional[str] = None,
                     save_frames: bool = False,
                     upload_to_pinecone: bool = True,
                     use_object_detection: bool = False) -> Dict[str, Any]:
        """
        Process a video file end-to-end
        
        Args:
            video_path: Path to video file
            video_name: Name for the video (uses filename if None)
            save_frames: Whether to save extracted frames to disk
            upload_to_pinecone: Whether to upload embeddings to Pinecone
            use_object_detection: Whether to use object detection + captioning pipeline
            
        Returns:
            Processing statistics and results
        """
        start_time = time.time()
        
        # Validate video file
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Set video name
        if not video_name:
            video_name = Path(video_path).stem
        
        self.current_video = video_name
        logger.info(f"Processing video: {video_name} ({video_path})")
        
        # Store video path for motion analysis
        self.video_paths[video_name] = video_path
        
        # Initialize components
        self._initialize_components()
        
        try:
            # Step 1: Extract frames
            logger.info("Step 1/4: Extracting frames...")
            frames = self.frame_extractor.extract_frames(
                video_path=video_path,
                use_similarity_filter=True
            )
            
            if save_frames:
                output_dir = os.path.join(self.config.OUTPUT_DIR, video_name, "frames")
                self.frame_extractor.save_frames_to_disk(output_dir)
            
            # Step 2: Generate captions
            logger.info("Step 2/4: Generating captions...")
            
            if use_object_detection:
                # Use object-focused captioning pipeline
                logger.info("Using object detection + captioning pipeline")
                
                if not self.object_pipeline:
                    from object_caption_pipeline import ObjectCaptionPipeline
                    self.object_pipeline = ObjectCaptionPipeline(
                        use_gpu=self.config.USE_GPU,
                        min_object_size=30,
                        max_objects_per_frame=10,
                        include_scene_caption=True  # Fallback to scene caption if no objects
                    )
                
                # Process frames with object detection
                object_captions = self.object_pipeline.process_frames(
                    frames=frames,
                    show_progress=True
                )
                
                # Convert ObjectCaption to CaptionedFrame format
                captioned_frames = []
                for oc in object_captions:
                    cf = CaptionedFrame(
                        frame_data=oc.frame_data,
                        caption=oc.attribute_caption,
                        confidence=oc.confidence
                    )
                    captioned_frames.append(cf)
                
                logger.info(f"Object detection pipeline generated {len(captioned_frames)} captions")
                
            else:
                # Use standard BLIP captioning
                captioned_frames = self.caption_generator.generate_captions(
                    frames=frames,
                    filter_empty=True
                )
            
            # Filter duplicate captions
            captioned_frames = self.caption_generator.filter_duplicate_captions(
                captioned_frames=captioned_frames,
                time_window=self.config.DUPLICATE_TIME_WINDOW
            )
            
            # Step 3: Generate embeddings
            logger.info("Step 3/4: Generating embeddings...")
            embedded_frames = self.embedding_generator.generate_embeddings(
                captioned_frames=captioned_frames
            )
            
            # Step 3.5: Deduplicate embeddings before upload
            logger.info("Deduplicating embeddings...")
            captions_before_dedupe = len(embedded_frames)
            # Fallback if embedding generator lacks deduplication API
            if hasattr(self.embedding_generator, 'deduplicate_embeddings'):
                embedded_frames = self.embedding_generator.deduplicate_embeddings(
                    embedded_frames=embedded_frames,
                    similarity_threshold=0.95  # Remove very similar embeddings
                )
            else:
                embedded_frames = self._deduplicate_embeddings(
                    embedded_frames=embedded_frames,
                    similarity_threshold=0.95
                )
            logger.info(f"After deduplication: {len(embedded_frames)} unique embeddings")
            
            # Step 4: Upload to Pinecone
            actual_uploaded = 0
            if upload_to_pinecone:
                logger.info("Step 4/4: Uploading to Pinecone...")
                pinecone_data = self.embedding_generator.prepare_for_pinecone(
                    embedded_frames=embedded_frames,
                    video_name=video_name,
                    source_file_path=video_path
                )
                
                actual_uploaded = self.pinecone_manager.upload_embeddings(
                    data=pinecone_data,
                    batch_size=self.config.PINECONE_BATCH_SIZE
                )
                
                # Print verification
                if actual_uploaded > 0:
                    sample_ids = [pinecone_data[i][0] for i in range(min(3, len(pinecone_data)))]
                    logger.info(f"✅ Pinecone upsert confirmed: {actual_uploaded} vectors uploaded for {video_name}")
                    logger.info(f"   Sample IDs: {', '.join(sample_ids)}...")
                else:
                    logger.warning("No vectors were successfully uploaded to Pinecone")
            
            # Store processed frames
            self.processed_frames = embedded_frames
            
            # Calculate statistics
            processing_time = time.time() - start_time
            
            # Calculate frame reduction correctly
            total_video_frames = len(frames)  # Frames before similarity filtering
            frames_after_caption = len(captioned_frames)  # Frames that got captions
            frame_reduction_pct = ((total_video_frames - frames_after_caption) / total_video_frames * 100) if total_video_frames > 0 else 0
            
            stats = {
                "video_name": video_name,
                "video_path": video_path,
                "total_frames_extracted": total_video_frames,
                "frames_with_captions": frames_after_caption,
                "captions_before_dedupe": captions_before_dedupe,
                "embeddings_generated": len(embedded_frames),  # After dedupe
                "embeddings_uploaded": actual_uploaded if upload_to_pinecone else 0,
                "processing_time_seconds": processing_time,
                "frame_reduction_percent": frame_reduction_pct,
                "caption_stats": self.caption_generator.get_caption_statistics(captioned_frames),
                "embedding_stats": self.embedding_generator.get_embedding_statistics(embedded_frames)
            }
            
            self.processing_stats = stats
            
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")
            logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
            
            # Save processing report
            self._save_processing_report(stats, video_name)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
        
        finally:
            # Clear GPU cache
            if self.caption_generator:
                self.caption_generator.clear_gpu_cache()
            if self.embedding_generator:
                self.embedding_generator.clear_cache()
            if self.object_pipeline:
                self.object_pipeline.clear_cache()
    
    def _deduplicate_embeddings(self,
                               embedded_frames: List[EmbeddedFrame],
                               similarity_threshold: float = 0.95) -> List[EmbeddedFrame]:
        """Deduplicate embeddings locally if generator lacks the method."""
        if not embedded_frames:
            return []
        if len(embedded_frames) == 1:
            return embedded_frames
        # Stack embeddings
        embeddings = np.array([ef.embedding for ef in embedded_frames])
        # If embeddings are normalized, cosine similarity is dot product
        normalized = True
        # Heuristic: check mean norm ~1.0
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms.mean(), 1.0, atol=1e-2):
            normalized = False
        keep = np.ones(len(embeddings), dtype=bool)
        for i in range(len(embeddings)):
            if not keep[i]:
                continue
            vec_i = embeddings[i]
            for j in range(i + 1, len(embeddings)):
                if not keep[j]:
                    continue
                vec_j = embeddings[j]
                if normalized:
                    sim = float(np.dot(vec_i, vec_j))
                else:
                    denom = (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)) or 1e-8
                    sim = float(np.dot(vec_i, vec_j) / denom)
                if sim >= similarity_threshold:
                    keep[j] = False
        return [ef for ef, k in zip(embedded_frames, keep) if k]

    def search(self,
              query: str,
              top_k: int = None,
              similarity_threshold: float = None,
              video_filter: Optional[str] = None,
              time_window: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """
        Search for video frames using natural language query
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            video_filter: Filter by video name
            time_window: Filter by time range (start, end) in seconds
            
        Returns:
            List of search results with timestamps and metadata
        """
        # Use config defaults if not specified
        top_k = top_k or self.config.QUERY_TOP_K
        similarity_threshold = similarity_threshold or self.config.QUERY_SIMILARITY_THRESHOLD
        
        # Initialize components if needed
        if not self.embedding_generator:
            self.embedding_generator = TextEmbeddingGenerator(
                model_name=self.config.EMBEDDING_MODEL,
                batch_size=self.config.EMBEDDING_BATCH_SIZE,
                use_gpu=self.config.USE_GPU,
                normalize=True
            )
        
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        logger.info(f"Searching for: '{query}'")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_query(query)
            
            # Search in Pinecone
            search_results = self.pinecone_manager.semantic_search(
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                video_filter=video_filter,
                time_window=time_window
            )
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "timestamp": result.timestamp,
                    "caption": result.caption,
                    "similarity_score": result.score,
                    "frame_id": result.frame_id,
                    "video_name": result.video_name,
                    "time_formatted": self._format_timestamp(result.timestamp)
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} results for query: '{query}'")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def search_with_bootstrapping(self,
                                 primary_query: str,
                                 related_queries: Optional[List[str]] = None,
                                 top_k: int = None,
                                 similarity_threshold: float = None,
                                 video_filter: Optional[str] = None,
                                 auto_extract_related: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search with temporal bootstrapping
        
        Part 1: Temporal Bootstrapping - Finds related objects at similar timestamps
        Part 2: Adaptive Window - Adjusts search window based on motion
        Part 3: Confidence-Aware - Weights boosts by detection confidence
        
        Args:
            primary_query: Primary search query (e.g., "red shirt")
            related_queries: List of related queries (e.g., ["blue bag", "person"])
                           If None and auto_extract_related=True, extracts from primary_query
            top_k: Number of results per query
            similarity_threshold: Minimum similarity score
            video_filter: Filter by video name
            auto_extract_related: Automatically extract related objects from query
            
        Returns:
            Dictionary mapping queries to boosted results
        """
        if not self.config.ENABLE_TEMPORAL_BOOTSTRAPPING:
            logger.warning("Temporal bootstrapping is disabled. Using standard search.")
            results = self.search(primary_query, top_k, similarity_threshold, video_filter)
            return {primary_query: results}
        
        # Use config defaults
        top_k = top_k or self.config.QUERY_TOP_K
        similarity_threshold = similarity_threshold or self.config.QUERY_SIMILARITY_THRESHOLD
        
        # Initialize components
        self._initialize_components()
        self._initialize_temporal_bootstrapping()
        
        logger.info(f"Temporal bootstrapping search for: '{primary_query}'")
        
        try:
            # Step 1: Search for primary query
            primary_embedding = self.embedding_generator.encode_query(primary_query)
            primary_results = self.pinecone_manager.semantic_search(
                query_embedding=primary_embedding,
                top_k=top_k * 2,  # Get more for better bootstrapping
                similarity_threshold=similarity_threshold,
                video_filter=video_filter
            )
            
            if not primary_results:
                logger.warning(f"No results found for primary query: '{primary_query}'")
                return {primary_query: []}
            
            logger.info(f"Found {len(primary_results)} primary results")
            
            # Step 2: Extract or use provided related queries
            if related_queries is None and auto_extract_related:
                related_queries = self._extract_related_queries(primary_query)
            
            if not related_queries:
                logger.info("No related queries found. Returning primary results only.")
                formatted = self._format_search_results(primary_results[:top_k])
                return {primary_query: formatted}
            
            # Step 3: Generate embeddings for related queries
            related_query_embeddings = []
            for related_query in related_queries:
                embedding = self.embedding_generator.encode_query(related_query)
                related_query_embeddings.append((related_query, embedding))
            
            # Step 4: Get video path and frame timestamps for motion analysis
            video_path = None
            frame_timestamps = None
            
            if video_filter and video_filter in self.video_paths:
                video_path = self.video_paths[video_filter]
                # Extract timestamps from primary results
                frame_timestamps = [r.timestamp for r in primary_results]
            
            # Step 5: Perform temporal bootstrapping
            boosted_results_dict = self.temporal_bootstrapper.bootstrap_search(
                primary_results=primary_results,
                related_queries=related_query_embeddings,
                pinecone_manager=self.pinecone_manager,
                video_name=video_filter,
                video_path=video_path,
                frame_timestamps=frame_timestamps,
                top_k_per_query=top_k
            )
            
            # Step 6: Format results
            formatted_results = {}
            
            # Format primary results
            formatted_results[primary_query] = self._format_search_results(
                primary_results[:top_k]
            )
            
            # Format boosted results
            for query_text, boosted_results in boosted_results_dict.items():
                formatted_results[query_text] = self._format_boosted_results(boosted_results)
            
            logger.info(f"Temporal bootstrapping completed. "
                       f"Primary query: {len(formatted_results[primary_query])} results, "
                       f"Related queries: {len(boosted_results_dict)} queries processed")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Temporal bootstrapping search failed: {e}")
            raise
    
    def _extract_related_queries(self, query: str) -> List[str]:
        """
        Extract related object queries from primary query
        Simple heuristic: looks for object patterns like "red shirt", "blue bag"
        
        Args:
            query: Primary search query
            
        Returns:
            List of related object queries
        """
        # Common object categories that often appear together
        object_categories = {
            "person": ["bag", "backpack", "clothing", "shirt", "pants", "shoes"],
            "bag": ["person", "backpack", "handbag"],
            "shirt": ["person", "pants", "jacket"],
            "bottle": ["person", "bag", "table"],
            "phone": ["person", "hand", "bag"],
            "laptop": ["person", "bag", "table"],
            "bicycle": ["person", "helmet", "bag"],
            "car": ["person", "window", "road"]
        }
        
        query_lower = query.lower()
        related = []
        
        # Extract color + object patterns (e.g., "red shirt" -> extract "shirt", "person")
        # This is a simple heuristic - can be enhanced with NLP
        for category, related_objects in object_categories.items():
            if category in query_lower:
                # Add related objects
                for obj in related_objects:
                    if obj not in query_lower:
                        related.append(obj)
        
        # If no patterns found, try common related objects
        if not related:
            # Common patterns: extract color + noun
            import re
            # Look for patterns like "red X", "blue Y"
            color_noun_pattern = r'\b(red|blue|green|yellow|black|white|gray|brown|orange|purple|pink)\s+(\w+)'
            matches = re.findall(color_noun_pattern, query_lower)
            
            if matches:
                # Extract the noun and suggest related objects
                for color, noun in matches:
                    if noun not in ["shirt", "bag", "bottle"]:
                        related.append(noun)
        
        # Remove duplicates and limit
        related = list(set(related))[:5]  # Limit to 5 related queries
        
        logger.info(f"Extracted {len(related)} related queries from '{query}': {related}")
        
        return related
    
    def _format_search_results(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Format search results for output"""
        formatted_results = []
        for result in results:
            formatted_result = {
                "timestamp": result.timestamp,
                "caption": result.caption,
                "similarity_score": result.score,
                "frame_id": result.frame_id,
                "video_name": result.video_name,
                "time_formatted": self._format_timestamp(result.timestamp)
            }
            formatted_results.append(formatted_result)
        return formatted_results
    
    def _format_boosted_results(self, boosted_results: List[BoostedResult]) -> List[Dict[str, Any]]:
        """Format boosted results for output"""
        formatted_results = []
        for boosted in boosted_results:
            result = boosted.result
            formatted_result = {
                "timestamp": result.timestamp,
                "caption": result.caption,
                "similarity_score": boosted.boosted_score,
                "original_score": boosted.original_score,
                "boost_amount": boosted.boost_amount,
                "boost_reason": boosted.boost_reason,
                "frame_id": result.frame_id,
                "video_name": result.video_name,
                "time_formatted": self._format_timestamp(result.timestamp)
            }
            formatted_results.append(formatted_result)
        return formatted_results
    
    def batch_search(self,
                    queries: List[str],
                    top_k: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform batch search for multiple queries
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            Dictionary mapping queries to results
        """
        results = {}
        
        for query in queries:
            try:
                results[query] = self.search(query, top_k=top_k)
            except Exception as e:
                logger.error(f"Failed to search for '{query}': {e}")
                results[query] = []
        
        return results
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
        else:
            return f"{minutes:02d}:{secs:05.2f}"
    
    def _save_processing_report(self, stats: Dict, video_name: str):
        """Save processing report to file"""
        report_dir = os.path.join(self.config.OUTPUT_DIR, video_name)
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, "processing_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing report saved to: {report_path}")
    
    def get_index_stats(self) -> Dict:
        """Get Pinecone index statistics"""
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        return self.pinecone_manager.get_index_stats()
    
    def clear_index(self) -> bool:
        """Clear all vectors from Pinecone index"""
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        return self.pinecone_manager.clear_index()
    
    def cleanup(self):
        """Clean up resources"""
        if self.caption_generator:
            self.caption_generator.unload_model()
        if self.embedding_generator:
            self.embedding_generator.unload_model()
        if self.object_pipeline:
            self.object_pipeline.unload_models()
        
        logger.info("Resources cleaned up")


def demo_usage():
    """Demonstrate usage of the Video Search Engine"""
    
    # Initialize engine
    engine = VideoSearchEngine()
    
    # Example: Process a video
    # stats = engine.process_video(
    #     video_path="sample_video.mp4",
    #     video_name="sample_demo",
    #     save_frames=False,
    #     upload_to_pinecone=True
    # )
    
    # Example: Search for content
    # results = engine.search(
    #     query="person walking with a black bag",
    #     top_k=5
    # )
    # 
    # for result in results:
    #     print(f"Time: {result['time_formatted']} - Score: {result['similarity_score']:.3f}")
    #     print(f"  Caption: {result['caption']}")
    #     print(f"  Video: {result['video_name']}")
    #     print()
    
    # Example: Batch search
    # queries = [
    #     "black bag",
    #     "yellow bottle",
    #     "person walking",
    #     "car driving"
    # ]
    # 
    # batch_results = engine.batch_search(queries, top_k=3)
    # 
    # for query, results in batch_results.items():
    #     print(f"\nQuery: '{query}' - Found {len(results)} results")
    #     for result in results[:2]:  # Show top 2
    #         print(f"  {result['time_formatted']} (score: {result['similarity_score']:.3f})")
    
    # Get index stats
    stats = engine.get_index_stats()
    print(f"Index statistics: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    demo_usage()