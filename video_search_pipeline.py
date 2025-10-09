"""
Lightweight version of the video search pipeline using text encoders and simple CV.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
import sys

from config import (
    VIDEOS_DIR, KEYFRAMES_DIR, FEATURES_DIR, INDEX_DIR,
    LOG_LEVEL, LOG_FILE
)

# Import modules
from frame_extractor import extract_frames_from_videos
from feature_extractor import extract_features_from_frames, LightweightFeatureExtractor
from retrieval_system import build_retrieval_system, load_retrieval_system, VideoRetrievalSystem
from enhancements import EnhancedRetrieval

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class VideoSearchPipeline:
    """Text-driven video search pipeline using computer vision and sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.feature_extractor = None
        self.retrieval_system = None
        self.enhanced_retrieval = None
        
    def setup_pipeline(self):
        """Initialize the pipeline components."""
        logger.info("Setting up video search pipeline...")
        
        # Initialize feature extractor
        self.feature_extractor = LightweightFeatureExtractor(model_name=self.model_name)
        self.feature_extractor.load_models()
        
        # Initialize enhanced retrieval system
        self.enhanced_retrieval = EnhancedRetrieval(self.feature_extractor)
        
        logger.info("Pipeline setup completed")
    
    def build_index(self, videos_dir: Path, force_rebuild: bool = False,
                    segment_captions: bool = False,
                    object_tags: bool = False) -> bool:
        """
        Build the complete search index from videos.
        
        Args:
            videos_dir: Directory containing video files
            force_rebuild: Whether to rebuild even if index exists
            
        Returns:
            True if index was built successfully
        """
        logger.info("Starting index building process...")
        
        # Check if index already exists (validate required files, not just directory)
        index_files = [
            INDEX_DIR / "faiss_index.bin",
            INDEX_DIR / "metadata.json",
            INDEX_DIR / "config.json",
            INDEX_DIR / "caption_index.json"
        ]
        has_complete_index = all(p.exists() for p in index_files)
        if has_complete_index and not force_rebuild:
            logger.info("Existing index detected. Skipping rebuild. Use --force-rebuild to rebuild.")
            return True
        
        try:
            # Step 1: Extract frames from videos
            logger.info("Step 1: Extracting frames from videos...")
            frame_metadata = extract_frames_from_videos(
                videos_dir, KEYFRAMES_DIR, method="shot_based"
            )
            
            if not frame_metadata:
                logger.error("No frames extracted. Check video directory.")
                return False
            
            # Step 2: Extract semantic features
            logger.info("Step 2: Extracting semantic features...")
            features = extract_features_from_frames(
                frame_metadata, FEATURES_DIR, model_name=self.model_name,
                segment_captions=segment_captions, object_tags=object_tags
            )
            
            if not features:
                logger.error("No features extracted. Check frame extraction.")
                return False
            
            # Step 3: Build retrieval index
            logger.info("Step 3: Building retrieval index...")
            features_file = FEATURES_DIR / "frame_features.json"
            self.retrieval_system = build_retrieval_system(features_file, INDEX_DIR)
            
            logger.info("Index building completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            return False
    
    def load_index(self) -> bool:
        """
        Load existing search index.
        
        Returns:
            True if index was loaded successfully
        """
        try:
            logger.info("Loading search index...")
            self.retrieval_system = load_retrieval_system(INDEX_DIR)
            logger.info("Index loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def search(self, query: str, use_enhancements: bool = True, 
               max_results: int = 10) -> List[Dict]:
        """
        Search for frames using text query.
        
        Args:
            query: Text search query
            use_enhancements: Whether to use enhanced retrieval
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        if self.retrieval_system is None:
            logger.error("Retrieval system not loaded. Call load_index() first.")
            return []
        
        logger.info(f"Searching for: '{query}'")
        
        try:
            if use_enhancements:
                # Use enhanced retrieval with all improvements
                results = self.enhanced_retrieval.enhanced_search(
                    self.retrieval_system, query,
                    use_expansion=True,
                    use_clustering=True,
                    use_fusion=True,
                    use_grounding=False
                )
            else:
                # Use basic retrieval
                results = self.retrieval_system.search_by_text(query, max_results)
            
            # Remove temporal duplicates
            results = self.retrieval_system.remove_temporal_duplicates(results)
            
            logger.info(f"Found {len(results)} results")
            return results[:max_results]
            
        except Exception:
            # Log full traceback for easier debugging
            logger.exception("Search failed")
            return []
    
    def get_result_thumbnails(self, results: List[Dict], 
                            thumbnail_size: tuple = (224, 224)) -> List:
        """
        Get thumbnail images for search results.
        
        Args:
            results: Search results
            thumbnail_size: Size of thumbnails
            
        Returns:
            List of thumbnail images
        """
        thumbnails = []
        for result in results:
            thumbnail = self.retrieval_system.get_frame_thumbnail(result, thumbnail_size)
            thumbnails.append(thumbnail)
        return thumbnails
    
    def save_search_results(self, results: List[Dict], query: str, 
                          output_dir: Path):
        """
        Save search results to disk.
        
        Args:
            results: Search results
            query: Original query
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_file = output_dir / f"search_results_{query.replace(' ', '_')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save thumbnails
        thumbnails = self.get_result_thumbnails(results)
        for i, (result, thumbnail) in enumerate(zip(results, thumbnails)):
            thumbnail_path = output_dir / f"thumbnail_{i:03d}.jpg"
            from PIL import Image
            Image.fromarray(thumbnail).save(thumbnail_path)
        
        logger.info(f"Search results saved to {output_dir}")
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        stats = {}
        
        if INDEX_DIR.exists():
            config_file = INDEX_DIR / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    stats["index_size"] = config.get("num_vectors", 0)
                    stats["embedding_dim"] = config.get("embedding_dim", 0)
        
        if KEYFRAMES_DIR.exists():
            frame_files = list(KEYFRAMES_DIR.glob("*.jpg"))
            stats["total_frames"] = len(frame_files)
        
        if VIDEOS_DIR.exists():
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
            video_files = [f for f in VIDEOS_DIR.iterdir() 
                          if f.suffix.lower() in video_extensions]
            stats["total_videos"] = len(video_files)
        
        # Add pipeline stats
        stats["model_type"] = "computer_vision"
        stats["text_encoder"] = self.model_name
        
        return stats


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Text-driven video search pipeline")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build index command
    build_parser = subparsers.add_parser('build', help='Build search index from videos')
    build_parser.add_argument('--videos-dir', type=Path, default=VIDEOS_DIR,
                             help='Directory containing video files')
    build_parser.add_argument('--force-rebuild', action='store_true',
                             help='Force rebuild even if index exists')
    build_parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                             help='Sentence transformer model name')
    build_parser.add_argument('--segment-captions', action='store_true',
                             help='Aggregate nearby keyframes into short event segments')
    build_parser.add_argument('--object-tags', action='store_true',
                             help='Run object detection to tag frames and store tags')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for frames')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--max-results', type=int, default=10,
                              help='Maximum number of results')
    search_parser.add_argument('--no-enhancements', action='store_true',
                              help='Disable enhanced retrieval')
    search_parser.add_argument('--save-results', type=Path,
                              help='Directory to save search results')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', 
                                             help='Interactive search mode')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show pipeline statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize pipeline
    model_name = getattr(args, 'model', 'all-MiniLM-L6-v2')
    pipeline = VideoSearchPipeline(model_name=model_name)
    pipeline.setup_pipeline()
    
    if args.command == 'build':
        success = pipeline.build_index(
            args.videos_dir,
            args.force_rebuild,
            segment_captions=getattr(args, 'segment_captions', False),
            object_tags=getattr(args, 'object_tags', False)
        )
        if success:
            print("Index built successfully!")
        else:
            print("Index building failed!")
            sys.exit(1)
    
    elif args.command == 'search':
        # Load index
        if not pipeline.load_index():
            print("Failed to load index. Build index first.")
            sys.exit(1)
        
        # Perform search
        results = pipeline.search(
            args.query, 
            use_enhancements=not args.no_enhancements,
            max_results=args.max_results
        )
        
        # Display results
        print(f"\nSearch results for '{args.query}':")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Caption: {result['metadata']['caption']}")
            print(f"   Path: {result['metadata']['image_path']}")
            if 'grounding_boost' in result:
                print(f"   Grounding boost: {result['grounding_boost']:.3f}")
            print()
        
        # Save results if requested
        if args.save_results:
            pipeline.save_search_results(results, args.query, args.save_results)
    
    elif args.command == 'interactive':
        # Load index
        if not pipeline.load_index():
            print("Failed to load index. Build index first.")
            sys.exit(1)
        
        print("Interactive search mode. Type 'quit' to exit.")
        print("Available commands:")
        print("  search <query> - Search for frames")
        print("  stats - Show pipeline statistics")
        print("  quit - Exit")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    stats = pipeline.get_statistics()
                    print("\nPipeline Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        results = pipeline.search(query, use_enhancements=True)
                        
                        print(f"\nSearch results for '{query}':")
                        for i, result in enumerate(results[:5], 1):  # Show top 5
                            print(f"{i}. Score: {result['score']:.3f}")
                            print(f"   Caption: {result['metadata']['caption']}")
                            print(f"   Path: {result['metadata']['image_path']}")
                            print()
                else:
                    print("Unknown command. Available: search, stats, quit")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.command == 'stats':
        stats = pipeline.get_statistics()
        print("Pipeline Statistics:")
        print("=" * 30)
        for key, value in stats.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
