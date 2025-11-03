"""
Configuration file for Video Frame Search System
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the video search system"""
    
    # API Keys (load from environment variables for security)
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    PINECONE_HOST = os.getenv('PINECONE_HOST', 'https://capstone-b5a0x4x.svc.aped-4627-b74a.pinecone.io')
    
    # Pinecone Index Configuration
    PINECONE_INDEX_NAME = 'capstone'
    PINECONE_DIMENSION = 1024  # For llama-text-embed-v2
    PINECONE_METRIC = 'cosine'
    PINECONE_CLOUD = 'aws'
    PINECONE_REGION = 'us-east-1'
    
    # Model Configuration
    # InstructBLIP is better for object-focused captions (colors, attributes)
    BLIP_MODEL = 'Salesforce/instructblip-flan-t5-xl'  # Better object attribute descriptions
    # Using a model compatible with 1024 dimensions
    EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'  # 1024 dimensions
    # Alternative: 'thenlper/gte-large' (1024 dimensions)
    
    # Frame Extraction Configuration
    FRAME_SIMILARITY_THRESHOLD = 0.90  # Higher threshold to capture more frames (only skip very similar frames)
    FRAME_EXTRACTION_INTERVAL = 2.0  # Extract frame every N seconds (if not using similarity)
    MAX_FRAMES_PER_VIDEO = 1000  # Maximum frames to extract per video
    FRAME_RESIZE_WIDTH = 640  # Resize frames for memory efficiency (None for original size)
    MIN_FRAMES_PER_VIDEO = 10  # Minimum frames to extract regardless of similarity
    
    # Enhanced Caption Configuration
    GENERATE_MULTIPLE_CAPTIONS = True  # Generate multiple object-focused captions per frame
    CAPTIONS_PER_FRAME = 3  # Number of different captions to generate per frame
    USE_OBJECT_FOCUSED_PROMPTS = True  # Use object-focused prompts for more detailed descriptions
    
    # Processing Configuration
    BLIP_BATCH_SIZE = 8  # Batch size for BLIP caption generation
    EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation
    PINECONE_BATCH_SIZE = 100  # Batch size for Pinecone uploads
    
    # Query Configuration
    QUERY_TOP_K = 10  # Number of results to return
    QUERY_SIMILARITY_THRESHOLD = 0.6  # Minimum similarity score for results
    DUPLICATE_TIME_WINDOW = 5.0  # Seconds within which to consider frames as duplicates (increased for multi-captions)
    
    # Temporal Bootstrapping Configuration
    ENABLE_TEMPORAL_BOOTSTRAPPING = True  # Enable temporal bootstrapping features
    TEMPORAL_BOOST_FACTOR = 0.3  # Base boost factor (0.0 to 1.0)
    CONFIDENCE_WEIGHT = 1.0  # How much to weight detection confidence (0.0 to 2.0)
    MIN_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to apply temporal boost
    MIN_WINDOW_SECONDS = 1.0  # Minimum temporal window for slow scenes
    MAX_WINDOW_SECONDS = 5.0  # Maximum temporal window for fast scenes
    BASE_WINDOW_SECONDS = 2.0  # Base temporal window size
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'video_search.log'
    
    # Performance Configuration
    USE_GPU = True  # Use GPU if available
    NUM_WORKERS = 4  # Number of workers for data loading
    
    # File paths
    TEMP_DIR = './temp'
    OUTPUT_DIR = './output'
    
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set. Please set it in .env file or environment variables")
        
        if cls.PINECONE_DIMENSION not in [384, 768, 1024, 1536]:
            print(f"Warning: Unusual embedding dimension {cls.PINECONE_DIMENSION}. Common values are 384, 768, 1024, or 1536")
        
        if cls.FRAME_SIMILARITY_THRESHOLD < 0 or cls.FRAME_SIMILARITY_THRESHOLD > 1:
            raise ValueError("FRAME_SIMILARITY_THRESHOLD must be between 0 and 1")
        
        # Create directories if they don't exist
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        
        return True