"""
Lightweight feature extraction using text encoders and simpler vision models.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import cv2
import warnings

from config import DEVICE, BLIP_MODEL_NAME
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class LightweightFeatureExtractor:
    """Lightweight feature extractor using sentence transformers and simple vision features."""
    
    def __init__(self, device: str = DEVICE, model_name: str = "all-MiniLM-L6-v2", use_blip_captions: bool = True):
        self.device = device
        self.model_name = model_name
        self.text_encoder = None
        self.image_features_extractor = None
        self.use_blip_captions = use_blip_captions
        self.blip_processor: Optional[BlipProcessor] = None
        self.blip_model: Optional[BlipForConditionalGeneration] = None
        
    def load_models(self):
        """Load lightweight models."""
        logger.info(f"Loading lightweight text encoder: {self.model_name}")
        
        # Load sentence transformer for text encoding
        self.text_encoder = SentenceTransformer(self.model_name)
        self.text_encoder.to(self.device)
        
        # Optionally load BLIP for captioning
        if self.use_blip_captions and (self.blip_model is None or self.blip_processor is None):
            try:
                logger.info(f"Loading BLIP model: {BLIP_MODEL_NAME}")
                self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
                self.blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
                self.blip_model.to(self.device)
                self.blip_model.eval()
                logger.info("BLIP loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load BLIP ({BLIP_MODEL_NAME}), falling back to simple captions: {e}")
                self.use_blip_captions = False
        
        logger.info("Lightweight models loaded successfully")
    
    def generate_caption_simple(self, image_path: Path) -> str:
        """
        Generate simple caption using basic computer vision techniques.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Simple descriptive caption
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return "unknown scene"
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Basic image analysis
            caption_parts = []
            
            # Detect dominant colors
            dominant_colors = self._get_dominant_colors(image_rgb)
            if dominant_colors:
                caption_parts.append(f"image with {dominant_colors[0]} colors")
            
            # Detect edges (simple activity detection)
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            
            if edge_density > 0.1:
                caption_parts.append("busy scene")
            else:
                caption_parts.append("calm scene")
            
            # Detect faces (if any)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                caption_parts.append(f"scene with {len(faces)} person")
            
            # Detect motion blur (simple)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 100:
                caption_parts.append("motion blur")
            
            return " ".join(caption_parts) if caption_parts else "general scene"
            
        except Exception as e:
            logger.error(f"Failed to generate caption for {image_path}: {e}")
            return "unknown scene"

    def generate_caption_blip(self, image_path: Path, max_length: int = 50) -> str:
        """
        Generate caption using BLIP if available.
        """
        if not self.use_blip_captions:
            return self.generate_caption_simple(image_path)
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.blip_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True
                )
            # BLIP processor does not have decode; use tokenizer decode via model
            caption = self.blip_processor.tokenizer.decode(out[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as e:
            logger.warning(f"BLIP caption failed for {image_path}, falling back: {e}")
            return self.generate_caption_simple(image_path)

    def generate_caption(self, image_path: Path) -> str:
        """Unified caption generation entry point."""
        if self.use_blip_captions and self.blip_model is not None:
            return self.generate_caption_blip(image_path)
        return self.generate_caption_simple(image_path)
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[str]:
        """Get dominant colors in the image."""
        try:
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
            
            # Simple k-means clustering for dominant colors
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            
            # Map RGB to color names
            color_names = []
            for color in colors:
                color_name = self._rgb_to_color_name(color)
                color_names.append(color_name)
            
            return color_names
            
        except Exception:
            return []
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Convert RGB values to approximate color names."""
        r, g, b = rgb
        
        # Simple color mapping
        if r > 200 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g > 200 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 200:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r < 100 and g < 100 and b < 100:
            return "black"
        elif r > 200 and g > 200 and b > 200:
            return "white"
        elif r > 150 and g < 150 and b > 150:
            return "purple"
        elif r > 150 and g > 100 and b < 100:
            return "orange"
        else:
            return "mixed"
    
    def extract_visual_features(self, image_path: Path) -> np.ndarray:
        """
        Extract simple visual features from image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Visual feature vector
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return np.zeros(512)  # Default feature size
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract multiple types of features
            features = []
            
            # 1. Color histogram features
            color_features = self._extract_color_histogram(image_rgb)
            features.extend(color_features)
            
            # 2. Texture features (LBP-like)
            texture_features = self._extract_texture_features(image_rgb)
            features.extend(texture_features)
            
            # 3. Edge features
            edge_features = self._extract_edge_features(image_rgb)
            features.extend(edge_features)
            
            # 4. Shape features
            shape_features = self._extract_shape_features(image_rgb)
            features.extend(shape_features)
            
            # Pad or truncate to fixed size
            target_size = 512
            if len(features) < target_size:
                features.extend([0] * (target_size - len(features)))
            elif len(features) > target_size:
                features = features[:target_size]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Failed to extract visual features for {image_path}: {e}")
            return np.zeros(512)
    
    def _extract_color_histogram(self, image: np.ndarray) -> List[float]:
        """Extract color histogram features."""
        # RGB histograms
        r_hist = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
        g_hist = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
        b_hist = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()
        
        # HSV histogram
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        
        # Normalize histograms
        features = []
        for hist in [r_hist, g_hist, b_hist, h_hist]:
            hist_norm = hist / (np.sum(hist) + 1e-8)
            features.extend(hist_norm.tolist())
        
        return features
    
    def _extract_texture_features(self, image: np.ndarray) -> List[float]:
        """Extract texture features using local binary patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Simple texture features
        features = []
        
        # 1. Local Binary Pattern approximation
        lbp = self._simple_lbp(gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [16], [0, 256]).flatten()
        features.extend((lbp_hist / (np.sum(lbp_hist) + 1e-8)).tolist())
        
        # 2. Gabor filter responses
        gabor_responses = self._gabor_filters(gray)
        features.extend(gabor_responses)
        
        return features
    
    def _simple_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Simple Local Binary Pattern implementation."""
        h, w = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                binary = 0
                
                # 8-neighbor comparison
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        binary |= (1 << k)
                
                lbp[i, j] = binary
        
        return lbp
    
    def _gabor_filters(self, gray: np.ndarray) -> List[float]:
        """Apply Gabor filters for texture analysis."""
        features = []
        
        # Simple Gabor-like filters
        kernel_sizes = [5, 9, 13]
        orientations = [0, 45, 90, 135]
        
        for size in kernel_sizes:
            for angle in orientations:
                # Create simple edge detection kernel
                kernel = np.zeros((size, size))
                center = size // 2
                
                # Simple oriented edge detector
                if angle == 0:  # Horizontal
                    kernel[center, :] = 1
                    kernel[center-1, :] = -1
                elif angle == 90:  # Vertical
                    kernel[:, center] = 1
                    kernel[:, center-1] = -1
                else:  # Diagonal
                    for i in range(size):
                        for j in range(size):
                            if abs(i - center) == abs(j - center):
                                kernel[i, j] = 1 if (i + j) % 2 == 0 else -1
                
                # Apply filter
                filtered = cv2.filter2D(gray, -1, kernel)
                features.append(np.mean(np.abs(filtered)))
                features.append(np.std(filtered))
        
        return features
    
    def _extract_edge_features(self, image: np.ndarray) -> List[float]:
        """Extract edge-based features."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        features.append(np.mean(sobel_magnitude))
        features.append(np.std(sobel_magnitude))
        
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        return features
    
    def _extract_shape_features(self, image: np.ndarray) -> List[float]:
        """Extract shape-based features."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # Contour-based features
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Shape descriptors
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                features.append(circularity)
            else:
                features.append(0)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            if h > 0:
                aspect_ratio = w / h
                features.append(aspect_ratio)
            else:
                features.append(1.0)
        else:
            features.extend([0, 1.0])
        
        return features
    
    def extract_text_embedding(self, text: str) -> np.ndarray:
        """
        Extract text embedding using sentence transformer.
        
        Args:
            text: Input text string
            
        Returns:
            Text embedding vector
        """
        if self.text_encoder is None:
            self.load_models()
        
        try:
            embedding = self.text_encoder.encode([text])
            return embedding[0]
        except Exception as e:
            logger.error(f"Failed to extract text embedding for '{text}': {e}")
            return np.zeros(384)  # Default sentence transformer size
    
    def extract_multimodal_features(self, image_path: Path) -> Dict:
        """
        Extract both caption and embedding for an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary containing caption and embedding
        """
        caption = self.generate_caption(image_path)
        visual_features = self.extract_visual_features(image_path)
        caption_embedding = self.extract_text_embedding(caption)
        
        # Combine visual and text features
        combined_embedding = np.concatenate([visual_features, caption_embedding])
        
        return {
            "image_path": str(image_path),
            "caption": caption,
            "embedding": combined_embedding.tolist(),
            "visual_features": visual_features.tolist(),
            "caption_embedding": caption_embedding.tolist(),
            "embedding_dim": len(combined_embedding)
        }
    
    def batch_extract_features(self, image_paths: List[Path], 
                              batch_size: int = 16) -> List[Dict]:
        """
        Extract features from multiple images in batches.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            List of feature dictionaries
        """
        if self.text_encoder is None:
            self.load_models()
        
        features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), 
                     desc="Extracting lightweight features"):
            batch_paths = image_paths[i:i + batch_size]
            
            for path in batch_paths:
                try:
                    feature_dict = self.extract_multimodal_features(path)
                    features.append(feature_dict)
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    continue
        
        return features


def extract_features_from_frames(frame_metadata: List[Dict], 
                               output_dir: Path,
                               model_name: str = "all-MiniLM-L6-v2",
                               segment_captions: bool = False,
                               object_tags: bool = False) -> List[Dict]:
    """
    Extract features from all frames and save to disk.
    
    Args:
        frame_metadata: List of frame metadata dictionaries
        output_dir: Directory to save feature files
        model_name: Sentence transformer model name
        
    Returns:
        List of extracted feature dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = LightweightFeatureExtractor(model_name=model_name)
    extractor.load_models()
    
    image_paths = [Path(frame["frame_path"]) for frame in frame_metadata]
    
    logger.info(f"Extracting features from {len(image_paths)} frames...")
    
    features = extractor.batch_extract_features(image_paths)

    # Optional: add simple object tags derived from captions
    if object_tags:
        keyword_tags = [
            "person", "people", "man", "woman", "bag", "backpack", "briefcase",
            "car", "vehicle", "bicycle", "motorcycle", "dog", "cat", "chair",
            "table", "laptop", "phone", "book", "tree", "bench"
        ]
        for f in features:
            caption_words = set(f.get("caption", "").lower().split())
            tags = sorted({kw for kw in keyword_tags if kw in caption_words})
            f["tags"] = tags

    # Optional: create event-level segment features by grouping nearby frames
    if segment_captions:
        # Build index from original frame metadata: path -> (video_path, timestamp)
        path_to_meta = {
            str(m.get("frame_path")): {
                "video_path": m.get("video_path", ""),
                "timestamp": float(m.get("timestamp", 0.0))
            }
            for m in frame_metadata
        }
        # Group by video, then sliding window (e.g., 5s)
        window_sec = 5.0
        segments: List[Dict] = []
        # Pair features with timing
        enriched = []
        for f in features:
            meta = path_to_meta.get(f.get("image_path"), {"video_path": "", "timestamp": 0.0})
            enriched.append({"f": f, **meta})
        # Sort by video, timestamp
        enriched.sort(key=lambda x: (x["video_path"], x["timestamp"]))
        # Create segments
        seg_buffer = []
        cur_video = None
        seg_start = 0.0
        for item in enriched:
            vid = item["video_path"]
            ts = item["timestamp"]
            if not seg_buffer:
                cur_video = vid
                seg_start = ts
                seg_buffer.append(item)
                continue
            # If same video and within window, accumulate
            if vid == cur_video and (ts - seg_start) <= window_sec:
                seg_buffer.append(item)
            else:
                # Flush segment
                segments.append(seg_buffer)
                # Start new
                cur_video = vid
                seg_start = ts
                seg_buffer = [item]
        if seg_buffer:
            segments.append(seg_buffer)

        # Build segment-level features
        segment_features: List[Dict] = []
        for seg in segments:
            if not seg:
                continue
            # Aggregate captions and embeddings
            captions = [s["f"].get("caption", "") for s in seg]
            tags_lists = [s["f"].get("tags", []) for s in seg]
            emb_list = [np.array(s["f"].get("caption_embedding", [])) for s in seg]
            # Fallback if empty
            emb_list = [e for e in emb_list if e.size > 0]
            seg_caption = "; ".join(captions)[:256]
            if emb_list:
                seg_text_emb = np.mean(emb_list, axis=0)
            else:
                seg_text_emb = extractor.extract_text_embedding(seg_caption)
            # Create combined embedding using only text to keep it compact
            combined_emb = seg_text_emb
            seg_tags = sorted({t for tags in tags_lists for t in tags})
            rep_image = seg[0]["f"].get("image_path")
            seg_start_ts = seg[0]["timestamp"]
            seg_end_ts = seg[-1]["timestamp"]
            segment_features.append({
                "image_path": rep_image,
                "caption": seg_caption,
                "embedding": combined_emb.tolist(),
                "embedding_dim": len(combined_emb),
                "segment": {"start": float(seg_start_ts), "end": float(seg_end_ts)},
                "tags": seg_tags
            })
        # Append segment features to features
        features.extend(segment_features)
    
    # Save features
    features_file = output_dir / "frame_features.json"
    with open(features_file, 'w') as f:
        json.dump(features, f, indent=2)
    
    # Save embeddings separately for FAISS
    embeddings = [f["embedding"] for f in features]
    embeddings_file = output_dir / "embeddings.npy"
    np.save(embeddings_file, np.array(embeddings))
    
    # Save metadata
    metadata_file = output_dir / "feature_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "num_features": len(features),
            "embedding_dim": len(embeddings[0]) if embeddings else 0,
            "feature_extraction_model": "lightweight_cv",
            "text_encoder_model": model_name,
            "use_heavy_models": False,
            "segment_captions": segment_captions,
            "object_tags": object_tags
        }, f, indent=2)
    
    logger.info(f"Saved {len(features)} feature vectors to {output_dir}")
    return features


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python feature_extractor.py <frame_metadata.json> <output_directory> [model_name]")
        sys.exit(1)
    
    metadata_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    model_name = sys.argv[3] if len(sys.argv) > 3 else "all-MiniLM-L6-v2"
    
    # Load frame metadata
    with open(metadata_file, 'r') as f:
        frame_metadata = json.load(f)
    
    # Extract features
    features = extract_features_from_frames(frame_metadata, output_dir, model_name)
    print(f"Extracted features from {len(features)} frames")
