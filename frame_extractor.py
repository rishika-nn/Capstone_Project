"""
Frame extraction and deduplication module for video search pipeline.
"""

import cv2
import numpy as np
import imagehash
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from tqdm import tqdm
import json
from collections import defaultdict

from config import (
    FRAME_EXTRACTION_RATE, 
    MAX_KEYFRAMES_PER_SHOT,
    PERCEPTUAL_HASH_THRESHOLD,
    KEYFRAMES_DIR,
    SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extracts keyframes from videos with deduplication."""
    
    def __init__(self, extraction_rate: float = FRAME_EXTRACTION_RATE):
        self.extraction_rate = extraction_rate
        self.frame_hashes = {}
        
    def extract_frames_from_video(self, video_path: Path, output_dir: Path) -> List[Dict]:
        """
        Extract keyframes from a video file.
        
        Args:
            video_path: Path to the input video
            output_dir: Directory to save extracted frames
            
        Returns:
            List of frame metadata dictionaries
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Processing video: {video_path.name}")
        logger.info(f"Duration: {duration:.2f}s, FPS: {fps:.2f}, Total frames: {total_frames}")
        
        frame_interval = int(fps / self.extraction_rate)
        extracted_frames = []
        frame_count = 0
        
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    frame_data = self._process_frame(
                        frame, frame_count, timestamp, video_path, output_dir
                    )
                    if frame_data:
                        extracted_frames.append(frame_data)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        
        # Save metadata
        metadata_path = output_dir / f"{video_path.stem}_frames.json"
        with open(metadata_path, 'w') as f:
            json.dump(extracted_frames, f, indent=2)
        
        logger.info(f"Extracted {len(extracted_frames)} frames from {video_path.name}")
        return extracted_frames
    
    def _process_frame(self, frame: np.ndarray, frame_number: int, 
                      timestamp: float, video_path: Path, output_dir: Path) -> Optional[Dict]:
        """Process a single frame and save if not duplicate."""
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Compute perceptual hash
        frame_hash = imagehash.phash(pil_image)
        
        # Check for duplicates
        if self._is_duplicate(frame_hash):
            return None
        
        # Save frame
        frame_filename = f"{video_path.stem}_frame_{frame_number:06d}_{timestamp:.2f}s.jpg"
        frame_path = output_dir / frame_filename
        pil_image.save(frame_path, quality=95)
        
        # Store hash for future comparisons
        self.frame_hashes[frame_hash] = frame_path
        
        return {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "frame_path": str(frame_path),
            "video_path": str(video_path),
            "hash": str(frame_hash)
        }
    
    def _is_duplicate(self, frame_hash: imagehash.ImageHash) -> bool:
        """Check if frame is a duplicate based on perceptual hash."""
        for existing_hash in self.frame_hashes.keys():
            if frame_hash - existing_hash <= PERCEPTUAL_HASH_THRESHOLD:
                return True
        return False
    
    def detect_shots(self, video_path: Path) -> List[Tuple[float, float]]:
        """
        Detect shot boundaries in video using frame difference.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of (start_time, end_time) tuples for each shot
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        shots = []
        prev_frame = None
        shot_start = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Compute frame difference
                diff = cv2.absdiff(prev_frame, gray_frame)
                mean_diff = np.mean(diff)
                
                # Shot boundary detection
                if mean_diff > 30:  # Threshold for shot change
                    shot_end = frame_count / fps
                    shots.append((shot_start, shot_end))
                    shot_start = shot_end
            
            prev_frame = gray_frame
            frame_count += 1
        
        # Add final shot
        if shots:
            final_time = frame_count / fps
            shots.append((shots[-1][1], final_time))
        else:
            shots.append((0, frame_count / fps))
        
        cap.release()
        return shots
    
    def extract_keyframes_from_shots(self, video_path: Path, output_dir: Path) -> List[Dict]:
        """
        Extract keyframes using shot boundary detection.
        
        Args:
            video_path: Path to the input video
            output_dir: Directory to save extracted frames
            
        Returns:
            List of frame metadata dictionaries
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect shots
        shots = self.detect_shots(video_path)
        logger.info(f"Detected {len(shots)} shots in {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        extracted_frames = []
        
        for shot_idx, (start_time, end_time) in enumerate(shots):
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Extract keyframes from this shot
            keyframe_indices = np.linspace(
                start_frame, end_frame - 1, 
                min(MAX_KEYFRAMES_PER_SHOT, end_frame - start_frame), 
                dtype=int
            )
            
            for frame_idx in keyframe_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    timestamp = frame_idx / fps
                    frame_data = self._process_frame(
                        frame, frame_idx, timestamp, video_path, output_dir
                    )
                    if frame_data:
                        frame_data["shot_idx"] = shot_idx
                        frame_data["shot_time"] = (start_time, end_time)
                        extracted_frames.append(frame_data)
        
        cap.release()
        
        # Save metadata
        metadata_path = output_dir / f"{video_path.stem}_shots.json"
        with open(metadata_path, 'w') as f:
            json.dump(extracted_frames, f, indent=2)
        
        logger.info(f"Extracted {len(extracted_frames)} keyframes from {len(shots)} shots")
        return extracted_frames


class FrameDeduplicator:
    """Advanced deduplication using feature similarity."""
    
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        
    def cluster_similar_frames(self, frame_paths: List[Path]) -> List[List[Path]]:
        """
        Cluster frames by visual similarity using feature extraction.
        
        Args:
            frame_paths: List of paths to frame images
            
        Returns:
            List of clusters, where each cluster contains similar frame paths
        """
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Extract features (simplified - using basic image features)
        features = []
        valid_paths = []
        
        for path in tqdm(frame_paths, desc="Extracting frame features"):
            try:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                    
                # Simple feature extraction (histogram)
                hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                features.append(hist.flatten())
                valid_paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")
                continue
        
        if not features:
            return []
        
        features = np.array(features)
        
        # Normalize features
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(features)
        
        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix
        
        # Cluster using DBSCAN
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold, 
            min_samples=2, 
            metric='precomputed'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group frames by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Not noise
                clusters[label].append(valid_paths[i])
        
        # Add noise points as individual clusters
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise
                clusters[f"noise_{i}"].append(valid_paths[i])
        
        return list(clusters.values())
    
    def select_representative_frames(self, clusters: List[List[Path]]) -> List[Path]:
        """
        Select representative frame from each cluster.
        
        Args:
            clusters: List of frame clusters
            
        Returns:
            List of representative frame paths
        """
        representatives = []
        
        for cluster in clusters:
            if len(cluster) == 1:
                representatives.append(cluster[0])
            else:
                # Select frame with median file size (proxy for quality)
                sizes = [(path, path.stat().st_size) for path in cluster]
                sizes.sort(key=lambda x: x[1])
                median_idx = len(sizes) // 2
                representatives.append(sizes[median_idx][0])
        
        return representatives


def extract_frames_from_videos(video_dir: Path, output_dir: Path, 
                              method: str = "uniform") -> List[Dict]:
    """
    Extract frames from all videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted frames
        method: Extraction method ("uniform" or "shot_based")
        
    Returns:
        List of all extracted frame metadata
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = [f for f in video_dir.iterdir() 
                   if f.suffix.lower() in video_extensions]
    
    if not video_files:
        logger.warning(f"No video files found in {video_dir}")
        return []
    
    extractor = FrameExtractor()
    all_frames = []
    
    for video_file in video_files:
        try:
            if method == "shot_based":
                frames = extractor.extract_keyframes_from_shots(video_file, output_dir)
            else:
                frames = extractor.extract_frames_from_video(video_file, output_dir)
            all_frames.extend(frames)
        except Exception as e:
            logger.error(f"Failed to process {video_file}: {e}")
            continue
    
    logger.info(f"Total frames extracted: {len(all_frames)}")
    return all_frames


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python frame_extractor.py <video_directory> <output_directory>")
        sys.exit(1)
    
    video_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    # Extract frames
    frames = extract_frames_from_videos(video_dir, output_dir, method="shot_based")
    print(f"Extracted {len(frames)} frames")

