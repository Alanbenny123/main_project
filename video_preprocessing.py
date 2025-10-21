"""
Video Preprocessing Module for Student Behavior Analysis
Handles video input, frame extraction, quality enhancement, and optimization
"""

import cv2
import numpy as np
from pathlib import Path
import os
from typing import List, Tuple, Optional, Dict, Generator
from dataclasses import dataclass
import time


@dataclass
class VideoMetadata:
    """Store video metadata"""
    filename: str
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int
    codec: str
    size_mb: float


@dataclass
class PreprocessingConfig:
    """Configuration for video preprocessing"""
    # Frame sampling
    sample_rate: int = 30  # Process every Nth frame
    max_frames: int = 1000  # Maximum frames to process
    
    # Resolution
    target_width: int = 640
    target_height: int = 480
    maintain_aspect_ratio: bool = True
    
    # Quality enhancement
    denoise: bool = True
    enhance_contrast: bool = True
    color_correction: bool = True
    
    # ROI (Region of Interest)
    detect_roi: bool = False  # Auto-detect classroom area
    roi_coordinates: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    
    # Output
    save_processed_frames: bool = False
    output_dir: str = 'processed_frames'


class VideoPreprocessor:
    """
    Main video preprocessing class
    Handles all video processing operations
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        
        if self.config.save_processed_frames:
            os.makedirs(self.config.output_dir, exist_ok=True)
    
    
    # ============= Video Loading & Metadata =============
    
    def load_video(self, video_path: str) -> cv2.VideoCapture:
        """
        Load video file and validate
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        print(f"✓ Loaded video: {video_path}")
        return cap
    
    
    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract video metadata
        """
        cap = self.load_video(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Decode codec
        codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])
        
        # Calculate duration
        duration = total_frames / fps if fps > 0 else 0
        
        # Get file size
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        cap.release()
        
        metadata = VideoMetadata(
            filename=os.path.basename(video_path),
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            width=width,
            height=height,
            codec=codec,
            size_mb=size_mb
        )
        
        print(f"\n{'='*60}")
        print(f"Video Metadata: {metadata.filename}")
        print(f"{'='*60}")
        print(f"Resolution: {metadata.width}x{metadata.height}")
        print(f"FPS: {metadata.fps:.2f}")
        print(f"Total Frames: {metadata.total_frames}")
        print(f"Duration: {metadata.duration:.2f}s ({metadata.duration/60:.2f} min)")
        print(f"Codec: {metadata.codec}")
        print(f"Size: {metadata.size_mb:.2f} MB")
        print(f"{'='*60}\n")
        
        return metadata
    
    
    # ============= Frame Extraction =============
    
    def extract_frames(
        self, 
        video_path: str,
        sample_rate: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields sampled frames
        Memory efficient - doesn't load all frames at once
        
        Yields:
            (frame_number, frame_image)
        """
        cap = self.load_video(video_path)
        sample_rate = sample_rate or self.config.sample_rate
        
        frame_count = 0
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample frames
            if frame_count % sample_rate == 0:
                if processed_count >= self.config.max_frames:
                    break
                
                yield frame_count, frame
                processed_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"✓ Extracted {processed_count} frames from {frame_count} total")
    
    
    def extract_frames_batch(
        self,
        video_path: str,
        sample_rate: Optional[int] = None
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract all sampled frames into memory
        Use for short videos only!
        """
        frames = list(self.extract_frames(video_path, sample_rate))
        print(f"✓ Loaded {len(frames)} frames into memory")
        return frames
    
    
    # ============= Frame Preprocessing =============
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to target resolution
        Maintains aspect ratio if configured
        """
        target_h = self.config.target_height
        target_w = self.config.target_width
        
        if self.config.maintain_aspect_ratio:
            # Calculate scaling factor
            h, w = frame.shape[:2]
            scale = min(target_w / w, target_h / h)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad to target size
            delta_w = target_w - new_w
            delta_h = target_h - new_h
            top = delta_h // 2
            bottom = delta_h - top
            left = delta_w // 2
            right = delta_w - left
            
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            
            return padded
        else:
            # Simple resize
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    
    def denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Remove noise from frame using Non-Local Means Denoising
        Good for low-quality classroom cameras
        """
        if not self.config.denoise:
            return frame
        
        # Convert to grayscale for denoising
        if len(frame.shape) == 3:
            # Color image
            denoised = cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                h=10,  # Filter strength
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            # Grayscale
            denoised = cv2.fastNlMeansDenoising(
                frame,
                None,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
        
        return denoised
    
    
    def enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Adaptive Histogram Equalization)
        Improves visibility in poor lighting
        """
        if not self.config.enhance_contrast:
            return frame
        
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l_enhanced, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    
    def correct_color(self, frame: np.ndarray) -> np.ndarray:
        """
        Auto white balance and color correction
        """
        if not self.config.color_correction:
            return frame
        
        # Simple white balance using Gray World assumption
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        
        return result
    
    
    def crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop to Region of Interest (classroom area)
        Reduces processing on irrelevant areas
        """
        if not self.config.roi_coordinates:
            return frame
        
        x, y, w, h = self.config.roi_coordinates
        return frame[y:y+h, x:x+w]
    
    
    def detect_motion_blur(self, frame: np.ndarray) -> float:
        """
        Detect motion blur using Laplacian variance
        Returns blur score (higher = less blur)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all preprocessing steps to a single frame
        Pipeline: ROI → Denoise → Contrast → Color → Resize
        """
        # Step 1: Crop ROI (if specified)
        if self.config.roi_coordinates:
            frame = self.crop_roi(frame)
        
        # Step 2: Denoise
        if self.config.denoise:
            frame = self.denoise_frame(frame)
        
        # Step 3: Enhance contrast
        if self.config.enhance_contrast:
            frame = self.enhance_contrast(frame)
        
        # Step 4: Color correction
        if self.config.color_correction:
            frame = self.correct_color(frame)
        
        # Step 5: Resize
        frame = self.resize_frame(frame)
        
        return frame
    
    
    # ============= Video Processing Pipeline =============
    
    def process_video(
        self,
        video_path: str,
        callback: Optional[callable] = None
    ) -> Dict:
        """
        Complete video processing pipeline
        
        Args:
            video_path: Path to video file
            callback: Optional function to call on each frame
                     callback(frame_number, processed_frame) -> result
        
        Returns:
            Dictionary with processing results
        """
        metadata = self.get_video_metadata(video_path)
        
        results = {
            'metadata': metadata,
            'processed_frames': 0,
            'total_frames': metadata.total_frames,
            'processing_time': 0,
            'frames_per_second': 0,
            'blur_scores': [],
            'callback_results': []
        }
        
        start_time = time.time()
        
        print(f"Processing video: {metadata.filename}")
        print(f"Config: sample_rate={self.config.sample_rate}, max_frames={self.config.max_frames}")
        print(f"Target resolution: {self.config.target_width}x{self.config.target_height}")
        print(f"Enhancements: denoise={self.config.denoise}, contrast={self.config.enhance_contrast}")
        print()
        
        for frame_num, frame in self.extract_frames(video_path):
            # Preprocess frame
            processed = self.preprocess_frame(frame)
            
            # Check blur (quality control)
            blur_score = self.detect_motion_blur(processed)
            results['blur_scores'].append(blur_score)
            
            # Save frame if configured
            if self.config.save_processed_frames:
                output_path = os.path.join(
                    self.config.output_dir,
                    f"frame_{frame_num:06d}.jpg"
                )
                cv2.imwrite(output_path, processed)
            
            # Call callback (for analysis)
            if callback:
                callback_result = callback(frame_num, processed)
                results['callback_results'].append(callback_result)
            
            results['processed_frames'] += 1
            
            # Progress indicator
            if results['processed_frames'] % 10 == 0:
                elapsed = time.time() - start_time
                fps = results['processed_frames'] / elapsed if elapsed > 0 else 0
                print(f"Processed {results['processed_frames']} frames | {fps:.2f} frames/sec", end='\r')
        
        # Finalize results
        end_time = time.time()
        results['processing_time'] = end_time - start_time
        results['frames_per_second'] = results['processed_frames'] / results['processing_time']
        results['avg_blur_score'] = np.mean(results['blur_scores']) if results['blur_scores'] else 0
        
        print(f"\n\n{'='*60}")
        print(f"Processing Complete!")
        print(f"{'='*60}")
        print(f"Processed frames: {results['processed_frames']}")
        print(f"Total time: {results['processing_time']:.2f}s")
        print(f"Speed: {results['frames_per_second']:.2f} frames/sec")
        print(f"Avg blur score: {results['avg_blur_score']:.2f}")
        print(f"{'='*60}\n")
        
        return results
    
    
    # ============= Advanced Features =============
    
    def detect_scene_changes(
        self,
        video_path: str,
        threshold: float = 30.0
    ) -> List[int]:
        """
        Detect scene changes (e.g., camera switches in classroom)
        Returns list of frame numbers where scenes change
        """
        scene_changes = []
        prev_frame = None
        
        for frame_num, frame in self.extract_frames(video_path, sample_rate=5):
            if prev_frame is not None:
                # Calculate frame difference
                gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
                diff = cv2.absdiff(gray_current, gray_prev)
                mean_diff = np.mean(diff)
                
                if mean_diff > threshold:
                    scene_changes.append(frame_num)
            
            prev_frame = frame
        
        print(f"✓ Detected {len(scene_changes)} scene changes")
        return scene_changes
    
    
    def stabilize_video(self, frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> np.ndarray:
        """
        Simple video stabilization using optical flow
        Reduces camera shake
        """
        if prev_frame is None:
            return frame
        
        # Convert to grayscale
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        prev_pts = cv2.goodFeaturesToTrack(
            gray_prev,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30
        )
        
        if prev_pts is None:
            return frame
        
        # Calculate optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_current, prev_pts, None
        )
        
        # Filter valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        if len(prev_pts) < 5:
            return frame
        
        # Estimate transform
        transform = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        
        if transform is None:
            return frame
        
        # Apply transform
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(frame, transform, (w, h))
        
        return stabilized


# ============= Utility Functions =============

def normalize_frame_for_model(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Normalize frame for ML model input
    Standard preprocessing for Swin Transformer
    """
    # Resize to model input size
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    # Add batch dimension
    batched = np.expand_dims(normalized, axis=0)
    
    return batched


def create_video_summary(frames: List[np.ndarray], grid_size: Tuple[int, int] = (4, 4)) -> np.ndarray:
    """
    Create a grid summary of video frames
    Useful for quick visualization
    """
    rows, cols = grid_size
    total_frames = rows * cols
    
    # Sample frames evenly
    if len(frames) > total_frames:
        indices = np.linspace(0, len(frames) - 1, total_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]
    else:
        sampled_frames = frames
    
    # Resize frames to same size
    frame_h, frame_w = sampled_frames[0].shape[:2]
    cell_h, cell_w = frame_h // 4, frame_w // 4
    
    resized = [cv2.resize(f, (cell_w, cell_h)) for f in sampled_frames]
    
    # Create grid
    grid_rows = []
    for i in range(rows):
        start_idx = i * cols
        end_idx = start_idx + cols
        row_frames = resized[start_idx:end_idx]
        
        # Pad if needed
        while len(row_frames) < cols:
            row_frames.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
        
        row = np.hstack(row_frames)
        grid_rows.append(row)
    
    grid = np.vstack(grid_rows)
    
    return grid


# ============= Example Usage =============

if __name__ == '__main__':
    # Configuration
    config = PreprocessingConfig(
        sample_rate=30,
        max_frames=100,
        target_width=640,
        target_height=480,
        denoise=True,
        enhance_contrast=True,
        color_correction=True,
        save_processed_frames=False
    )
    
    # Create preprocessor
    preprocessor = VideoPreprocessor(config)
    
    # Example video path
    video_path = 'uploads/classroom_video.mp4'
    
    if os.path.exists(video_path):
        # Get metadata
        metadata = preprocessor.get_video_metadata(video_path)
        
        # Process video with callback
        def analysis_callback(frame_num, processed_frame):
            # Here you would call YOLO, InsightFace, Swin Transformer
            print(f"Analyzing frame {frame_num}...")
            return {'frame': frame_num, 'analyzed': True}
        
        results = preprocessor.process_video(video_path, callback=analysis_callback)
        
        print(f"\nResults: {results['processed_frames']} frames processed")
        print(f"Average processing speed: {results['frames_per_second']:.2f} fps")
    else:
        print(f"Example video not found: {video_path}")
        print("Place a video file to test the preprocessor")

