"""
Webcam input capture and preprocessing for neural network.
Provides downsampled, continuous visual input.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import threading
import time


class WebcamInput:
    """
    Continuous webcam capture with downsampling.
    Runs in separate thread for real-time performance.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64), 
                 camera_index: int = 0,
                 fps: int = 30):
        """
        Initialize webcam input.
        
        Args:
            target_size: (width, height) for downsampled output
            camera_index: Camera device index (0 for default)
            fps: Target frames per second
        """
        self.target_size = target_size
        self.camera_index = camera_index
        self.target_fps = fps
        self.frame_time = 1.0 / fps
        
        # Current frames (binocular simulation: same frame, slight offset)
        self.left_frame = np.zeros((*target_size, 3))
        self.right_frame = np.zeros((*target_size, 3))
        
        # Webcam capture
        self.capture = None
        self.is_running = False
        self.capture_thread = None
        
        # Statistics
        self.frame_count = 0
        self.actual_fps = 0.0
        self.last_frame_time = 0.0
        
    def start(self) -> bool:
        """Start webcam capture."""
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            
            if not self.capture.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            print(f"Webcam started successfully (target: {self.target_fps} fps)")
            return True
            
        except Exception as e:
            print(f"Error starting webcam: {e}")
            return False
    
    def stop(self):
        """Stop webcam capture."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.capture:
            self.capture.release()
        
        print("Webcam stopped")
    
    def _capture_loop(self):
        """Continuous capture loop (runs in separate thread)."""
        last_time = time.time()
        
        while self.is_running:
            loop_start = time.time()
            
            # Capture frame
            ret, frame = self.capture.read()
            
            if not ret:
                print("Warning: Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Process frame
            processed = self._process_frame(frame)
            
            # Update both eyes (simulate binocular vision)
            # For now: same frame, could add slight offset for stereo
            self.left_frame = processed.copy()
            self.right_frame = processed  # Could add horizontal shift here
            
            # Update statistics
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - last_time
            
            if elapsed > 1.0:  # Update FPS every second
                self.actual_fps = self.frame_count / elapsed
                self.frame_count = 0
                last_time = current_time
            
            self.last_frame_time = current_time
            
            # Frame rate control
            loop_time = time.time() - loop_start
            sleep_time = max(0, self.frame_time - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process raw frame: resize, normalize, convert to RGB.
        
        Args:
            frame: Raw BGR frame from OpenCV
            
        Returns:
            Processed RGB frame normalized to [0, 1]
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        resized = cv2.resize(frame_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current left and right eye frames.
        
        Returns:
            (left_frame, right_frame) both shape (height, width, 3), values [0, 1]
        """
        return self.left_frame.copy(), self.right_frame.copy()
    
    def get_status(self) -> Dict:
        """Get webcam status."""
        return {
            'is_running': self.is_running,
            'target_size': self.target_size,
            'target_fps': self.target_fps,
            'actual_fps': round(self.actual_fps, 1),
            'frame_count': self.frame_count,
            'camera_index': self.camera_index
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


class SimulatedWebcam:
    """
    Simulated webcam for testing without hardware.
    Generates procedural patterns.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64), fps: int = 30):
        self.target_size = target_size
        self.target_fps = fps
        self.frame_time = 1.0 / fps
        
        self.left_frame = np.zeros((*target_size, 3))
        self.right_frame = np.zeros((*target_size, 3))
        
        self.is_running = False
        self.capture_thread = None
        self.frame_count = 0
        self.actual_fps = fps
        self.last_frame_time = 0.0
        
        self.time = 0.0  # Simulation time
    
    def start(self) -> bool:
        """Start simulated capture."""
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._generate_loop, daemon=True)
        self.capture_thread.start()
        print(f"Simulated webcam started ({self.target_fps} fps)")
        return True
    
    def stop(self):
        """Stop simulated capture."""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        print("Simulated webcam stopped")
    
    def _generate_loop(self):
        """Generate frames continuously."""
        last_time = time.time()
        
        while self.is_running:
            loop_start = time.time()
            
            # Generate frame
            frame = self._generate_frame()
            
            self.left_frame = frame.copy()
            self.right_frame = frame
            
            # Update stats
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - last_time
            
            if elapsed > 1.0:
                self.actual_fps = self.frame_count / elapsed
                self.frame_count = 0
                last_time = current_time
            
            self.last_frame_time = current_time
            self.time += self.frame_time
            
            # Rate control
            loop_time = time.time() - loop_start
            sleep_time = max(0, self.frame_time - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _generate_frame(self) -> np.ndarray:
        """Generate procedural test pattern with strong contrast."""
        h, w = self.target_size
        frame = np.ones((h, w, 3)) * 0.3  # Gray background
        
        # Create several bright spots moving around
        # Main bright spot (moving)
        center_x1 = int(w / 2 + w / 3 * np.sin(self.time * 0.5))
        center_y1 = int(h / 2 + h / 3 * np.cos(self.time * 0.7))
        
        # Secondary spot
        center_x2 = int(w / 2 + w / 4 * np.sin(self.time * 0.8 + 1.5))
        center_y2 = int(h / 2 + h / 4 * np.cos(self.time * 0.6 + 2.0))
        
        # Draw bright spots (high contrast)
        for (cx, cy, radius, color) in [
            (center_x1, center_y1, 15, [1.0, 1.0, 1.0]),  # White spot
            (center_x2, center_y2, 10, [1.0, 0.2, 0.2]),  # Red spot
        ]:
            for i in range(max(0, cy-radius), min(h, cy+radius)):
                for j in range(max(0, cx-radius), min(w, cx+radius)):
                    dist = np.sqrt((i-cy)**2 + (j-cx)**2)
                    if dist < radius:
                        brightness = (1 - dist/radius) ** 2  # Smooth falloff
                        # Blend with background
                        frame[i, j] = frame[i, j] * (1 - brightness) + np.array(color) * brightness
        
        # Add some vertical bars for spatial structure
        bar_positions = [w//4, w//2, 3*w//4]
        for bar_x in bar_positions:
            frame[:, max(0, bar_x-2):min(w, bar_x+2)] = 0.7
        
        return np.clip(frame, 0, 1)
    
    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current frames."""
        return self.left_frame.copy(), self.right_frame.copy()
    
    def get_status(self) -> Dict:
        """Get status."""
        return {
            'is_running': self.is_running,
            'target_size': self.target_size,
            'target_fps': self.target_fps,
            'actual_fps': round(self.actual_fps, 1),
            'frame_count': self.frame_count,
            'simulated': True
        }
    
    def __del__(self):
        self.stop()

