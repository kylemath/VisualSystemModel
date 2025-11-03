"""
Foveal Input System with Eye Movements
=======================================

Integrates eye movement controller with webcam input to simulate
realistic foveal vision with saccades, microsaccades, and vergence.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from eye_movements import EyeMovementController
from webcam_input import WebcamInput, SimulatedWebcam


class FovealInputSystem:
    """
    Combines eye movements with webcam input to extract foveal regions.
    
    The full webcam provides a large visual field, and eye movements
    select which region each eye views (like moving a high-res camera
    across a large scene).
    """
    
    def __init__(self, 
                 use_real_webcam: bool = False,
                 webcam_size: Tuple[int, int] = (320, 240),  # Full "visual field"
                 foveal_size: Tuple[int, int] = (64, 64),     # High-res "fovea"
                 camera_index: int = 0):
        """
        Initialize foveal input system.
        
        Args:
            use_real_webcam: Use real webcam vs simulated
            webcam_size: Size of full visual field
            foveal_size: Size of extracted foveal region
            camera_index: Camera device index if using real webcam
        """
        self.webcam_size = webcam_size
        self.foveal_size = foveal_size
        
        # Initialize webcam (provides full field of view)
        if use_real_webcam:
            self.webcam = WebcamInput(
                target_size=webcam_size,
                camera_index=camera_index
            )
            self.webcam.start()
        else:
            self.webcam = SimulatedWebcam(
                target_size=webcam_size
            )
            self.webcam.start()
        
        # Initialize eye movement controller
        self.eye_controller = EyeMovementController(
            microsaccade_rate=1.0,  # 1 per second
            drift_magnitude=0.002
        )
        
        # Storage for full frames (numpy uses height, width, channels)
        self.full_left_frame = np.zeros((webcam_size[1], webcam_size[0], 3))
        self.full_right_frame = np.zeros((webcam_size[1], webcam_size[0], 3))
        
        # Storage for extracted foveal regions
        self.foveal_left = np.zeros((foveal_size[1], foveal_size[0], 3))
        self.foveal_right = np.zeros((foveal_size[1], foveal_size[0], 3))
    
    def update(self, dt: float):
        """
        Update eye movements and extract new foveal regions.
        
        Args:
            dt: Time step in milliseconds
        """
        # Update eye positions
        self.eye_controller.update(dt)
        
        # Get full frames from webcam
        self.full_left_frame, self.full_right_frame = self.webcam.get_frames()
        
        # Extract foveal regions based on eye positions
        self._extract_foveal_regions()
    
    def _extract_foveal_regions(self):
        """Extract foveal regions from full frames based on eye positions."""
        # Get bounding boxes for each eye
        regions = self.eye_controller.get_frame_regions(
            self.webcam_size[0],  # width
            self.webcam_size[1],  # height
            fov_size=self.foveal_size[0]
        )
        
        # Extract and resize regions
        for eye in ['left', 'right']:
            x1, y1, x2, y2 = regions[eye]
            
            # Get full frame for this eye
            full_frame = self.full_left_frame if eye == 'left' else self.full_right_frame
            
            # Extract region
            region = full_frame[y1:y2, x1:x2]
            
            # Handle edge cases where region might be smaller than expected
            if region.shape[0] > 0 and region.shape[1] > 0:
                # Resize to foveal size
                resized = cv2.resize(region, self.foveal_size, 
                                    interpolation=cv2.INTER_LINEAR)
            else:
                # Use blank frame if region is invalid
                resized = np.zeros((*self.foveal_size, 3))
            
            # Store
            if eye == 'left':
                self.foveal_left = resized
            else:
                self.foveal_right = resized
    
    def get_foveal_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current foveal regions for each eye.
        
        Returns:
            (left_fovea, right_fovea) both shape (foveal_size[1], foveal_size[0], 3)
        """
        return self.foveal_left.copy(), self.foveal_right.copy()
    
    def get_full_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get full visual field frames (for visualization)."""
        return self.full_left_frame.copy(), self.full_right_frame.copy()
    
    def saccade_to(self, x: float, y: float):
        """
        Command a saccade to target location.
        
        Args:
            x: Target x in normalized coordinates [-1, 1]
            y: Target y in normalized coordinates [-1, 1]
        """
        self.eye_controller.saccade_to(x, y, binocular=True)
    
    def set_vergence(self, angle: float):
        """
        Set vergence angle for depth.
        
        Args:
            angle: Vergence in degrees (0=parallel/far, +15=converged/near)
        """
        self.eye_controller.set_vergence(angle)
    
    def get_eye_state(self) -> Dict:
        """Get current eye movement state."""
        state = self.eye_controller.get_state()
        
        # Add visualization info
        regions = self.eye_controller.get_frame_regions(
            self.webcam_size[0],
            self.webcam_size[1],
            fov_size=self.foveal_size[0]
        )
        state['foveal_regions'] = regions
        
        return state
    
    def stop(self):
        """Stop webcam capture."""
        if hasattr(self.webcam, 'stop'):
            self.webcam.stop()

