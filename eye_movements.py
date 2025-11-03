"""
Eye Movement Controller
=======================

Simulates realistic eye movements including:
- Saccades (fast jumps to new fixation points)
- Microsaccades (small fixational movements)
- Smooth pursuit (tracking moving objects)
- Vergence (binocular depth adjustment)
- Drift and tremor (maintaining differential input)

Eye positions select regions from the full webcam frame,
allowing the fovea to explore different parts of the visual scene.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import time


class EyeMovementController:
    """
    Controls eye positions and movements for binocular vision.
    
    Coordinate system:
    - (0, 0) = center of webcam frame
    - x: horizontal (-1 left, +1 right)
    - y: vertical (-1 bottom, +1 top)
    - vergence: angle difference between eyes (0 = parallel, + = converged)
    """
    
    def __init__(self, 
                 interocular_distance: float = 0.065,  # meters, typical human
                 microsaccade_rate: float = 1.0,  # per second
                 drift_magnitude: float = 0.002):  # deg/ms
        """
        Initialize eye movement controller.
        
        Args:
            interocular_distance: Distance between eyes (for binocular disparity)
            microsaccade_rate: Frequency of microsaccades
            drift_magnitude: Speed of ocular drift
        """
        # Eye positions (in normalized frame coordinates)
        self.left_eye_pos = np.array([0.0, 0.0])  # [x, y]
        self.right_eye_pos = np.array([0.0, 0.0])  # [x, y]
        
        # Vergence state (in degrees, + = converged for near objects)
        self.vergence_angle = 0.0  # degrees
        self.target_vergence = 0.0
        
        # Movement parameters
        self.interocular_distance = interocular_distance
        self.microsaccade_rate = microsaccade_rate
        self.drift_magnitude = drift_magnitude
        
        # Microsaccade timing
        self.last_microsaccade_time = time.time()
        self.microsaccade_interval = 1.0 / microsaccade_rate  # seconds
        
        # Drift accumulator
        self.drift_direction = np.random.randn(2)
        self.drift_direction /= np.linalg.norm(self.drift_direction)
        
        # Saccade state
        self.is_saccading = False
        self.saccade_start_time = 0
        self.saccade_duration = 0
        self.saccade_start_pos = {'left': np.array([0.0, 0.0]), 
                                  'right': np.array([0.0, 0.0])}
        self.saccade_target_pos = {'left': np.array([0.0, 0.0]), 
                                   'right': np.array([0.0, 0.0])}
        
        # Binocular disparity (horizontal offset for stereopsis)
        self.baseline_disparity = 0.032  # ~3.2cm in normalized coords
        
    def update(self, dt: float):
        """
        Update eye positions with natural movements.
        
        Args:
            dt: Time step in milliseconds
        """
        current_time = time.time()
        
        # 1. Execute ongoing saccade
        if self.is_saccading:
            self._update_saccade(current_time, dt)
        else:
            # 2. Microsaccades (prevent adaptation)
            if current_time - self.last_microsaccade_time > self.microsaccade_interval:
                self._trigger_microsaccade()
                self.last_microsaccade_time = current_time
            
            # 3. Ocular drift (slow random walk)
            self._apply_drift(dt)
            
            # 4. Ocular tremor (high-frequency oscillation)
            self._apply_tremor()
        
        # 5. Update vergence (smooth convergence/divergence)
        self._update_vergence(dt)
        
        # 6. Apply binocular disparity
        self._apply_binocular_disparity()
    
    def _trigger_microsaccade(self):
        """Trigger a small microsaccade."""
        # Microsaccades: 0.5-2 degrees, typically ~1 degree
        magnitude = np.random.uniform(0.01, 0.03)  # in normalized coords
        direction = np.random.randn(2)
        direction /= np.linalg.norm(direction)
        
        displacement = direction * magnitude
        
        # Apply to both eyes (conjugate movement)
        target_left = self.left_eye_pos + displacement
        target_right = self.right_eye_pos + displacement
        
        # Trigger saccade
        self._initiate_saccade(
            target_left, target_right,
            duration=20.0  # 20ms typical microsaccade duration
        )
    
    def _apply_drift(self, dt: float):
        """Apply slow ocular drift."""
        if self.is_saccading:
            return
        
        # Change drift direction occasionally
        if np.random.rand() < 0.01:  # 1% chance per update
            self.drift_direction = np.random.randn(2)
            self.drift_direction /= np.linalg.norm(self.drift_direction)
        
        # Apply drift
        drift = self.drift_direction * self.drift_magnitude * dt / 1000.0
        self.left_eye_pos += drift
        self.right_eye_pos += drift
        
        # Constrain to valid range
        self._constrain_positions()
    
    def _apply_tremor(self):
        """Apply high-frequency ocular tremor (~90 Hz, very small amplitude)."""
        if self.is_saccading:
            return
        
        # Tremor: ~0.02 degrees amplitude, very small
        tremor_amplitude = 0.0002  # in normalized coords
        tremor = np.random.randn(2) * tremor_amplitude
        
        self.left_eye_pos += tremor
        self.right_eye_pos += tremor
        
        self._constrain_positions()
    
    def _initiate_saccade(self, target_left: np.ndarray, target_right: np.ndarray, 
                          duration: float = 50.0):
        """
        Initiate a saccade to target position.
        
        Args:
            target_left: Target position for left eye
            target_right: Target position for right eye
            duration: Saccade duration in ms
        """
        self.is_saccading = True
        self.saccade_start_time = time.time()
        self.saccade_duration = duration / 1000.0  # convert to seconds
        
        self.saccade_start_pos['left'] = self.left_eye_pos.copy()
        self.saccade_start_pos['right'] = self.right_eye_pos.copy()
        
        self.saccade_target_pos['left'] = target_left
        self.saccade_target_pos['right'] = target_right
    
    def _update_saccade(self, current_time: float, dt: float):
        """Update ongoing saccade with realistic dynamics."""
        elapsed = current_time - self.saccade_start_time
        progress = elapsed / self.saccade_duration
        
        if progress >= 1.0:
            # Saccade complete
            self.left_eye_pos = self.saccade_target_pos['left'].copy()
            self.right_eye_pos = self.saccade_target_pos['right'].copy()
            self.is_saccading = False
        else:
            # Smooth interpolation with sigmoid
            # Saccades have characteristic velocity profile (bell-shaped)
            t = progress
            smooth = 3*t**2 - 2*t**3  # Smoothstep function
            
            self.left_eye_pos = (
                self.saccade_start_pos['left'] + 
                smooth * (self.saccade_target_pos['left'] - self.saccade_start_pos['left'])
            )
            self.right_eye_pos = (
                self.saccade_start_pos['right'] + 
                smooth * (self.saccade_target_pos['right'] - self.saccade_start_pos['right'])
            )
        
        self._constrain_positions()
    
    def _update_vergence(self, dt: float):
        """Update vergence angle (smooth convergence/divergence)."""
        # Smooth vergence change
        vergence_speed = 10.0  # degrees per second
        diff = self.target_vergence - self.vergence_angle
        
        if abs(diff) > 0.01:
            change = np.sign(diff) * min(abs(diff), vergence_speed * dt / 1000.0)
            self.vergence_angle += change
    
    def _apply_binocular_disparity(self):
        """Apply binocular disparity based on vergence."""
        # Horizontal offset between eyes
        # Positive vergence = converged (near object) = reduced disparity
        # Zero vergence = parallel (far object) = max disparity
        
        disparity = self.baseline_disparity * (1.0 - self.vergence_angle / 10.0)
        disparity = np.clip(disparity, 0.01, self.baseline_disparity)
        
        # Left eye looks slightly right, right eye looks slightly left
        # This creates the baseline horizontal disparity for stereopsis
        # (Applied as offset in frame sampling, not position change)
    
    def _constrain_positions(self):
        """Constrain eye positions to valid range."""
        # Keep within frame bounds with margin
        margin = 0.3  # Allow eyes to look ~30% outside frame
        self.left_eye_pos = np.clip(self.left_eye_pos, -1-margin, 1+margin)
        self.right_eye_pos = np.clip(self.right_eye_pos, -1-margin, 1+margin)
    
    def saccade_to(self, target_x: float, target_y: float, binocular: bool = True):
        """
        Execute a saccade to target location.
        
        Args:
            target_x: Target x coordinate in normalized frame [-1, 1]
            target_y: Target y coordinate in normalized frame [-1, 1]
            binocular: If True, both eyes move together (conjugate)
        """
        target = np.array([target_x, target_y])
        
        # Calculate saccade duration based on amplitude (main sequence)
        amplitude = np.linalg.norm(target - self.left_eye_pos)
        duration = 20.0 + amplitude * 50.0  # ms, roughly linear relationship
        duration = np.clip(duration, 20.0, 100.0)
        
        if binocular:
            # Conjugate saccade (both eyes move together)
            self._initiate_saccade(target, target, duration)
        else:
            # Disjunctive saccade (vergence)
            # For now, just use conjugate
            self._initiate_saccade(target, target, duration)
    
    def set_vergence(self, angle: float):
        """
        Set target vergence angle.
        
        Args:
            angle: Vergence angle in degrees (0 = parallel, + = converged for near)
        """
        self.target_vergence = np.clip(angle, 0.0, 15.0)
    
    def get_eye_positions(self) -> Dict[str, Tuple[float, float]]:
        """Get current eye positions."""
        return {
            'left': tuple(self.left_eye_pos),
            'right': tuple(self.right_eye_pos)
        }
    
    def get_frame_regions(self, frame_width: int, frame_height: int, 
                          fov_size: int = 64) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Get frame regions (bounding boxes) for each eye based on current position.
        
        Args:
            frame_width: Full webcam frame width
            frame_height: Full webcam frame height
            fov_size: Size of the foveal field of view (in pixels)
        
        Returns:
            Dict with 'left' and 'right' keys, each containing (x1, y1, x2, y2)
        """
        regions = {}
        
        for eye_name, eye_pos in [('left', self.left_eye_pos), 
                                    ('right', self.right_eye_pos)]:
            # Convert normalized position [-1, 1] to frame coordinates
            # Center of frame is (0, 0) in normalized, (width/2, height/2) in pixels
            center_x = int((eye_pos[0] + 1.0) / 2.0 * frame_width)
            center_y = int((eye_pos[1] + 1.0) / 2.0 * frame_height)
            
            # Calculate bounding box
            half_fov = fov_size // 2
            x1 = max(0, center_x - half_fov)
            y1 = max(0, center_y - half_fov)
            x2 = min(frame_width, center_x + half_fov)
            y2 = min(frame_height, center_y + half_fov)
            
            regions[eye_name] = (x1, y1, x2, y2)
        
        return regions
    
    def get_state(self) -> Dict:
        """Get full state for serialization/API."""
        return {
            'left_eye': {
                'x': float(self.left_eye_pos[0]),
                'y': float(self.left_eye_pos[1])
            },
            'right_eye': {
                'x': float(self.right_eye_pos[0]),
                'y': float(self.right_eye_pos[1])
            },
            'vergence_angle': float(self.vergence_angle),
            'target_vergence': float(self.target_vergence),
            'is_saccading': self.is_saccading
        }

