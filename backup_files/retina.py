"""
Retinal layer implementation with photoreceptors.
Simulates the first stage of visual processing.
"""

import numpy as np
from typing import List, Tuple, Dict
from neuron import Photoreceptor


class RetinaLayer:
    """
    Retinal photoreceptor layer with binocular vision.
    Contains rods and three types of cones (RGB) for each eye.
    """
    
    def __init__(self, grid_size: int = 32):
        """
        Initialize retinal layer with two eyes.
        
        Args:
            grid_size: Size of the n×n photoreceptor grid for each eye
        """
        self.grid_size = grid_size
        self.photoreceptors = {
            'left': self._create_eye_photoreceptors('left'),
            'right': self._create_eye_photoreceptors('right')
        }
        
        # Statistics
        self.total_photoreceptors = sum(
            len(receptors) for eye_receptors in self.photoreceptors.values() 
            for receptors in eye_receptors.values()
        )
        
    def _create_eye_photoreceptors(self, eye: str) -> Dict[str, List[Photoreceptor]]:
        """Create all photoreceptors for one eye."""
        receptors = {
            'rods': [],
            'red_cones': [],
            'green_cones': [],
            'blue_cones': []
        }
        
        # Create photoreceptor grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                position = (x, y)
                
                # Create all four types of photoreceptors at each position
                receptors['rods'].append(
                    Photoreceptor('rod', position, eye)
                )
                receptors['red_cones'].append(
                    Photoreceptor('red', position, eye)
                )
                receptors['green_cones'].append(
                    Photoreceptor('green', position, eye)
                )
                receptors['blue_cones'].append(
                    Photoreceptor('blue', position, eye)
                )
        
        return receptors
    
    def process_image(self, left_image: np.ndarray, right_image: np.ndarray, 
                     intensity: float = 1.0):
        """
        Process input images through photoreceptors.
        
        Args:
            left_image: RGB image for left eye (grid_size, grid_size, 3)
            right_image: RGB image for right eye (grid_size, grid_size, 3)
            intensity: Overall light intensity (0.0-1.0)
        """
        # Ensure images are the right size
        assert left_image.shape == (self.grid_size, self.grid_size, 3), \
            f"Left image must be ({self.grid_size}, {self.grid_size}, 3)"
        assert right_image.shape == (self.grid_size, self.grid_size, 3), \
            f"Right image must be ({self.grid_size}, {self.grid_size}, 3)"
        
        # Process each eye
        self._process_eye_image(left_image, 'left', intensity)
        self._process_eye_image(right_image, 'right', intensity)
    
    def _process_eye_image(self, image: np.ndarray, eye: str, intensity: float):
        """Process image for one eye."""
        receptors = self.photoreceptors[eye]
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                idx = x * self.grid_size + y
                rgb_value = tuple(image[x, y])
                
                # Stimulate each type of photoreceptor
                receptors['rods'][idx].set_light_input(rgb_value, intensity)
                receptors['red_cones'][idx].set_light_input(rgb_value, intensity)
                receptors['green_cones'][idx].set_light_input(rgb_value, intensity)
                receptors['blue_cones'][idx].set_light_input(rgb_value, intensity)
    
    def get_activity_maps(self, eye: str) -> Dict[str, np.ndarray]:
        """
        Get activity maps for each photoreceptor type in one eye.
        
        Returns:
            Dictionary with activity maps for rods, red_cones, green_cones, blue_cones
        """
        receptors = self.photoreceptors[eye]
        maps = {}
        
        for receptor_type, receptor_list in receptors.items():
            activity_map = np.zeros((self.grid_size, self.grid_size))
            for i, receptor in enumerate(receptor_list):
                x, y = receptor.position
                activity_map[x, y] = receptor.activity
            maps[receptor_type] = activity_map
        
        return maps
    
    def get_combined_output(self, eye: str) -> np.ndarray:
        """
        Get combined RGB output from cone photoreceptors.
        
        Returns:
            RGB image reconstructed from cone responses
        """
        receptors = self.photoreceptors[eye]
        output = np.zeros((self.grid_size, self.grid_size, 3))
        
        for i in range(len(receptors['red_cones'])):
            x, y = receptors['red_cones'][i].position
            output[x, y, 0] = receptors['red_cones'][i].activity
            output[x, y, 1] = receptors['green_cones'][i].activity
            output[x, y, 2] = receptors['blue_cones'][i].activity
        
        return output
    
    def get_state(self) -> Dict:
        """Get complete state of retinal layer for visualization."""
        return {
            'type': 'retina',
            'grid_size': self.grid_size,
            'total_photoreceptors': self.total_photoreceptors,
            'photoreceptors_per_eye': self.grid_size * self.grid_size * 4,
            'eyes': {
                eye: {
                    receptor_type: [r.get_state() for r in receptors]
                    for receptor_type, receptors in eye_receptors.items()
                }
                for eye, eye_receptors in self.photoreceptors.items()
            }
        }
    
    def get_summary(self) -> Dict:
        """Get summary statistics without full photoreceptor data."""
        left_maps = self.get_activity_maps('left')
        right_maps = self.get_activity_maps('right')
        
        return {
            'type': 'retina',
            'grid_size': self.grid_size,
            'total_photoreceptors': self.total_photoreceptors,
            'left_eye': {
                receptor_type: {
                    'mean_activity': float(np.mean(activity_map)),
                    'max_activity': float(np.max(activity_map)),
                    'active_receptors': int(np.sum(activity_map > 0.1))
                }
                for receptor_type, activity_map in left_maps.items()
            },
            'right_eye': {
                receptor_type: {
                    'mean_activity': float(np.mean(activity_map)),
                    'max_activity': float(np.max(activity_map)),
                    'active_receptors': int(np.sum(activity_map > 0.1))
                }
                for receptor_type, activity_map in right_maps.items()
            }
        }

