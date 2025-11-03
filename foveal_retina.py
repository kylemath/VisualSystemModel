"""
Foveal retina with realistic eccentricity-based receptor distribution.
Higher cone density in fovea, more rods in periphery.
"""

import numpy as np
from typing import List, Dict, Tuple
from temporal_neuron import TemporalPhotoreceptor


class FovealRetina:
    """
    Retina with foveal structure and eccentricity-based receptor distribution.
    
    Features:
    - Fovea (center): High cone density, few rods, fine detail
    - Periphery: More rods, sparse cones, coarse detail
    - Continuous spatial coordinates
    - Eccentricity-based pooling ratios (for downstream bipolar cells)
    """
    
    def __init__(self, grid_size: int = 64, fovea_radius: float = 0.15):
        """
        Initialize foveal retina.
        
        Args:
            grid_size: Resolution of the input grid
            fovea_radius: Radius of fovea as fraction of grid (0-0.5)
        """
        self.grid_size = grid_size
        self.fovea_radius = fovea_radius
        
        # Spatial extent: [-1, 1] x [-1, 1]
        # Center (0, 0) is fovea
        self.spatial_extent = 2.0
        
        # Blind spot parameters (optic disc where optic nerve exits)
        # Located temporal (nasal side) to fovea - anatomically correct
        # Left eye: blind spot is temporal (right/east side in visual field)
        # Right eye: blind spot is temporal (left/west side in visual field)
        self.blind_spot_offset = {
            'left': np.array([0.18, 0.0]),   # East (right side) - temporal for left eye
            'right': np.array([-0.18, 0.0])  # West (left side) - temporal for right eye
        }
        self.blind_spot_radius = 0.04  # About 5 degrees visual angle
        
        # Create photoreceptors for each eye
        self.photoreceptors = {
            'left': self._create_eye_receptors('left'),
            'right': self._create_eye_receptors('right')
        }
        
        # Pooling zones for bipolar cells (eccentricity-based)
        self.pooling_zones = self._define_pooling_zones()
        
        # Current time
        self.current_time = 0.0
        self.dt = 1.0  # ms per update
        
        # Statistics
        self.total_photoreceptors = sum(
            len(receptors) for eye_data in self.photoreceptors.values()
            for receptors in eye_data.values()
        )
        
    def _create_eye_receptors(self, eye: str) -> Dict[str, List[TemporalPhotoreceptor]]:
        """Create photoreceptors with foveal distribution for one eye."""
        receptors = {
            'rods': [],
            'red_cones': [],
            'green_cones': [],
            'blue_cones': []
        }
        
        # Create receptors on a regular grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Convert to spatial coordinates [-1, 1]
                x = (i / (self.grid_size - 1)) * 2.0 - 1.0
                y = (j / (self.grid_size - 1)) * 2.0 - 1.0
                position = (x, y)
                
                # Calculate eccentricity (distance from center)
                eccentricity = np.sqrt(x**2 + y**2)
                
                # Check if in blind spot (optic disc)
                blind_spot_center = self.blind_spot_offset[eye]
                dist_from_blind_spot = np.sqrt(
                    (x - blind_spot_center[0])**2 + (y - blind_spot_center[1])**2
                )
                in_blind_spot = dist_from_blind_spot < self.blind_spot_radius
                
                # Skip creating receptors in blind spot
                if in_blind_spot:
                    continue
                
                # Determine receptor densities based on eccentricity
                in_fovea = eccentricity < self.fovea_radius
                
                # Cone density: gradual exponential falloff from fovea
                # More gradual than before - uses smoother decay function
                cone_density_factor = np.exp(-eccentricity * 1.2)  # Gradual falloff
                
                # Red and green cones: high in fovea, gradual falloff
                if np.random.rand() < cone_density_factor:
                    receptors['red_cones'].append(
                        TemporalPhotoreceptor('red', position, eye, eccentricity)
                    )
                
                if np.random.rand() < cone_density_factor:
                    receptors['green_cones'].append(
                        TemporalPhotoreceptor('green', position, eye, eccentricity)
                    )
                
                # Blue cones: sparse everywhere, but still follow gradient
                if np.random.rand() < cone_density_factor * 0.4:
                    receptors['blue_cones'].append(
                        TemporalPhotoreceptor('blue', position, eye, eccentricity)
                    )
                
                # Rod density: Annular pattern with peak where cones start to drop
                # Highest density in annulus where fovea transitions to periphery
                # This creates a "rods fill the gaps" pattern
                fovea_edge = self.fovea_radius
                periphery_start = fovea_edge * 1.5  # Start of high rod density
                periphery_end = fovea_edge * 3.0    # End of high rod density
                
                if eccentricity < fovea_edge:
                    # Inside fovea: very few rods
                    if np.random.rand() < 0.05:
                        receptors['rods'].append(
                            TemporalPhotoreceptor('rod', position, eye, eccentricity)
                        )
                elif fovea_edge <= eccentricity < periphery_start:
                    # Transition zone: rods start appearing
                    rod_prob = 0.3 + (eccentricity - fovea_edge) / (periphery_start - fovea_edge) * 0.7
                    if np.random.rand() < rod_prob:
                        receptors['rods'].append(
                            TemporalPhotoreceptor('rod', position, eye, eccentricity)
                        )
                elif periphery_start <= eccentricity < periphery_end:
                    # Peak rod density zone (annulus)
                    rod_prob = 1.0 - (eccentricity - periphery_start) / (periphery_end - periphery_start) * 0.3
                    rod_prob = max(0.7, rod_prob)  # Maintain high density
                    if np.random.rand() < rod_prob:
                        receptors['rods'].append(
                            TemporalPhotoreceptor('rod', position, eye, eccentricity)
                        )
                else:
                    # Far periphery: rods still present but slightly decreasing
                    rod_prob = 0.9 - (eccentricity - periphery_end) * 0.1
                    rod_prob = max(0.5, rod_prob)  # Never go below 50%
                    if np.random.rand() < rod_prob:
                        receptors['rods'].append(
                            TemporalPhotoreceptor('rod', position, eye, eccentricity)
                        )
        
        return receptors
    
    def _define_pooling_zones(self) -> List[Dict]:
        """
        Define pooling zones for bipolar cells.
        Smaller receptive fields in fovea, larger in periphery.
        """
        zones = []
        
        # Create concentric zones from fovea to periphery
        # Zone 0: Fovea (0.0 - 0.15) - 1:1 or small pooling for cones
        # Zone 1: Parafovea (0.15 - 0.4) - moderate pooling
        # Zone 2: Near periphery (0.4 - 0.8) - larger pooling  
        # Zone 3: Far periphery (0.8 - 1.414) - largest pooling
        
        zone_definitions = [
            {'name': 'fovea', 'ecc_min': 0.0, 'ecc_max': 0.15, 
             'cone_pool_size': 1, 'rod_pool_size': 3, 'pathway': 'P'},
            {'name': 'parafovea', 'ecc_min': 0.15, 'ecc_max': 0.4,
             'cone_pool_size': 2, 'rod_pool_size': 5, 'pathway': 'P'},
            {'name': 'near_periphery', 'ecc_min': 0.4, 'ecc_max': 0.8,
             'cone_pool_size': 3, 'rod_pool_size': 7, 'pathway': 'M'},
            {'name': 'far_periphery', 'ecc_min': 0.8, 'ecc_max': 1.5,
             'cone_pool_size': 5, 'rod_pool_size': 11, 'pathway': 'M'}
        ]
        
        return zone_definitions
    
    def get_pooling_params(self, eccentricity: float) -> Dict:
        """Get pooling parameters for given eccentricity."""
        for zone in self.pooling_zones:
            if zone['ecc_min'] <= eccentricity < zone['ecc_max']:
                return zone
        # Default to far periphery
        return self.pooling_zones[-1]
    
    def process_image(self, left_image: np.ndarray, right_image: np.ndarray,
                     intensity: float = 1.0):
        """
        Process images through photoreceptors.
        Sets light input for all receptors.
        """
        self._process_eye_image(left_image, 'left', intensity)
        self._process_eye_image(right_image, 'right', intensity)
    
    def _process_eye_image(self, image: np.ndarray, eye: str, intensity: float):
        """Process image for one eye."""
        receptors = self.photoreceptors[eye]
        
        # For each receptor, sample the image at its position
        for receptor_type, receptor_list in receptors.items():
            for receptor in receptor_list:
                x, y = receptor.position
                
                # Convert spatial position to image indices
                i = int((x + 1.0) / 2.0 * (self.grid_size - 1))
                j = int((y + 1.0) / 2.0 * (self.grid_size - 1))
                
                # Ensure in bounds
                i = np.clip(i, 0, image.shape[0] - 1)
                j = np.clip(j, 0, image.shape[1] - 1)
                
                rgb_value = tuple(image[i, j])
                receptor.set_light_input(rgb_value, intensity)
    
    def update(self, dt: float = 1.0):
        """
        Update all photoreceptors (temporal dynamics).
        
        Args:
            dt: Time step in ms
        """
        self.current_time += dt
        
        for eye_data in self.photoreceptors.values():
            for receptor_list in eye_data.values():
                for receptor in receptor_list:
                    receptor.update(dt, self.current_time)
    
    def get_receptor_positions(self, eye: str, receptor_type: str = None) -> np.ndarray:
        """Get positions of receptors for visualization."""
        positions = []
        
        if receptor_type:
            receptor_lists = [self.photoreceptors[eye][receptor_type]]
        else:
            receptor_lists = self.photoreceptors[eye].values()
        
        for receptor_list in receptor_lists:
            for receptor in receptor_list:
                positions.append(receptor.position)
        
        return np.array(positions)
    
    def get_activity_map(self, eye: str, receptor_type: str) -> np.ndarray:
        """
        Get activity map for specific receptor type.
        Use smooth interpolation to avoid grid artifacts from sparse receptor distribution.
        """
        activity_map = np.zeros((self.grid_size, self.grid_size))
        count_map = np.zeros((self.grid_size, self.grid_size))
        
        receptors = self.photoreceptors[eye][receptor_type]
        
        if len(receptors) == 0:
            return activity_map
        
        # First pass: accumulate activations with bilinear weighting
        # This prevents discrete binning artifacts
        for receptor in receptors:
            x, y = receptor.position
            activation = receptor.get_activation()
            
            # Convert to grid coordinates (floating point)
            grid_i = (x + 1.0) / 2.0 * (self.grid_size - 1)
            grid_j = (y + 1.0) / 2.0 * (self.grid_size - 1)
            
            # Get integer indices
            i0 = int(np.floor(grid_i))
            i1 = min(i0 + 1, self.grid_size - 1)
            j0 = int(np.floor(grid_j))
            j1 = min(j0 + 1, self.grid_size - 1)
            
            # Bilinear interpolation weights
            di = grid_i - i0
            dj = grid_j - j0
            
            # Distribute activation to four nearest grid points
            w00 = (1 - di) * (1 - dj)
            w01 = (1 - di) * dj
            w10 = di * (1 - dj)
            w11 = di * dj
            
            # Add weighted contributions
            activity_map[i0, j0] += activation * w00
            activity_map[i0, j1] += activation * w01
            activity_map[i1, j0] += activation * w10
            activity_map[i1, j1] += activation * w11
            
            count_map[i0, j0] += w00
            count_map[i0, j1] += w01
            count_map[i1, j0] += w10
            count_map[i1, j1] += w11
        
        # Normalize by weights to get proper averages
        mask = count_map > 1e-6  # Avoid division by very small numbers
        activity_map[mask] /= count_map[mask]
        
        # Fill remaining zeros with light smoothing to remove grid artifacts
        from scipy.ndimage import gaussian_filter
        # Only smooth areas with data, preserve zeros elsewhere
        smoothed = gaussian_filter(activity_map, sigma=0.8, mode='constant', cval=0.0)
        # Only use smoothed values where original was zero but nearby values exist
        mask_zeros = activity_map == 0
        mask_nearby = gaussian_filter(count_map, sigma=1.0) > 0.1
        fill_mask = mask_zeros & mask_nearby
        activity_map[fill_mask] = smoothed[fill_mask]
        
        return activity_map
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {
            'type': 'foveal_retina',
            'grid_size': self.grid_size,
            'fovea_radius': self.fovea_radius,
            'total_photoreceptors': self.total_photoreceptors,
            'current_time': self.current_time,
            'pooling_zones': self.pooling_zones
        }
        
        # Count receptors by type and zone
        for eye in ['left', 'right']:
            eye_summary = {}
            for receptor_type, receptors in self.photoreceptors[eye].items():
                # Count by zone
                zone_counts = {zone['name']: 0 for zone in self.pooling_zones}
                total_activity = 0.0
                
                for receptor in receptors:
                    zone_params = self.get_pooling_params(receptor.eccentricity)
                    zone_counts[zone_params['name']] += 1
                    total_activity += receptor.get_activation()
                
                eye_summary[receptor_type] = {
                    'total_count': len(receptors),
                    'mean_activity': total_activity / len(receptors) if len(receptors) > 0 else 0,
                    'zone_distribution': zone_counts
                }
            
            summary[eye] = eye_summary
        
        return summary
    
    def get_state(self) -> Dict:
        """Get complete state for detailed visualization."""
        return {
            'summary': self.get_summary(),
            'photoreceptors': {
                eye: {
                    receptor_type: [r.get_state() for r in receptors]
                    for receptor_type, receptors in eye_data.items()
                }
                for eye, eye_data in self.photoreceptors.items()
            }
        }

