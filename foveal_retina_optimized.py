"""
OPTIMIZED foveal retina with vectorized updates and caching.
Expected 5-10x performance improvement over original.
"""

import numpy as np
from typing import List, Dict, Tuple
from temporal_neuron import TemporalPhotoreceptor
from performance_utils import ActivityMapCache, benchmark_function, SpatialIndex


class FovealRetinaOptimized:
    """
    Optimized foveal retina with:
    - Activity map caching
    - Vectorized receptor updates
    - Pre-computed spatial indices
    - Efficient Gaussian smoothing
    """
    
    def __init__(self, grid_size: int = 64, fovea_radius: float = 0.15):
        """
        Initialize foveal retina with optimizations.
        
        Args:
            grid_size: Resolution of the input grid
            fovea_radius: Radius of fovea as fraction of grid (0-0.5)
        """
        self.grid_size = grid_size
        self.fovea_radius = fovea_radius
        
        # Spatial extent: [-1, 1] x [-1, 1]
        self.spatial_extent = 2.0
        
        # Blind spot parameters
        self.blind_spot_offset = {
            'left': np.array([0.18, 0.0]),
            'right': np.array([-0.18, 0.0])
        }
        self.blind_spot_radius = 0.04
        
        # Create photoreceptors
        print("Creating photoreceptors...")
        self.photoreceptors = {
            'left': self._create_eye_receptors('left'),
            'right': self._create_eye_receptors('right')
        }
        
        # OPTIMIZATION: Convert to vectorized arrays
        print("Converting to vectorized arrays...")
        self.vectorized_receptors = self._create_vectorized_arrays()
        
        # OPTIMIZATION: Build spatial indices
        print("Building spatial indices...")
        self.spatial_indices = self._build_spatial_indices()
        
        # OPTIMIZATION: Pre-compute interpolation weights for activity maps
        print("Pre-computing interpolation weights...")
        self.interp_weights = self._precompute_interpolation_weights()
        
        # OPTIMIZATION: Activity map cache
        self.activity_cache = ActivityMapCache()
        
        # Pooling zones
        self.pooling_zones = self._define_pooling_zones()
        
        # Current time
        self.current_time = 0.0
        self.dt = 1.0
        
        # Statistics
        self.total_photoreceptors = sum(
            len(self.vectorized_receptors[eye][rtype]['positions'])
            for eye in ['left', 'right']
            for rtype in ['rods', 'red_cones', 'green_cones', 'blue_cones']
        )
        
        print(f"Created {self.total_photoreceptors} photoreceptors (optimized)")
    
    def _create_eye_receptors(self, eye: str) -> Dict[str, List[TemporalPhotoreceptor]]:
        """Create photoreceptors (same as original, for compatibility)"""
        receptors = {
            'rods': [],
            'red_cones': [],
            'green_cones': [],
            'blue_cones': []
        }
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = (i / (self.grid_size - 1)) * 2.0 - 1.0
                y = (j / (self.grid_size - 1)) * 2.0 - 1.0
                position = (x, y)
                
                eccentricity = np.sqrt(x**2 + y**2)
                
                # Check blind spot
                blind_spot_center = self.blind_spot_offset[eye]
                dist_from_blind_spot = np.sqrt(
                    (x - blind_spot_center[0])**2 + (y - blind_spot_center[1])**2
                )
                in_blind_spot = dist_from_blind_spot < self.blind_spot_radius
                
                if in_blind_spot:
                    continue
                
                in_fovea = eccentricity < self.fovea_radius
                cone_density_factor = np.exp(-eccentricity * 1.2)
                
                # Red and green cones
                if np.random.rand() < cone_density_factor:
                    receptors['red_cones'].append(
                        TemporalPhotoreceptor('red', position, eye, eccentricity)
                    )
                
                if np.random.rand() < cone_density_factor:
                    receptors['green_cones'].append(
                        TemporalPhotoreceptor('green', position, eye, eccentricity)
                    )
                
                # Blue cones
                if np.random.rand() < cone_density_factor * 0.4:
                    receptors['blue_cones'].append(
                        TemporalPhotoreceptor('blue', position, eye, eccentricity)
                    )
                
                # Rods (annular pattern)
                fovea_edge = self.fovea_radius
                periphery_start = fovea_edge * 1.5
                periphery_end = fovea_edge * 3.0
                
                if eccentricity < fovea_edge:
                    if np.random.rand() < 0.05:
                        receptors['rods'].append(
                            TemporalPhotoreceptor('rod', position, eye, eccentricity)
                        )
                elif fovea_edge <= eccentricity < periphery_start:
                    rod_prob = 0.3 + (eccentricity - fovea_edge) / (periphery_start - fovea_edge) * 0.7
                    if np.random.rand() < rod_prob:
                        receptors['rods'].append(
                            TemporalPhotoreceptor('rod', position, eye, eccentricity)
                        )
                elif periphery_start <= eccentricity < periphery_end:
                    rod_prob = 1.0 - (eccentricity - periphery_start) / (periphery_end - periphery_start) * 0.3
                    rod_prob = max(0.7, rod_prob)
                    if np.random.rand() < rod_prob:
                        receptors['rods'].append(
                            TemporalPhotoreceptor('rod', position, eye, eccentricity)
                        )
                else:
                    rod_prob = 0.9 - (eccentricity - periphery_end) * 0.1
                    rod_prob = max(0.5, rod_prob)
                    if np.random.rand() < rod_prob:
                        receptors['rods'].append(
                            TemporalPhotoreceptor('rod', position, eye, eccentricity)
                        )
        
        return receptors
    
    def _create_vectorized_arrays(self) -> Dict:
        """
        OPTIMIZATION: Convert receptor lists to NumPy arrays for vectorized updates.
        This allows updating all receptors of same type in parallel.
        """
        vectorized = {}
        
        for eye in ['left', 'right']:
            vectorized[eye] = {}
            
            for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
                receptors = self.photoreceptors[eye][receptor_type]
                n = len(receptors)
                
                if n == 0:
                    vectorized[eye][receptor_type] = {
                        'positions': np.array([]).reshape(0, 2),
                        'v_membrane': np.array([]),
                        'light_input': np.array([]),
                        'adapted_response': np.array([]),
                        'eccentricity': np.array([]),
                        'receptor_objects': []  # Keep original objects for compatibility
                    }
                    continue
                
                # Extract properties into arrays
                positions = np.array([r.position for r in receptors])
                v_membrane = np.array([r.v_membrane for r in receptors])
                light_input = np.array([r.light_input for r in receptors])
                adapted_response = np.array([r.adapted_response for r in receptors])
                eccentricity = np.array([r.eccentricity for r in receptors])
                
                vectorized[eye][receptor_type] = {
                    'positions': positions,
                    'v_membrane': v_membrane,
                    'light_input': light_input,
                    'adapted_response': adapted_response,
                    'eccentricity': eccentricity,
                    'receptor_objects': receptors  # Keep for compatibility
                }
        
        return vectorized
    
    def _build_spatial_indices(self) -> Dict:
        """OPTIMIZATION: Build spatial indices for fast neighbor lookups"""
        indices = {}
        
        for eye in ['left', 'right']:
            indices[eye] = {}
            for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
                positions = self.vectorized_receptors[eye][receptor_type]['positions']
                if len(positions) > 0:
                    indices[eye][receptor_type] = SpatialIndex(positions, grid_resolution=32)
                else:
                    indices[eye][receptor_type] = None
        
        return indices
    
    def _precompute_interpolation_weights(self) -> Dict:
        """
        OPTIMIZATION: Pre-compute bilinear interpolation weights for activity map generation.
        This avoids recomputing weights every frame.
        """
        interp_weights = {}
        
        for eye in ['left', 'right']:
            interp_weights[eye] = {}
            
            for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
                positions = self.vectorized_receptors[eye][receptor_type]['positions']
                n = len(positions)
                
                if n == 0:
                    interp_weights[eye][receptor_type] = None
                    continue
                
                # For each receptor, compute grid indices and interpolation weights
                # Grid coordinates (floating point)
                # positions are (x, y) where x is horizontal [-1, 1] and y is vertical [-1, 1]
                grid_coords = (positions + 1.0) / 2.0 * (self.grid_size - 1)
                
                # Convert to grid indices
                # x (horizontal) -> column index
                # y (vertical) -> row index
                # For array indexing: array[row, col] = array[y_index, x_index]
                col_coords = grid_coords[:, 0]  # x -> column
                row_coords = grid_coords[:, 1]  # y -> row
                
                # Integer indices
                col0 = np.floor(col_coords).astype(int)  # column index
                col1 = np.minimum(col0 + 1, self.grid_size - 1)
                row0 = np.floor(row_coords).astype(int)  # row index
                row1 = np.minimum(row0 + 1, self.grid_size - 1)
                
                # Fractional parts
                dcol = col_coords - col0
                drow = row_coords - row0
                
                # Bilinear weights
                w00 = (1 - dcol) * (1 - drow)
                w01 = (1 - dcol) * drow
                w10 = dcol * (1 - drow)
                w11 = dcol * drow
                
                interp_weights[eye][receptor_type] = {
                    'row0': row0, 'row1': row1, 'col0': col0, 'col1': col1,
                    'w00': w00, 'w01': w01, 'w10': w10, 'w11': w11
                }
        
        return interp_weights
    
    def _define_pooling_zones(self) -> List[Dict]:
        """Define pooling zones (same as original)"""
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
        """Get pooling parameters for given eccentricity"""
        for zone in self.pooling_zones:
            if zone['ecc_min'] <= eccentricity < zone['ecc_max']:
                return zone
        return self.pooling_zones[-1]
    
    @benchmark_function('process_image')
    def process_image(self, left_image: np.ndarray, right_image: np.ndarray,
                     intensity: float = 1.0):
        """
        OPTIMIZED: Process images through photoreceptors with vectorization.
        """
        self._process_eye_image_optimized(left_image, 'left', intensity)
        self._process_eye_image_optimized(right_image, 'right', intensity)
        
        # Invalidate activity cache
        self.activity_cache.invalidate_all()
    
    def _process_eye_image_optimized(self, image: np.ndarray, eye: str, intensity: float):
        """OPTIMIZED: Vectorized image processing"""
        for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
            vec_data = self.vectorized_receptors[eye][receptor_type]
            positions = vec_data['positions']
            
            if len(positions) == 0:
                continue
            
            # Convert spatial positions to image indices (vectorized)
            # positions are (x, y) where x is horizontal [-1, 1] and y is vertical [-1, 1]
            # For image indexing: image[row, col] = image[y_index, x_index]
            grid_coords = (positions + 1.0) / 2.0 * (self.grid_size - 1)
            x_indices = grid_coords[:, 0].astype(int)  # column (width)
            y_indices = grid_coords[:, 1].astype(int)  # row (height)
            
            # Clip to valid image bounds
            x_indices = np.clip(x_indices, 0, self.grid_size - 1)
            y_indices = np.clip(y_indices, 0, self.grid_size - 1)
            
            # Sample image at all receptor positions (vectorized)
            # Note: image shape is (height, width, channels) = (rows, cols, 3)
            rgb_values = image[y_indices, x_indices]
            
            # Compute spectral responses (vectorized)
            if receptor_type == 'rods':
                response = (0.299 * rgb_values[:, 0] + 
                           0.587 * rgb_values[:, 1] + 
                           0.114 * rgb_values[:, 2]) * intensity
            elif receptor_type == 'red_cones':
                response = (0.8 * rgb_values[:, 0] + 0.3 * rgb_values[:, 1]) * intensity
            elif receptor_type == 'green_cones':
                response = (0.6 * rgb_values[:, 1] + 0.2 * rgb_values[:, 0]) * intensity
            else:  # blue_cones
                response = 0.9 * rgb_values[:, 2] * intensity
            
            vec_data['light_input'] = response
    
    @benchmark_function('update')
    def update(self, dt: float = 1.0):
        """
        OPTIMIZED: Vectorized update for all photoreceptors.
        Updates all receptors of same type in parallel.
        """
        self.current_time += dt
        
        # Constants
        tau_adaptation = 100.0
        tau_membrane = 20.0
        v_dark = -40.0
        v_light = -70.0
        
        for eye in ['left', 'right']:
            for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
                vec_data = self.vectorized_receptors[eye][receptor_type]
                
                if len(vec_data['positions']) == 0:
                    continue
                
                # VECTORIZED adaptation update (all receptors at once)
                adaptation_rate = dt / tau_adaptation
                vec_data['adapted_response'] += (
                    (vec_data['light_input'] - vec_data['adapted_response']) 
                    * adaptation_rate
                )
                
                # VECTORIZED membrane update
                target_v = v_dark + (v_light - v_dark) * vec_data['adapted_response']
                vec_data['v_membrane'] += (
                    (target_v - vec_data['v_membrane']) * (dt / tau_membrane)
                )
        
        # Invalidate activity cache
        self.activity_cache.invalidate_all()
    
    def get_receptor_positions(self, eye: str, receptor_type: str = None) -> np.ndarray:
        """Get positions of receptors (optimized with cached arrays)"""
        if receptor_type:
            return self.vectorized_receptors[eye][receptor_type]['positions']
        else:
            # Concatenate all types
            positions = []
            for rtype in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
                pos = self.vectorized_receptors[eye][rtype]['positions']
                if len(pos) > 0:
                    positions.append(pos)
            
            if len(positions) > 0:
                return np.vstack(positions)
            else:
                return np.array([]).reshape(0, 2)
    
    @benchmark_function('get_activity_map')
    def get_activity_map(self, eye: str, receptor_type: str) -> np.ndarray:
        """
        OPTIMIZED: Get activity map with caching and pre-computed interpolation.
        """
        cache_key = f"{eye}_{receptor_type}"
        
        # Check cache
        cached = self.activity_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Compute activity map
        activity_map = self._compute_activity_map_optimized(eye, receptor_type)
        
        # Cache it
        self.activity_cache.set(cache_key, activity_map)
        
        return activity_map
    
    def _compute_activity_map_optimized(self, eye: str, receptor_type: str) -> np.ndarray:
        """OPTIMIZED: Use pre-computed interpolation weights"""
        activity_map = np.zeros((self.grid_size, self.grid_size))
        count_map = np.zeros((self.grid_size, self.grid_size))
        
        vec_data = self.vectorized_receptors[eye][receptor_type]
        interp = self.interp_weights[eye][receptor_type]
        
        if interp is None or len(vec_data['positions']) == 0:
            return activity_map
        
        # Get activations (vectorized)
        activations = vec_data['adapted_response']
        
        # Distribute to grid using pre-computed weights (vectorized)
        # Note: activity_map[row, col] = activity_map[y_index, x_index]
        np.add.at(activity_map, (interp['row0'], interp['col0']), activations * interp['w00'])
        np.add.at(activity_map, (interp['row0'], interp['col1']), activations * interp['w01'])
        np.add.at(activity_map, (interp['row1'], interp['col0']), activations * interp['w10'])
        np.add.at(activity_map, (interp['row1'], interp['col1']), activations * interp['w11'])
        
        np.add.at(count_map, (interp['row0'], interp['col0']), interp['w00'])
        np.add.at(count_map, (interp['row0'], interp['col1']), interp['w01'])
        np.add.at(count_map, (interp['row1'], interp['col0']), interp['w10'])
        np.add.at(count_map, (interp['row1'], interp['col1']), interp['w11'])
        
        # Normalize
        mask = count_map > 1e-6
        activity_map[mask] /= count_map[mask]
        
        # Light smoothing (cached Gaussian kernel)
        if not hasattr(self, '_gaussian_kernel'):
            from scipy.ndimage import gaussian_filter
            self._gaussian_filter = lambda x: gaussian_filter(x, sigma=0.8, mode='constant', cval=0.0)
        
        smoothed = self._gaussian_filter(activity_map)
        
        mask_zeros = activity_map == 0
        mask_nearby = self._gaussian_filter(count_map) > 0.1
        fill_mask = mask_zeros & mask_nearby
        activity_map[fill_mask] = smoothed[fill_mask]
        
        return activity_map
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {
            'type': 'foveal_retina_optimized',
            'grid_size': self.grid_size,
            'fovea_radius': self.fovea_radius,
            'total_photoreceptors': self.total_photoreceptors,
            'current_time': self.current_time,
            'pooling_zones': self.pooling_zones,
            'cache_stats': self.activity_cache.get_stats()
        }
        
        for eye in ['left', 'right']:
            eye_summary = {}
            for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
                vec_data = self.vectorized_receptors[eye][receptor_type]
                
                if len(vec_data['positions']) == 0:
                    eye_summary[receptor_type] = {
                        'total_count': 0,
                        'mean_activity': 0,
                        'zone_distribution': {}
                    }
                    continue
                
                # Count by zone (vectorized)
                zone_counts = {zone['name']: 0 for zone in self.pooling_zones}
                eccentricities = vec_data['eccentricity']
                
                for zone in self.pooling_zones:
                    in_zone = (eccentricities >= zone['ecc_min']) & (eccentricities < zone['ecc_max'])
                    zone_counts[zone['name']] = int(np.sum(in_zone))
                
                eye_summary[receptor_type] = {
                    'total_count': len(vec_data['positions']),
                    'mean_activity': float(np.mean(vec_data['adapted_response'])),
                    'zone_distribution': zone_counts
                }
            
            summary[eye] = eye_summary
        
        return summary
    
    def get_state(self) -> Dict:
        """Get complete state"""
        # Return original objects for compatibility
        return {
            'summary': self.get_summary(),
            'photoreceptors': {
                eye: {
                    receptor_type: [r.get_state() for r in self.vectorized_receptors[eye][receptor_type]['receptor_objects']]
                    for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']
                }
                for eye in ['left', 'right']
            }
        }

