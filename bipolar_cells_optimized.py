"""
OPTIMIZED bipolar cell layer with vectorized updates and sparse connectivity.
Expected 10-50x performance improvement over original.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from temporal_neuron import TemporalNeuron, TemporalPhotoreceptor
from bipolar_cells import BipolarCell  # Keep for compatibility
from performance_utils import (ActivityMapCache, benchmark_function, 
                               SparseConnectivityMatrix, SpatialIndex)

# Try to use optimized retina, fall back to original
try:
    from foveal_retina_optimized import FovealRetinaOptimized as FovealRetina
except ImportError:
    from foveal_retina import FovealRetina


class BipolarLayerOptimized:
    """
    OPTIMIZED bipolar cell layer with:
    - Vectorized neuron updates
    - Sparse connectivity matrices for receptive fields
    - Activity map caching
    - Pre-computed spatial indices
    """
    
    def __init__(self, retina: FovealRetina, receptor_type: str = 'all'):
        """
        Create optimized bipolar cell layer.
        
        Args:
            retina: FovealRetina instance (original or optimized)
            receptor_type: Which receptors to connect ('rods', 'cones', 'all')
        """
        print("Creating OPTIMIZED bipolar layer...")
        
        self.retina = retina
        self.receptor_type = receptor_type
        
        # Bipolar cells for each eye (keep original objects for compatibility)
        self.bipolar_cells = {
            'left': {'ON': [], 'OFF': []},
            'right': {'ON': [], 'OFF': []}
        }
        
        # OPTIMIZATION: Vectorized arrays
        self.vectorized_bipolars = {}
        
        # OPTIMIZATION: Sparse connectivity matrices
        self.connectivity_matrices = {}
        
        # OPTIMIZATION: Activity map cache
        self.activity_cache = ActivityMapCache()
        
        # Create bipolar cells
        self._create_bipolar_cells()
        
        # OPTIMIZATION: Convert to vectorized representation
        print("Converting to vectorized arrays...")
        self._create_vectorized_arrays()
        
        # OPTIMIZATION: Build sparse connectivity matrices
        print("Building sparse connectivity matrices...")
        self._build_connectivity_matrices()
        
        # Statistics
        self.total_bipolar_cells = sum(
            len(self.vectorized_bipolars[eye][ctype]['positions'])
            for eye in ['left', 'right']
            for ctype in ['ON', 'OFF']
        )
        
        print(f"Created {self.total_bipolar_cells} bipolar cells (optimized)")
    
    def _create_bipolar_cells(self):
        """Create bipolar cells (same as original for compatibility)"""
        for eye in ['left', 'right']:
            self._create_eye_bipolar_cells(eye)
    
    def _create_eye_bipolar_cells(self, eye: str):
        """Create bipolar cells for one eye"""
        # Get all photoreceptors
        if self.receptor_type == 'rods':
            receptor_types = ['rods']
        elif self.receptor_type == 'cones':
            receptor_types = ['red_cones', 'green_cones', 'blue_cones']
        else:
            receptor_types = ['rods', 'red_cones', 'green_cones', 'blue_cones']
        
        # Collect all receptors
        all_receptors = []
        if hasattr(self.retina, 'vectorized_receptors'):
            # Using optimized retina
            for rtype in receptor_types:
                receptor_objs = self.retina.vectorized_receptors[eye][rtype]['receptor_objects']
                all_receptors.extend(receptor_objs)
        else:
            # Using original retina
            for rtype in receptor_types:
                all_receptors.extend(self.retina.photoreceptors[eye][rtype])
        
        # Generate bipolar positions
        bipolar_positions = self._generate_bipolar_positions()
        
        for position in bipolar_positions:
            x, y = position
            eccentricity = np.sqrt(x**2 + y**2)
            
            # Get pooling parameters
            pooling_params = self.retina.get_pooling_params(eccentricity)
            pathway = pooling_params['pathway']
            
            # Determine pooling radius
            if self.receptor_type == 'rods':
                pool_radius = pooling_params['rod_pool_size'] * 0.05
            elif self.receptor_type == 'cones':
                pool_radius = pooling_params['cone_pool_size'] * 0.03
            else:
                pool_radius = pooling_params['cone_pool_size'] * 0.04
            
            # Create ON and OFF cells
            on_cell = BipolarCell('ON', position, eye, eccentricity, pathway)
            self._connect_receptive_field(on_cell, all_receptors, pool_radius)
            
            if len(on_cell.center_photoreceptors) > 0:
                self.bipolar_cells[eye]['ON'].append(on_cell)
            
            off_cell = BipolarCell('OFF', position, eye, eccentricity, pathway)
            self._connect_receptive_field(off_cell, all_receptors, pool_radius)
            
            if len(off_cell.center_photoreceptors) > 0:
                self.bipolar_cells[eye]['OFF'].append(off_cell)
    
    def _generate_bipolar_positions(self) -> List[Tuple[float, float]]:
        """Generate positions for bipolar cells (denser than original)"""
        positions = []
        
        # Fovea: dense sampling
        fovea_grid = 24
        for i in range(fovea_grid):
            for j in range(fovea_grid):
                x = (i / (fovea_grid - 1)) * 2 * self.retina.fovea_radius - self.retina.fovea_radius
                y = (j / (fovea_grid - 1)) * 2 * self.retina.fovea_radius - self.retina.fovea_radius
                positions.append((x, y))
        
        # Periphery: denser sampling
        periphery_grid = 20
        for i in range(periphery_grid):
            for j in range(periphery_grid):
                x = (i / (periphery_grid - 1)) * 2.0 - 1.0
                y = (j / (periphery_grid - 1)) * 2.0 - 1.0
                
                eccentricity = np.sqrt(x**2 + y**2)
                
                if eccentricity > self.retina.fovea_radius:
                    positions.append((x, y))
        
        return positions
    
    def _connect_receptive_field(self, bipolar: BipolarCell,
                                 receptors: List[TemporalPhotoreceptor],
                                 pool_radius: float):
        """Connect photoreceptors to bipolar cell's receptive field"""
        bx, by = bipolar.position
        
        center_radius = pool_radius
        surround_radius = pool_radius * 2.0
        
        for receptor in receptors:
            rx, ry = receptor.position
            distance = np.sqrt((bx - rx)**2 + (by - ry)**2)
            
            if distance < center_radius:
                bipolar.add_center_receptor(receptor)
            elif distance < surround_radius:
                bipolar.add_surround_receptor(receptor)
    
    def _create_vectorized_arrays(self):
        """OPTIMIZATION: Convert bipolar cells to vectorized arrays"""
        for eye in ['left', 'right']:
            self.vectorized_bipolars[eye] = {}
            
            for cell_type in ['ON', 'OFF']:
                cells = self.bipolar_cells[eye][cell_type]
                n = len(cells)
                
                if n == 0:
                    self.vectorized_bipolars[eye][cell_type] = {
                        'positions': np.array([]).reshape(0, 2),
                        'v_membrane': np.array([]),
                        'v_rest': np.array([]),
                        'eccentricity': np.array([]),
                        'pathway': np.array([]),
                        'polarity': np.array([]),
                        'cell_objects': []
                    }
                    continue
                
                # Extract properties
                positions = np.array([c.position for c in cells])
                v_membrane = np.array([c.v_membrane for c in cells])
                v_rest = np.array([c.v_rest for c in cells])
                eccentricity = np.array([c.eccentricity for c in cells])
                pathway = np.array([c.pathway for c in cells])
                polarity = np.array([c.polarity for c in cells])
                
                self.vectorized_bipolars[eye][cell_type] = {
                    'positions': positions,
                    'v_membrane': v_membrane,
                    'v_rest': v_rest,
                    'eccentricity': eccentricity,
                    'pathway': pathway,
                    'polarity': polarity,
                    'cell_objects': cells
                }
    
    def _build_connectivity_matrices(self):
        """OPTIMIZATION: Build sparse connectivity matrices for receptive fields"""
        for eye in ['left', 'right']:
            self.connectivity_matrices[eye] = {}
            
            for cell_type in ['ON', 'OFF']:
                cells = self.bipolar_cells[eye][cell_type]
                
                if len(cells) == 0:
                    self.connectivity_matrices[eye][cell_type] = None
                    continue
                
                # Get all unique receptors
                all_receptors = set()
                for cell in cells:
                    all_receptors.update(cell.center_photoreceptors)
                    all_receptors.update(cell.surround_photoreceptors)
                
                all_receptors = list(all_receptors)
                receptor_to_idx = {id(r): i for i, r in enumerate(all_receptors)}
                
                # Build center and surround connectivity matrices
                num_bipolar = len(cells)
                num_receptors = len(all_receptors)
                
                center_matrix = SparseConnectivityMatrix(num_bipolar, num_receptors)
                surround_matrix = SparseConnectivityMatrix(num_bipolar, num_receptors)
                
                for bipolar_idx, cell in enumerate(cells):
                    # Center connections (positive weight)
                    for receptor in cell.center_photoreceptors:
                        receptor_idx = receptor_to_idx[id(receptor)]
                        center_matrix.add_connection(bipolar_idx, receptor_idx, 1.0)
                    
                    # Surround connections (negative weight)
                    for receptor in cell.surround_photoreceptors:
                        receptor_idx = receptor_to_idx[id(receptor)]
                        surround_matrix.add_connection(bipolar_idx, receptor_idx, -0.6)
                
                # Finalize matrices (convert to CSR for fast multiplication)
                center_matrix.finalize()
                surround_matrix.finalize()
                
                self.connectivity_matrices[eye][cell_type] = {
                    'center': center_matrix,
                    'surround': surround_matrix,
                    'receptors': all_receptors
                }
    
    @benchmark_function('bipolar_update')
    def update(self, dt: float = 1.0):
        """
        OPTIMIZED: Vectorized update using sparse matrix multiplication.
        """
        tau_membrane = 20.0
        
        for eye in ['left', 'right']:
            for cell_type in ['ON', 'OFF']:
                vec_data = self.vectorized_bipolars[eye][cell_type]
                conn_data = self.connectivity_matrices[eye][cell_type]
                
                if conn_data is None or len(vec_data['positions']) == 0:
                    continue
                
                # Get receptor activations (from vectorized retina if available)
                receptor_activations = self._get_receptor_activations(eye, conn_data['receptors'])
                
                # VECTORIZED receptive field computation using sparse matrices
                center_input = conn_data['center'].compute_outputs(receptor_activations)
                surround_input = conn_data['surround'].compute_outputs(receptor_activations)
                
                # Center-surround with polarity
                rf_input = vec_data['polarity'] * (center_input + surround_input) * 20.0
                
                # VECTORIZED membrane integration
                target_v = vec_data['v_rest'] + rf_input
                dv = ((target_v - vec_data['v_membrane']) / tau_membrane) * dt
                vec_data['v_membrane'] += dv
        
        # Invalidate activity cache
        self.activity_cache.invalidate_all()
    
    def _get_receptor_activations(self, eye: str, receptors: List) -> np.ndarray:
        """
        Get receptor activations, using vectorized arrays if available.
        This fixes the bug where optimized retina updates vectors but not objects.
        """
        # Check if retina has vectorized receptors (optimized)
        if hasattr(self.retina, 'vectorized_receptors'):
            # Map receptor_type from object format to vectorized array format
            # Object: 'rod', 'red', 'green', 'blue'
            # Array:  'rods', 'red_cones', 'green_cones', 'blue_cones'
            type_map = {
                'rod': 'rods',
                'red': 'red_cones',
                'green': 'green_cones',
                'blue': 'blue_cones'
            }
            
            # Build activation map from vectorized arrays
            activations = []
            for r in receptors:
                # Find which receptor type and index this is
                receptor_type_obj = r.receptor_type  # e.g., 'rod'
                receptor_type_vec = type_map.get(receptor_type_obj, receptor_type_obj)  # e.g., 'rods'
                
                try:
                    vec_data = self.retina.vectorized_receptors[eye][receptor_type_vec]
                    
                    # Find this receptor in the receptor_objects list
                    idx = vec_data['receptor_objects'].index(r)
                    
                    # Get activation from vectorized array
                    v_membrane = vec_data['v_membrane'][idx]
                    v_rest = r.v_rest
                    v_threshold = -55.0
                    activation = np.clip((v_membrane - v_rest) / (v_threshold - v_rest), 0, 1)
                    activations.append(activation)
                except (KeyError, ValueError):
                    # Fallback to object method if not found
                    activations.append(r.get_activation())
            
            return np.array(activations)
        else:
            # Original retina - use object method
            return np.array([r.get_activation() for r in receptors])
    
    @benchmark_function('bipolar_get_activity_map')
    def get_activity_map(self, eye: str, cell_type: str = 'ON') -> np.ndarray:
        """OPTIMIZED: Get activity map with caching"""
        cache_key = f"{eye}_{cell_type}"
        
        # Check cache
        cached = self.activity_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Compute activity map
        activity_map = self._compute_activity_map(eye, cell_type)
        
        # Cache it
        self.activity_cache.set(cache_key, activity_map)
        
        return activity_map
    
    def _compute_activity_map(self, eye: str, cell_type: str) -> np.ndarray:
        """Compute activity map from vectorized data"""
        grid_size = self.retina.grid_size
        activity_map = np.zeros((grid_size, grid_size))
        count_map = np.zeros((grid_size, grid_size))
        
        vec_data = self.vectorized_bipolars[eye][cell_type]
        positions = vec_data['positions']
        
        if len(positions) == 0:
            return activity_map
        
        # Get activations (vectorized)
        v_membrane = vec_data['v_membrane']
        v_rest = vec_data['v_rest']
        v_threshold = -55.0
        
        # Normalize to [0, 1]
        activations = np.clip((v_membrane - v_rest) / (v_threshold - v_rest), 0, 1)
        
        # Convert positions to grid indices
        # positions are (x, y) where x is horizontal and y is vertical
        # For array indexing: array[row, col] = array[y_index, x_index]
        grid_coords = ((positions + 1.0) / 2.0 * (grid_size - 1)).astype(int)
        grid_coords = np.clip(grid_coords, 0, grid_size - 1)
        
        # Extract row and column indices
        row_indices = grid_coords[:, 1]  # y -> row
        col_indices = grid_coords[:, 0]  # x -> col
        
        # Accumulate activations
        for idx in range(len(positions)):
            activity_map[row_indices[idx], col_indices[idx]] += activations[idx]
            count_map[row_indices[idx], col_indices[idx]] += 1
        
        # Normalize
        mask = count_map > 0
        activity_map[mask] /= count_map[mask]
        
        return activity_map
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {
            'type': 'bipolar_layer_optimized',
            'receptor_type': self.receptor_type,
            'total_bipolar_cells': self.total_bipolar_cells,
            'cache_stats': self.activity_cache.get_stats()
        }
        
        for eye in ['left', 'right']:
            eye_summary = {}
            for cell_type in ['ON', 'OFF']:
                vec_data = self.vectorized_bipolars[eye][cell_type]
                
                if len(vec_data['positions']) == 0:
                    eye_summary[cell_type] = {
                        'count': 0,
                        'P_pathway': 0,
                        'M_pathway': 0,
                        'mean_activity': 0
                    }
                    continue
                
                # Count by pathway
                p_cells = np.sum(vec_data['pathway'] == 'P')
                m_cells = np.sum(vec_data['pathway'] == 'M')
                
                # Mean activity
                v_membrane = vec_data['v_membrane']
                v_rest = vec_data['v_rest']
                activations = np.clip((v_membrane - v_rest) / 15.0, 0, 1)
                mean_activity = float(np.mean(activations))
                
                eye_summary[cell_type] = {
                    'count': int(len(vec_data['positions'])),
                    'P_pathway': int(p_cells),
                    'M_pathway': int(m_cells),
                    'mean_activity': float(mean_activity)
                }
            
            summary[eye] = eye_summary
        
        return summary
    
    def get_state(self) -> Dict:
        """Get complete state (compatibility)"""
        return {
            'summary': self.get_summary(),
            'bipolar_cells': {
                eye: {
                    cell_type: [c.get_state() for c in self.vectorized_bipolars[eye][cell_type]['cell_objects']]
                    for cell_type in ['ON', 'OFF']
                }
                for eye in ['left', 'right']
            }
        }
    
    def find_neuron_at_position(self, eye: str, x: float, y: float,
                                cell_type: str = None, max_distance: float = 0.1) -> Optional[BipolarCell]:
        """Find bipolar cell closest to position"""
        closest = None
        min_dist = max_distance
        
        types_to_search = [cell_type] if cell_type else ['ON', 'OFF']
        
        for ct in types_to_search:
            vec_data = self.vectorized_bipolars[eye][ct]
            positions = vec_data['positions']
            
            if len(positions) == 0:
                continue
            
            # Vectorized distance calculation
            distances = np.sqrt((positions[:, 0] - x)**2 + (positions[:, 1] - y)**2)
            min_idx = np.argmin(distances)
            
            if distances[min_idx] < min_dist:
                min_dist = distances[min_idx]
                closest = vec_data['cell_objects'][min_idx]
        
        return closest
    
    def get_receptive_field_info(self, cell: BipolarCell) -> Dict:
        """Get receptive field information for a bipolar cell"""
        center_positions = [(r.position[0], r.position[1]) for r in cell.center_photoreceptors]
        surround_positions = [(r.position[0], r.position[1]) for r in cell.surround_photoreceptors]
        
        return {
            'center_photoreceptors': center_positions,
            'surround_photoreceptors': surround_positions,
            'cell_position': cell.position,
            'cell_type': cell.cell_type,
            'pathway': cell.pathway
        }
    
    def get_output_targets(self, cell: BipolarCell) -> List[Dict]:
        """Get output targets (placeholder)"""
        return []

