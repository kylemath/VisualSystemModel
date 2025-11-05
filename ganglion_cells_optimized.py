"""
OPTIMIZED ganglion cell layer with vectorized updates and sparse connectivity.
Expected 10-50x performance improvement over original.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from temporal_neuron import TemporalNeuron
from ganglion_cells import GanglionCell  # Keep for compatibility
from performance_utils import (ActivityMapCache, benchmark_function,
                               SparseConnectivityMatrix)

# Try to use optimized layers
try:
    from bipolar_cells_optimized import BipolarLayerOptimized as BipolarLayer
except ImportError:
    from bipolar_cells import BipolarLayer


class GanglionLayerOptimized:
    """
    OPTIMIZED ganglion cell layer with:
    - Vectorized neuron updates
    - Sparse connectivity matrices
    - Activity map caching
    - Fast optic nerve output generation
    """
    
    def __init__(self, bipolar_layer: BipolarLayer):
        """
        Create optimized ganglion cell layer.
        
        Args:
            bipolar_layer: BipolarLayer instance (original or optimized)
        """
        print("Creating OPTIMIZED ganglion layer...")
        
        self.bipolar_layer = bipolar_layer
        self.retina = bipolar_layer.retina
        
        # Ganglion cells for each eye (keep original for compatibility)
        self.ganglion_cells = {
            'left': {'P': [], 'M': [], 'ipRGC': []},
            'right': {'P': [], 'M': [], 'ipRGC': []}
        }
        
        # OPTIMIZATION: Vectorized arrays
        self.vectorized_ganglions = {}
        
        # OPTIMIZATION: Sparse connectivity matrices
        self.connectivity_matrices = {}
        
        # OPTIMIZATION: Activity map cache
        self.activity_cache = ActivityMapCache()
        
        # Create ganglion cells
        self._create_ganglion_cells()
        
        # OPTIMIZATION: Convert to vectorized representation
        print("Converting to vectorized arrays...")
        self._create_vectorized_arrays()
        
        # OPTIMIZATION: Build sparse connectivity matrices
        print("Building sparse connectivity matrices...")
        self._build_connectivity_matrices()
        
        # Statistics
        self.total_ganglion_cells = sum(
            len(self.vectorized_ganglions[eye][ctype]['positions'])
            for eye in ['left', 'right']
            for ctype in ['P', 'M', 'ipRGC']
        )
        
        print(f"Created {self.total_ganglion_cells} ganglion cells (optimized)")
    
    def _create_ganglion_cells(self):
        """Create ganglion cells"""
        for eye in ['left', 'right']:
            self._create_eye_ganglion_cells(eye)
    
    def _create_eye_ganglion_cells(self, eye: str):
        """Create ganglion cells for one eye"""
        # Get bipolar cells
        if hasattr(self.bipolar_layer, 'vectorized_bipolars'):
            # Using optimized bipolar layer
            on_bipolars = self.bipolar_layer.vectorized_bipolars[eye]['ON']['cell_objects']
            off_bipolars = self.bipolar_layer.vectorized_bipolars[eye]['OFF']['cell_objects']
        else:
            # Using original bipolar layer
            on_bipolars = self.bipolar_layer.bipolar_cells[eye]['ON']
            off_bipolars = self.bipolar_layer.bipolar_cells[eye]['OFF']
        
        all_bipolars = on_bipolars + off_bipolars
        
        if len(all_bipolars) == 0:
            return
        
        # Generate ganglion positions
        ganglion_positions = self._generate_ganglion_positions()
        
        for position in ganglion_positions:
            x, y = position
            eccentricity = np.sqrt(x**2 + y**2)
            
            # Determine cell type based on eccentricity
            if eccentricity < self.retina.fovea_radius:
                # Fovea: mostly P cells
                type_prob = np.random.rand()
                if type_prob < 0.70:
                    cell_type = 'P'
                elif type_prob < 0.95:
                    cell_type = 'M'
                else:
                    cell_type = 'ipRGC'
            else:
                # Periphery: more M cells
                type_prob = np.random.rand()
                if type_prob < 0.30:
                    cell_type = 'P'
                elif type_prob < 0.95:
                    cell_type = 'M'
                else:
                    cell_type = 'ipRGC'
            
            # Create ganglion cell
            ganglion = GanglionCell(cell_type, position, eye, eccentricity)
            
            # Determine pooling radius
            if cell_type == 'P':
                if eccentricity < self.retina.fovea_radius:
                    pool_radius = 0.02
                else:
                    pool_radius = 0.05
            elif cell_type == 'M':
                if eccentricity < self.retina.fovea_radius:
                    pool_radius = 0.06
                else:
                    pool_radius = 0.15
            else:  # ipRGC
                pool_radius = 0.3
            
            # Connect to bipolar cells
            self._connect_to_bipolars(ganglion, all_bipolars, pool_radius)
            
            # Add to layer
            self.ganglion_cells[eye][cell_type].append(ganglion)
    
    def _generate_ganglion_positions(self) -> List[Tuple[float, float]]:
        """Generate positions for ganglion cells"""
        positions = []
        
        # Fovea
        fovea_grid = 12
        for i in range(fovea_grid):
            for j in range(fovea_grid):
                x = (i / (fovea_grid - 1)) * 2 * self.retina.fovea_radius - self.retina.fovea_radius
                y = (j / (fovea_grid - 1)) * 2 * self.retina.fovea_radius - self.retina.fovea_radius
                positions.append((x, y))
        
        # Periphery
        periphery_grid = 8
        for i in range(periphery_grid):
            for j in range(periphery_grid):
                x = (i / (periphery_grid - 1)) * 2.0 - 1.0
                y = (j / (periphery_grid - 1)) * 2.0 - 1.0
                
                eccentricity = np.sqrt(x**2 + y**2)
                
                if eccentricity > self.retina.fovea_radius:
                    positions.append((x, y))
        
        return positions
    
    def _connect_to_bipolars(self, ganglion: GanglionCell,
                            bipolars: List,
                            pool_radius: float):
        """Connect ganglion cell to nearby bipolar cells"""
        gx, gy = ganglion.position
        
        for bipolar in bipolars:
            bx, by = bipolar.position
            distance = np.sqrt((gx - bx)**2 + (gy - by)**2)
            
            if distance < pool_radius:
                weight = 1.0 - (distance / pool_radius)
                ganglion.add_bipolar_input(bipolar, weight)
    
    def _create_vectorized_arrays(self):
        """OPTIMIZATION: Convert ganglion cells to vectorized arrays"""
        for eye in ['left', 'right']:
            self.vectorized_ganglions[eye] = {}
            
            for cell_type in ['P', 'M', 'ipRGC']:
                cells = self.ganglion_cells[eye][cell_type]
                n = len(cells)
                
                if n == 0:
                    self.vectorized_ganglions[eye][cell_type] = {
                        'positions': np.array([]).reshape(0, 2),
                        'v_membrane': np.array([]),
                        'v_rest': np.array([]),
                        'v_threshold': np.array([]),
                        'eccentricity': np.array([]),
                        'is_spiking': np.array([], dtype=bool),
                        'spike_accumulator': np.array([]),  # Track recent spikes
                        'cell_objects': []
                    }
                    continue
                
                # Extract properties
                positions = np.array([c.position for c in cells])
                v_membrane = np.array([c.v_membrane for c in cells])
                v_rest = np.array([c.v_rest for c in cells])
                v_threshold = np.array([c.v_threshold for c in cells])
                eccentricity = np.array([c.eccentricity for c in cells])
                is_spiking = np.array([c.is_spiking for c in cells], dtype=bool)
                
                self.vectorized_ganglions[eye][cell_type] = {
                    'positions': positions,
                    'v_membrane': v_membrane,
                    'v_rest': v_rest,
                    'v_threshold': v_threshold,
                    'eccentricity': eccentricity,
                    'is_spiking': is_spiking,
                    'spike_accumulator': np.zeros(n),  # Accumulate spikes over time (decays)
                    'cell_objects': cells
                }
    
    def _build_connectivity_matrices(self):
        """OPTIMIZATION: Build sparse connectivity matrices"""
        for eye in ['left', 'right']:
            self.connectivity_matrices[eye] = {}
            
            for cell_type in ['P', 'M', 'ipRGC']:
                cells = self.ganglion_cells[eye][cell_type]
                
                if len(cells) == 0:
                    self.connectivity_matrices[eye][cell_type] = None
                    continue
                
                # Get all unique bipolar cells
                all_bipolars = set()
                for cell in cells:
                    all_bipolars.update(cell.on_bipolar_inputs)
                    all_bipolars.update(cell.off_bipolar_inputs)
                
                all_bipolars = list(all_bipolars)
                bipolar_to_idx = {id(b): i for i, b in enumerate(all_bipolars)}
                
                # Build connectivity matrix
                num_ganglion = len(cells)
                num_bipolar = len(all_bipolars)
                
                conn_matrix = SparseConnectivityMatrix(num_ganglion, num_bipolar)
                
                for ganglion_idx, cell in enumerate(cells):
                    # ON bipolar connections
                    for bipolar in cell.on_bipolar_inputs:
                        bipolar_idx = bipolar_to_idx[id(bipolar)]
                        conn_matrix.add_connection(ganglion_idx, bipolar_idx, 1.0)
                    
                    # OFF bipolar connections
                    for bipolar in cell.off_bipolar_inputs:
                        bipolar_idx = bipolar_to_idx[id(bipolar)]
                        conn_matrix.add_connection(ganglion_idx, bipolar_idx, 1.0)
                
                # Finalize matrix
                conn_matrix.finalize()
                
                self.connectivity_matrices[eye][cell_type] = {
                    'matrix': conn_matrix,
                    'bipolars': all_bipolars
                }
    
    @benchmark_function('ganglion_update')
    def update(self, dt: float = 1.0):
        """OPTIMIZED: Vectorized update with sparse matrix multiplication"""
        tau_membrane = 20.0
        debug_counter = getattr(self, '_debug_counter', 0)
        self._debug_counter = debug_counter + 1
        
        for eye in ['left', 'right']:
            for cell_type in ['P', 'M', 'ipRGC']:
                vec_data = self.vectorized_ganglions[eye][cell_type]
                conn_data = self.connectivity_matrices[eye][cell_type]
                
                if conn_data is None or len(vec_data['positions']) == 0:
                    continue
                
                # Get bipolar activations (from vectorized bipolar if available)
                bipolar_activations = self._get_bipolar_activations(eye, conn_data['bipolars'])
                
                # VECTORIZED receptive field computation
                bipolar_input = conn_data['matrix'].compute_outputs(bipolar_activations)
                
                # Scale to physiological range (increased gain for visibility)
                total_current = bipolar_input * 40.0
                
                # VECTORIZED membrane integration
                target_v = vec_data['v_rest'] + total_current
                dv = ((target_v - vec_data['v_membrane']) / tau_membrane) * dt
                vec_data['v_membrane'] += dv
                
                # Detect spikes (vectorized)
                spiking_mask = vec_data['v_membrane'] >= vec_data['v_threshold']
                vec_data['is_spiking'] = spiking_mask
                
                # Accumulate spikes with temporal decay
                # Decay previous spikes (half-life of ~100ms = 3 frames at 30fps)
                decay_factor = 0.8  # Each frame retains 80% of previous activity
                vec_data['spike_accumulator'] *= decay_factor
                # Add new spikes (full strength = 1.0)
                vec_data['spike_accumulator'][spiking_mask] = 1.0
                
                # Reset spiking neurons
                vec_data['v_membrane'][spiking_mask] = -75.0  # v_reset
        
        # Invalidate activity cache
        self.activity_cache.invalidate_all()
    
    def _get_bipolar_activations(self, eye: str, bipolars: List) -> np.ndarray:
        """
        Get bipolar activations, using vectorized arrays if available.
        Fixes bug where optimized bipolar updates vectors but not objects.
        """
        # Check if bipolar layer has vectorized cells (optimized)
        if hasattr(self.bipolar_layer, 'vectorized_bipolars'):
            activations = []
            for b in bipolars:
                # Find which cell type this is
                cell_type = b.cell_type  # 'ON' or 'OFF'
                vec_data = self.bipolar_layer.vectorized_bipolars[eye][cell_type]
                
                # Find this bipolar in the cell_objects list
                try:
                    idx = vec_data['cell_objects'].index(b)
                    # Get activation from vectorized array
                    v_membrane = vec_data['v_membrane'][idx]
                    v_rest = b.v_rest
                    v_threshold = -55.0
                    activation = np.clip((v_membrane - v_rest) / (v_threshold - v_rest), 0, 1)
                    activations.append(activation)
                except ValueError:
                    # Fallback to object method
                    activations.append(b.get_activation())
            
            return np.array(activations)
        else:
            # Original bipolar layer - use object method
            return np.array([b.get_activation() for b in bipolars])
    
    @benchmark_function('ganglion_get_activity_map')
    def get_activity_map(self, eye: str, cell_type: str = 'P') -> np.ndarray:
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
        """Compute activity map from vectorized data - uses spiking state for ganglion cells"""
        grid_size = self.retina.grid_size
        activity_map = np.zeros((grid_size, grid_size))
        count_map = np.zeros((grid_size, grid_size))
        
        vec_data = self.vectorized_ganglions[eye][cell_type]
        positions = vec_data['positions']
        
        if len(positions) == 0:
            return activity_map
        
        # FIX: Ganglion cells are spiking neurons - use spike_accumulator instead of v_membrane
        # because v_membrane gets reset to -75 after each spike.
        # spike_accumulator tracks recent spikes with temporal decay.
        activations = vec_data['spike_accumulator']
        
        # Convert positions to grid indices
        # positions are (x, y) where x is horizontal and y is vertical
        # For array indexing: array[row, col] = array[y_index, x_index]
        grid_coords = ((positions + 1.0) / 2.0 * (grid_size - 1)).astype(int)
        grid_coords = np.clip(grid_coords, 0, grid_size - 1)
        
        # Extract row and column indices
        row_indices = grid_coords[:, 1]  # y -> row
        col_indices = grid_coords[:, 0]  # x -> col
        
        # Accumulate activations (spike counts)
        for idx in range(len(positions)):
            activity_map[row_indices[idx], col_indices[idx]] += activations[idx]
            count_map[row_indices[idx], col_indices[idx]] += 1
        
        # Average spikes per position (will be 0 or 1 for single neurons)
        mask = count_map > 0
        activity_map[mask] /= count_map[mask]
        
        return activity_map
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {
            'type': 'ganglion_layer_optimized',
            'total_ganglion_cells': self.total_ganglion_cells,
            'cache_stats': self.activity_cache.get_stats()
        }
        
        for eye in ['left', 'right']:
            eye_summary = {}
            for cell_type in ['P', 'M', 'ipRGC']:
                vec_data = self.vectorized_ganglions[eye][cell_type]
                
                if len(vec_data['positions']) == 0:
                    eye_summary[cell_type] = {
                        'count': 0,
                        'foveal': 0,
                        'peripheral': 0,
                        'mean_activity': 0,
                        'mean_spike_rate': 0
                    }
                    continue
                
                # Count by eccentricity
                foveal = np.sum(vec_data['eccentricity'] < self.retina.fovea_radius)
                peripheral = len(vec_data['positions']) - foveal
                
                # Mean activity
                v_membrane = vec_data['v_membrane']
                v_rest = vec_data['v_rest']
                v_threshold = vec_data['v_threshold']
                activations = np.clip(
                    (v_membrane - v_rest) / (v_threshold - v_rest),
                    0, 1
                )
                mean_activity = float(np.mean(activations))
                
                # Spike rate (approximate from is_spiking)
                mean_spike_rate = float(np.sum(vec_data['is_spiking'])) / len(vec_data['positions']) * 100
                
                eye_summary[cell_type] = {
                    'count': int(len(vec_data['positions'])),
                    'foveal': int(foveal),
                    'peripheral': int(peripheral),
                    'mean_activity': float(mean_activity),
                    'mean_spike_rate': float(mean_spike_rate)
                }
            
            summary[eye] = eye_summary
        
        return summary
    
    def get_optic_nerve_output(self, eye: str) -> List[Dict]:
        """Get optic nerve output (spiking ganglion cells)"""
        output = []
        
        for cell_type in ['P', 'M', 'ipRGC']:
            vec_data = self.vectorized_ganglions[eye][cell_type]
            
            if len(vec_data['positions']) == 0:
                continue
            
            # Find spiking cells (vectorized)
            spiking_indices = np.where(vec_data['is_spiking'])[0]
            
            for idx in spiking_indices:
                cell = vec_data['cell_objects'][idx]
                output.append({
                    'cell_id': cell.id,
                    'cell_type': cell_type,
                    'position': cell.position,
                    'spike_rate': cell.spike_rate,
                    'membrane_potential': cell.v_membrane
                })
        
        return output
    
    def find_neuron_at_position(self, eye: str, x: float, y: float,
                                cell_type: str = None, max_distance: float = 0.15) -> Optional[GanglionCell]:
        """Find ganglion cell closest to position"""
        closest = None
        min_dist = max_distance
        
        types_to_search = [cell_type] if cell_type else ['P', 'M', 'ipRGC']
        
        for ct in types_to_search:
            vec_data = self.vectorized_ganglions[eye][ct]
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
    
    def get_receptive_field_info(self, cell: GanglionCell) -> Dict:
        """Get receptive field information for a ganglion cell"""
        on_bipolar_positions = [(b.position[0], b.position[1])
                               for b in cell.on_bipolar_inputs]
        off_bipolar_positions = [(b.position[0], b.position[1])
                                for b in cell.off_bipolar_inputs]
        
        return {
            'on_bipolar_inputs': on_bipolar_positions,
            'off_bipolar_inputs': off_bipolar_positions,
            'cell_position': cell.position,
            'cell_type': cell.cell_subtype
        }
    
    def find_ganglions_connected_to_bipolar(self, bipolar_cell) -> List[Dict]:
        """Find ganglion cells connected to a specific bipolar cell"""
        connected = []
        bipolar_pos = bipolar_cell.position
        
        for eye in ['left', 'right']:
            if bipolar_cell.eye != eye:
                continue
            
            for cell_type in ['P', 'M', 'ipRGC']:
                for ganglion in self.ganglion_cells[eye][cell_type]:
                    all_inputs = ganglion.on_bipolar_inputs + ganglion.off_bipolar_inputs
                    if bipolar_cell in all_inputs:
                        connected.append({
                            'position': ganglion.position,
                            'cell_type': ganglion.cell_subtype,
                            'id': ganglion.id
                        })
        
        return connected
    
    def get_state(self) -> Dict:
        """Get complete state (compatibility)"""
        return {
            'summary': self.get_summary(),
            'ganglion_cells': {
                eye: {
                    cell_type: [c.get_state() for c in self.vectorized_ganglions[eye][cell_type]['cell_objects']]
                    for cell_type in ['P', 'M', 'ipRGC']
                }
                for eye in ['left', 'right']
            }
        }

