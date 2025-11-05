"""
Performance optimization utilities for neural system.
Includes caching, binary encoding, vectorization helpers.
"""

import numpy as np
import base64
from typing import Dict, Any, Optional, Tuple
import time
from functools import wraps


class PerformanceMonitor:
    """Track performance metrics"""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
    
    def time_function(self, func_name):
        """Decorator to time function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                if func_name not in self.timings:
                    self.timings[func_name] = []
                    self.call_counts[func_name] = 0
                
                self.timings[func_name].append(elapsed)
                self.call_counts[func_name] += 1
                
                return result
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics"""
        stats = {}
        for func_name, times in self.timings.items():
            stats[func_name] = {
                'count': self.call_counts[func_name],
                'total_ms': sum(times) * 1000,
                'mean_ms': np.mean(times) * 1000,
                'std_ms': np.std(times) * 1000,
                'min_ms': min(times) * 1000,
                'max_ms': max(times) * 1000,
            }
        return stats
    
    def print_report(self):
        """Print performance report"""
        print("\n" + "="*70)
        print("PERFORMANCE REPORT")
        print("="*70)
        
        stats = self.get_stats()
        
        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_ms'], reverse=True)
        
        print(f"{'Function':<30} {'Calls':>8} {'Total(ms)':>12} {'Mean(ms)':>12}")
        print("-"*70)
        
        for func_name, s in sorted_stats:
            print(f"{func_name:<30} {s['count']:>8} {s['total_ms']:>12.2f} {s['mean_ms']:>12.3f}")
        
        print("="*70 + "\n")


class BinaryEncoder:
    """Encode NumPy arrays as binary for fast network transfer"""
    
    @staticmethod
    def encode_array(array: np.ndarray, use_float16: bool = True) -> Dict[str, Any]:
        """
        Encode NumPy array to base64 binary.
        
        Args:
            array: NumPy array to encode
            use_float16: Use 16-bit floats for 50% size reduction
        
        Returns:
            Dict with encoded data, shape, and dtype
        """
        if use_float16 and array.dtype in [np.float32, np.float64]:
            array = array.astype(np.float16)
        
        return {
            'data': base64.b64encode(array.tobytes()).decode('ascii'),
            'shape': array.shape,
            'dtype': str(array.dtype)
        }
    
    @staticmethod
    def decode_array(encoded: Dict[str, Any]) -> np.ndarray:
        """Decode base64 binary to NumPy array"""
        data = base64.b64decode(encoded['data'])
        array = np.frombuffer(data, dtype=np.dtype(encoded['dtype']))
        return array.reshape(encoded['shape'])
    
    @staticmethod
    def encode_multiple(arrays: Dict[str, np.ndarray], use_float16: bool = True) -> Dict[str, Dict]:
        """Encode multiple arrays"""
        return {
            key: BinaryEncoder.encode_array(array, use_float16)
            for key, array in arrays.items()
        }


class ActivityMapCache:
    """Cache for activity maps to avoid recomputation"""
    
    def __init__(self):
        self._cache = {}
        self._valid = {}
        self._timestamps = {}
        self._access_count = {}
    
    def invalidate(self, key: Optional[str] = None):
        """Invalidate cache (mark as needing recomputation)"""
        if key is None:
            # Invalidate all
            self._valid = {}
        else:
            self._valid[key] = False
    
    def invalidate_all(self):
        """Invalidate all cached data"""
        self._valid = {}
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached array if valid"""
        if self._valid.get(key, False):
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._cache.get(key)
        return None
    
    def set(self, key: str, array: np.ndarray):
        """Cache array and mark as valid"""
        self._cache[key] = array.copy()  # Store copy to avoid mutations
        self._valid[key] = True
        self._timestamps[key] = time.time()
        self._access_count[key] = self._access_count.get(key, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(arr.nbytes for arr in self._cache.values())
        
        # Convert access_counts to ensure Python ints (not numpy int64)
        access_counts = {k: int(v) for k, v in self._access_count.items()}
        
        return {
            'num_cached': int(len(self._cache)),
            'total_size_mb': float(total_size / (1024 * 1024)),
            'access_counts': access_counts,
            'valid_count': int(sum(1 for v in self._valid.values() if v))
        }


class VectorizedNeuronArray:
    """
    Store neuron properties as NumPy arrays for vectorized updates.
    Much faster than looping through individual neuron objects.
    """
    
    def __init__(self, num_neurons: int):
        self.num_neurons = num_neurons
        
        # Membrane dynamics (all neurons)
        self.v_rest = np.full(num_neurons, -70.0)
        self.v_threshold = np.full(num_neurons, -55.0)
        self.v_membrane = np.full(num_neurons, -70.0)
        self.v_reset = np.full(num_neurons, -75.0)
        
        # Time constants
        self.tau_membrane = np.full(num_neurons, 20.0)
        
        # State
        self.is_spiking = np.zeros(num_neurons, dtype=bool)
        self.time_since_spike = np.full(num_neurons, 1000.0)
        
        # Positions (for spatial indexing)
        self.positions = np.zeros((num_neurons, 2))
        self.eccentricity = np.zeros(num_neurons)
    
    def update(self, dt: float, input_current: np.ndarray):
        """
        Vectorized update for all neurons.
        
        Args:
            dt: Time step in ms
            input_current: Array of input currents (shape: num_neurons)
        """
        # Leaky integrate-and-fire (vectorized for all neurons)
        dv = ((self.v_rest - self.v_membrane + input_current) / self.tau_membrane) * dt
        self.v_membrane += dv
        
        # Detect spikes
        spiking_mask = self.v_membrane >= self.v_threshold
        
        # Reset spiking neurons
        self.v_membrane[spiking_mask] = self.v_reset[spiking_mask]
        self.is_spiking = spiking_mask
        self.time_since_spike[spiking_mask] = 0.0
        
        # Increment time since spike for non-spiking
        self.time_since_spike[~spiking_mask] += dt
    
    def get_activation(self) -> np.ndarray:
        """Get normalized activation for all neurons"""
        return np.clip(
            (self.v_membrane - self.v_rest) / (self.v_threshold - self.v_rest),
            0, 1
        )


class SparseConnectivityMatrix:
    """
    Pre-computed sparse connectivity for fast receptive field computation.
    Uses scipy sparse matrices for memory-efficient storage.
    """
    
    def __init__(self, num_outputs: int, num_inputs: int):
        from scipy.sparse import lil_matrix
        
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        
        # Use LIL (List of Lists) format for construction
        self.matrix = lil_matrix((num_outputs, num_inputs))
    
    def add_connection(self, output_idx: int, input_idx: int, weight: float = 1.0):
        """Add connection from input to output"""
        self.matrix[output_idx, input_idx] = weight
    
    def finalize(self):
        """Convert to CSR format for fast matrix-vector products"""
        self.matrix = self.matrix.tocsr()
    
    def compute_outputs(self, input_activations: np.ndarray) -> np.ndarray:
        """
        Compute output activations from input activations.
        
        Args:
            input_activations: Array of input activations (shape: num_inputs)
        
        Returns:
            Array of output activations (shape: num_outputs)
        """
        return self.matrix.dot(input_activations)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        data_mb = self.matrix.data.nbytes / (1024 * 1024)
        indices_mb = self.matrix.indices.nbytes / (1024 * 1024)
        indptr_mb = self.matrix.indptr.nbytes / (1024 * 1024)
        
        return {
            'total_mb': data_mb + indices_mb + indptr_mb,
            'num_connections': self.matrix.nnz,
            'sparsity': 1.0 - (self.matrix.nnz / (self.num_outputs * self.num_inputs))
        }


class SpatialIndex:
    """
    Fast spatial lookups using grid-based indexing.
    Reduces O(N) neighbor search to O(1) grid lookup.
    """
    
    def __init__(self, positions: np.ndarray, grid_resolution: int = 32):
        """
        Build spatial index from positions.
        
        Args:
            positions: Array of (x, y) positions, shape (N, 2), range [-1, 1]
            grid_resolution: Number of grid cells per dimension
        """
        self.grid_resolution = grid_resolution
        self.positions = positions
        
        # Convert positions to grid indices
        grid_coords = ((positions + 1.0) / 2.0 * (grid_resolution - 1)).astype(int)
        grid_coords = np.clip(grid_coords, 0, grid_resolution - 1)
        
        # Build grid dictionary: (grid_x, grid_y) -> [neuron_indices]
        self.grid = {}
        for idx, (gx, gy) in enumerate(grid_coords):
            key = (gx, gy)
            if key not in self.grid:
                self.grid[key] = []
            self.grid[key].append(idx)
        
        # Convert lists to numpy arrays for faster access
        self.grid = {k: np.array(v) for k, v in self.grid.items()}
    
    def find_within_radius(self, position: Tuple[float, float], radius: float) -> np.ndarray:
        """
        Find all neurons within radius of position.
        
        Args:
            position: (x, y) position in [-1, 1] range
            radius: Search radius
        
        Returns:
            Array of neuron indices within radius
        """
        x, y = position
        
        # Convert position to grid coordinates
        grid_x = int((x + 1.0) / 2.0 * (self.grid_resolution - 1))
        grid_y = int((y + 1.0) / 2.0 * (self.grid_resolution - 1))
        
        # Determine grid search radius
        grid_radius = int(radius * self.grid_resolution / 2.0) + 1
        
        # Collect candidates from nearby grid cells
        candidates = []
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                key = (grid_x + dx, grid_y + dy)
                if key in self.grid:
                    candidates.append(self.grid[key])
        
        if len(candidates) == 0:
            return np.array([], dtype=int)
        
        candidate_indices = np.concatenate(candidates)
        
        # Filter by actual distance
        candidate_positions = self.positions[candidate_indices]
        distances = np.sqrt(
            (candidate_positions[:, 0] - x)**2 + 
            (candidate_positions[:, 1] - y)**2
        )
        
        within_radius = candidate_indices[distances <= radius]
        
        return within_radius


class DeltaEncoder:
    """
    Delta encoding for network transfer.
    Only send changed values to reduce bandwidth.
    """
    
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.previous = {}
    
    def encode(self, key: str, current: np.ndarray) -> Dict[str, Any]:
        """
        Encode array with delta compression.
        
        Returns dict with either:
        - {'type': 'full', 'data': encoded_array}
        - {'type': 'delta', 'indices': [...], 'values': [...]}
        """
        if key not in self.previous:
            # First time: send full
            self.previous[key] = current.copy()
            return {
                'type': 'full',
                'data': BinaryEncoder.encode_array(current)
            }
        
        # Compute difference
        diff = current - self.previous[key]
        changed_mask = np.abs(diff) > self.threshold
        num_changed = np.sum(changed_mask)
        
        if num_changed < 0.1 * current.size:
            # Less than 10% changed: send sparse delta
            changed_indices = np.where(changed_mask.flatten())[0]
            changed_values = current.flatten()[changed_indices]
            
            self.previous[key] = current.copy()
            
            return {
                'type': 'delta',
                'indices': changed_indices.tolist(),
                'values': changed_values.tolist()
            }
        else:
            # More than 10% changed: send full
            self.previous[key] = current.copy()
            return {
                'type': 'full',
                'data': BinaryEncoder.encode_array(current)
            }
    
    def reset(self):
        """Reset encoder (clear previous frames)"""
        self.previous = {}


# Global performance monitor
perf_monitor = PerformanceMonitor()


def benchmark_function(name: str):
    """Decorator for benchmarking"""
    return perf_monitor.time_function(name)

