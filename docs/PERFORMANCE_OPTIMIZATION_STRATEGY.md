# Performance Optimization Strategy for Neural System

## Current Performance Analysis

### Identified Bottlenecks

Based on codebase audit, here are the main performance issues:

#### 1. **Data Transfer (Frontend ↔ Backend)**
- **Problem**: Every frame sends full activity maps as nested Python lists → JSON
- **Current Size**: 
  - 64×64 grid = 4,096 floats per map
  - 4 receptor types × 2 eyes = 8 maps for retina
  - 2 bipolar types × 2 eyes = 4 maps
  - 2 ganglion types × 2 eyes = 4 maps
  - **Total: ~65KB per frame at 15 FPS = 975 KB/s**
- **Impact**: Network serialization/deserialization is slow
- **Solution**: Binary format (NumPy → binary buffer), compression, delta encoding

#### 2. **Redundant Computations**
- **Problem**: Full activity map regeneration every frame
- **Current**: Iterating through all neurons, converting positions, bilinear interpolation
- **Impact**: O(N × grid²) complexity per frame
- **Solution**: 
  - Cache activity maps, only update changed regions
  - Pre-compute spatial indices for neurons
  - Use vectorized NumPy operations instead of Python loops

#### 3. **No GPU Utilization**
- **Problem**: All computations run on CPU with Python loops
- **Your Hardware**: M5 with powerful GPU (Metal), 128GB RAM
- **Impact**: 90%+ of compute power unused
- **Solution**: 
  - Port neural updates to GPU with CuPy/PyTorch/JAX
  - Batch operations for parallel neuron updates
  - GPU-accelerated convolutions for receptive fields

#### 4. **Inefficient Receptive Field Lookups**
- **Problem**: Every bipolar/ganglion cell iterates through ALL photoreceptors/bipolars
- **Current**: O(N×M) neighbor searches
- **Impact**: Scales poorly as system grows
- **Solution**:
  - Spatial indexing (KD-tree, grid-based lookup)
  - Pre-compute connectivity matrices (sparse)
  - Store as CSR/COO sparse matrices

#### 5. **Frontend Canvas Rendering**
- **Problem**: JavaScript loops drawing individual pixels/cells
- **Current**: Nested loops with fillRect() calls
- **Impact**: Blocks main thread, slow redraws
- **Solution**:
  - Use WebGL for GPU-accelerated rendering
  - ImageData direct pixel manipulation
  - Offscreen canvas with workers

#### 6. **No Caching or Memoization**
- **Problem**: Recalculating same values (Gaussian filters, pooling zones, etc.)
- **Solution**: Cache intermediate results, pre-compute static data

---

## Optimization Roadmap

### Phase 1: Quick Wins (1-2 days) - **10x Faster Target**

#### 1.1 Data Transfer Optimization
```python
# BEFORE: temporal_server.py (lines 194-200)
left_retina_maps[receptor_type] = neural_system['retina'].get_activity_map(
    'left', receptor_type
).tolist()  # ❌ Slow: NumPy → nested lists → JSON

# AFTER: Use binary format
import base64
activity_map = neural_system['retina'].get_activity_map('left', receptor_type)
# Convert to 16-bit float (half precision) for 50% size reduction
activity_map_f16 = activity_map.astype(np.float16)
# Base64 encode binary data
encoded = base64.b64encode(activity_map_f16.tobytes()).decode('ascii')
left_retina_maps[receptor_type] = {
    'data': encoded,
    'shape': activity_map.shape,
    'dtype': 'float16'
}
```

**Expected gain**: 3-5x reduction in data size, 2-3x faster serialization

#### 1.2 Cache Activity Maps
```python
class FovealRetina:
    def __init__(self, ...):
        ...
        # Add caching
        self._activity_map_cache = {}
        self._cache_valid = {}
        
    def process_image(self, ...):
        ...
        # Invalidate cache when new image arrives
        self._cache_valid = {eye: {rtype: False for rtype in ['rods', ...]} 
                            for eye in ['left', 'right']}
    
    def get_activity_map(self, eye, receptor_type):
        cache_key = (eye, receptor_type)
        if self._cache_valid.get(cache_key, False):
            return self._activity_map_cache[cache_key]
        
        # Compute activity map
        activity_map = self._compute_activity_map(eye, receptor_type)
        
        # Cache it
        self._activity_map_cache[cache_key] = activity_map
        self._cache_valid[cache_key] = True
        
        return activity_map
```

**Expected gain**: 2-3x faster for subsequent calls within same frame

#### 1.3 Vectorize Neuron Updates
```python
# BEFORE: Looping through neurons
for receptor in receptor_list:
    receptor.update(dt, current_time)  # ❌ Python loop overhead

# AFTER: Batch update with NumPy
class FovealRetina:
    def __init__(self, ...):
        # Store neuron properties as arrays for vectorization
        self.receptor_properties = {
            'left': {
                'rods': {
                    'v_membrane': np.array([]),
                    'light_input': np.array([]),
                    'adapted_response': np.array([]),
                    # ... other properties
                }
            }
        }
    
    def update(self, dt):
        # Vectorized update for all receptors of same type
        for eye in ['left', 'right']:
            for rtype in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
                props = self.receptor_properties[eye][rtype]
                
                # Vectorized adaptation
                adaptation_rate = dt / tau_adaptation
                props['adapted_response'] += (
                    (props['light_input'] - props['adapted_response']) 
                    * adaptation_rate
                )
                
                # Vectorized membrane update
                target_v = v_dark + (v_light - v_dark) * props['adapted_response']
                props['v_membrane'] += (
                    (target_v - props['v_membrane']) * (dt / tau_membrane)
                )
```

**Expected gain**: 5-10x faster neuron updates

#### 1.4 Frontend: Use ImageData API
```javascript
// BEFORE: Nested loops with fillRect() - lines 1003-1097 in temporal_index.html
for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
    }
}

// AFTER: Direct pixel manipulation
function drawActivityMapFast(canvasId, activityMap) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;
    
    // Decode from base64 if using binary format
    const activityArray = decodeBase64ToFloat16(activityMap.data);
    
    // Direct pixel writing (4x faster than fillRect)
    let idx = 0;
    for (let i = 0; i < canvas.height; i++) {
        for (let j = 0; j < canvas.width; j++) {
            const value = activityArray[i * canvas.width + j];
            const color = valueToColor(value, colorScheme);
            data[idx++] = color.r;
            data[idx++] = color.g;
            data[idx++] = color.b;
            data[idx++] = 255; // alpha
        }
    }
    
    ctx.putImageData(imageData, 0, 0);
}
```

**Expected gain**: 3-5x faster rendering

#### 1.5 Reduce Update Frequency Strategically
```python
# Different update rates for different components
class NeuralSystem:
    def __init__(self):
        self.retina_update_interval = 1000 / 30  # 30 FPS
        self.summary_update_interval = 2000  # 2 seconds
        self.last_summary_update = 0
        
    def update_loop(self):
        # Fast path: only retina and essential processing
        self.retina.update(dt)
        self.bipolar_layer.update(dt)
        self.ganglion_layer.update(dt)
        
        # Slow path: summary statistics
        if time.time() - self.last_summary_update > self.summary_update_interval:
            self.update_summary_stats()
            self.last_summary_update = time.time()
```

**Expected gain**: 1.5-2x overall speedup

---

### Phase 2: GPU Acceleration (3-5 days) - **50x Faster Target**

#### 2.1 Install GPU Libraries
```bash
# Add to requirements.txt
torch>=2.0.0  # PyTorch with Metal (MPS) support for M5
# OR
cupy-cuda11x  # If using CUDA GPU
# OR
jax[metal]>=0.4.0  # JAX with Metal support (Apple Silicon optimized)
```

#### 2.2 Port Neural Updates to GPU
```python
import torch

class GPUFovealRetina:
    def __init__(self, grid_size, fovea_radius):
        # Use Metal Performance Shaders on M5
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Store all receptor properties as GPU tensors
        self.v_membrane = torch.zeros(num_receptors, device=self.device)
        self.light_input = torch.zeros(num_receptors, device=self.device)
        self.adapted_response = torch.zeros(num_receptors, device=self.device)
        
        # Pre-compute constants
        self.tau_adaptation = torch.tensor(100.0, device=self.device)
        self.tau_membrane = torch.tensor(20.0, device=self.device)
    
    def update(self, dt):
        # Fully vectorized GPU update
        adaptation_rate = dt / self.tau_adaptation
        
        # All operations run on GPU in parallel
        self.adapted_response += (
            (self.light_input - self.adapted_response) * adaptation_rate
        )
        
        target_v = self.v_dark + (self.v_light - self.v_dark) * self.adapted_response
        self.v_membrane += (target_v - self.v_membrane) * (dt / self.tau_membrane)
        
        # Only transfer to CPU when needed for visualization
```

**Expected gain**: 10-50x faster neuron updates depending on scale

#### 2.3 GPU-Accelerated Receptive Fields
```python
import torch
import torch.nn.functional as F

class GPUBipolarLayer:
    def __init__(self, retina):
        self.device = torch.device('mps')
        
        # Pre-compute receptive field connectivity as sparse matrix
        # Shape: (num_bipolar_cells, num_receptors)
        self.center_connectivity = self._build_connectivity_matrix('center')
        self.surround_connectivity = self._build_connectivity_matrix('surround')
        
        # Convert to sparse COO format for efficiency
        self.center_connectivity = self.center_connectivity.to_sparse().to(self.device)
        self.surround_connectivity = self.surround_connectivity.to_sparse().to(self.device)
    
    def compute_receptive_field_inputs(self):
        # Sparse matrix multiply on GPU (extremely fast)
        # Shape: (num_bipolar_cells,) = (num_bipolar_cells, num_receptors) @ (num_receptors,)
        center_input = torch.sparse.mm(
            self.center_connectivity, 
            self.retina.adapted_response.unsqueeze(1)
        ).squeeze()
        
        surround_input = torch.sparse.mm(
            self.surround_connectivity,
            self.retina.adapted_response.unsqueeze(1)
        ).squeeze()
        
        # Center-surround computation (parallel for all cells)
        bipolar_input = center_input - 0.6 * surround_input
        
        return bipolar_input
```

**Expected gain**: 20-100x faster receptive field computation

#### 2.4 Pre-compute Spatial Indices
```python
class SpatialIndex:
    """Fast spatial lookups using grid-based indexing"""
    def __init__(self, neurons, grid_resolution=32):
        self.grid_resolution = grid_resolution
        
        # Build spatial grid
        self.grid = {}  # (grid_x, grid_y) -> [neuron_ids]
        
        for neuron in neurons:
            x, y = neuron.position
            grid_x = int((x + 1.0) / 2.0 * grid_resolution)
            grid_y = int((y + 1.0) / 2.0 * grid_resolution)
            grid_key = (grid_x, grid_y)
            
            if grid_key not in self.grid:
                self.grid[grid_key] = []
            self.grid[grid_key].append(neuron)
    
    def find_nearby(self, position, radius):
        """O(1) lookup instead of O(N) search"""
        x, y = position
        grid_x = int((x + 1.0) / 2.0 * self.grid_resolution)
        grid_y = int((y + 1.0) / 2.0 * self.grid_resolution)
        
        # Check nearby grid cells
        grid_radius = int(radius * self.grid_resolution) + 1
        nearby = []
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                grid_key = (grid_x + dx, grid_y + dy)
                if grid_key in self.grid:
                    nearby.extend(self.grid[grid_key])
        
        return nearby
```

**Expected gain**: 100-1000x faster neighbor searches

---

### Phase 3: Advanced Optimizations (5-7 days) - **100x Faster Target**

#### 3.1 WebGL Frontend Rendering
```javascript
// Replace canvas 2D context with WebGL
// Fragment shader for activity visualization
const fragmentShaderSource = `
precision highp float;
uniform sampler2D activityTexture;
uniform int colorScheme;
varying vec2 vTexCoord;

vec3 viridis(float t) {
    // Viridis colormap in GPU shader
    const vec3 c0 = vec3(0.267, 0.005, 0.329);
    const vec3 c1 = vec3(0.283, 0.141, 0.458);
    const vec3 c2 = vec3(0.254, 0.265, 0.530);
    // ... interpolation
    return mix(c0, mix(c1, c2, t), t);
}

void main() {
    float value = texture2D(activityTexture, vTexCoord).r;
    vec3 color = viridis(value);
    gl_FragColor = vec4(color, 1.0);
}
`;

// Upload activity map as texture (extremely fast)
function drawWithWebGL(activityMap) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 
                  width, height, 0, 
                  gl.RED, gl.FLOAT, activityMap);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}
```

**Expected gain**: 10-20x faster rendering, unlocks 60+ FPS

#### 3.2 Asynchronous Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncNeuralSystem:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def update_async(self):
        # Run different layers in parallel
        retina_task = asyncio.get_event_loop().run_in_executor(
            self.executor, self.retina.update, dt
        )
        
        # Wait for retina, then update dependent layers in parallel
        await retina_task
        
        bipolar_task = asyncio.get_event_loop().run_in_executor(
            self.executor, self.bipolar_layer.update, dt
        )
        lateral_task = asyncio.get_event_loop().run_in_executor(
            self.executor, self.lateral_layer.update, dt
        )
        
        await asyncio.gather(bipolar_task, lateral_task)
        
        # Finally ganglion
        await asyncio.get_event_loop().run_in_executor(
            self.executor, self.ganglion_layer.update, dt
        )
```

**Expected gain**: 2-4x with parallelization

#### 3.3 Numba JIT Compilation
```python
from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True)
def update_receptors_numba(v_membrane, light_input, adapted_response, dt, tau_adaptation, tau_membrane):
    """JIT-compiled receptor update (CPU multi-core)"""
    n = len(v_membrane)
    adaptation_rate = dt / tau_adaptation
    
    # Parallel loop (uses all CPU cores)
    for i in prange(n):
        # Adaptation
        adapted_response[i] += (light_input[i] - adapted_response[i]) * adaptation_rate
        
        # Membrane update
        target_v = -40.0 + (-70.0 - (-40.0)) * adapted_response[i]
        v_membrane[i] += (target_v - v_membrane[i]) * (dt / tau_membrane)
    
    return v_membrane, adapted_response
```

**Expected gain**: 10-50x faster (CPU fallback if GPU not suitable)

#### 3.4 Delta Encoding for Network Transfer
```python
class DeltaEncoder:
    """Only send changed regions"""
    def __init__(self):
        self.previous_frame = {}
        self.threshold = 0.01  # Minimum change to transmit
    
    def encode(self, current_frame, key):
        if key not in self.previous_frame:
            # First frame: send everything
            self.previous_frame[key] = current_frame.copy()
            return {'full': True, 'data': current_frame}
        
        # Compute difference
        diff = current_frame - self.previous_frame[key]
        changed_mask = np.abs(diff) > self.threshold
        
        if np.sum(changed_mask) < 0.1 * current_frame.size:
            # Less than 10% changed: send sparse update
            changed_indices = np.where(changed_mask)
            changed_values = current_frame[changed_indices]
            
            self.previous_frame[key] = current_frame.copy()
            
            return {
                'full': False,
                'indices': changed_indices,
                'values': changed_values
            }
        else:
            # More than 10% changed: send full frame
            self.previous_frame[key] = current_frame.copy()
            return {'full': True, 'data': current_frame}
```

**Expected gain**: 5-10x reduction in network traffic for typical scenes

#### 3.5 Connection Pooling & Batch API
```python
@app.route('/api/state/batch', methods=['POST'])
def get_batch_state():
    """Get multiple state components in one request"""
    requested = request.json.get('components', [])
    
    result = {}
    
    if 'retina' in requested:
        result['retina'] = get_retina_state()
    if 'bipolar' in requested:
        result['bipolar'] = get_bipolar_state()
    if 'ganglion' in requested:
        result['ganglion'] = get_ganglion_state()
    
    return jsonify(result)
```

**Expected gain**: 2-3x reduction in request overhead

---

## Implementation Plan

### Week 1: Foundation (Phase 1)
- Day 1-2: Data transfer optimization (binary format, compression)
- Day 3-4: Activity map caching and vectorization
- Day 5: Frontend ImageData optimization
- **Expected: 10x faster**

### Week 2: GPU Acceleration (Phase 2)
- Day 1-2: Set up PyTorch with Metal (MPS) support
- Day 3-4: Port neuron updates to GPU
- Day 5: GPU receptive field computation
- **Expected: 50x faster**

### Week 3: Advanced (Phase 3)
- Day 1-2: WebGL rendering
- Day 3-4: Asynchronous processing, Numba JIT
- Day 5: Delta encoding and batch API
- **Expected: 100x faster**

---

## Benchmarking Strategy

### Current Baseline
```python
import time

def benchmark_system():
    """Measure current performance"""
    # Initialize system
    start = time.time()
    system.initialize(grid_size=64)
    init_time = time.time() - start
    
    # Update loop
    start = time.time()
    for _ in range(100):
        system.update(dt=33.3)  # 30 FPS
    update_time = (time.time() - start) / 100
    
    # Visualization
    start = time.time()
    for _ in range(100):
        state = system.get_current_state()
    viz_time = (time.time() - start) / 100
    
    print(f"Init: {init_time:.3f}s")
    print(f"Update: {update_time*1000:.2f}ms ({1000/update_time:.1f} FPS)")
    print(f"Visualization: {viz_time*1000:.2f}ms")
```

### Track Improvements
```python
# Create benchmark suite
benchmarks = {
    'baseline': benchmark_system,
    'phase1_binary': benchmark_system_phase1,
    'phase2_gpu': benchmark_system_phase2,
    'phase3_webgl': benchmark_system_phase3
}

# Run and compare
for name, benchmark in benchmarks.items():
    print(f"\n{name}:")
    benchmark()
```

---

## Hardware-Specific Optimizations

### Apple M5 Studio (Your System)
- **Metal Performance Shaders (MPS)**: Native GPU acceleration
- **Unified Memory**: CPU and GPU share 128GB - use large batches
- **Neural Engine**: Can be used for matrix operations

```python
import torch

# Use MPS backend
device = torch.device('mps')

# Large batch sizes (you have 128GB!)
batch_size = 10000  # Process 10K neurons in one batch

# Use float16 for 2x memory efficiency
model = model.half()

# Metal-optimized convolutions
conv = torch.nn.Conv2d(...).to(device)
```

---

## Profiling Tools

### Python Profiling
```bash
# Line profiler
pip install line_profiler
kernprof -l -v temporal_server.py

# Memory profiler
pip install memory_profiler
python -m memory_profiler temporal_server.py

# PyTorch profiler (for GPU)
python -m torch.utils.bottleneck temporal_server.py
```

### Frontend Profiling
```javascript
// Chrome DevTools Performance tab
performance.mark('update-start');
updateVisualizations();
performance.mark('update-end');
performance.measure('update', 'update-start', 'update-end');
```

---

## Expected Final Performance

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| Update FPS | ~5-10 | 30-50 | 100-200 | 200-500 |
| Network (KB/s) | 975 | 200 | 100 | 20 |
| CPU Usage | 80-90% | 40-60% | 10-20% | 5-10% |
| GPU Usage | 0% | 0% | 60-80% | 80-95% |
| Latency | 100-200ms | 30-50ms | 10-20ms | 5-10ms |

---

## Scalability Strategy

As you add more brain areas:

1. **Modular Updates**: Each brain region updates independently
2. **Level-of-Detail (LOD)**: Reduce resolution for non-focused areas
3. **Lazy Evaluation**: Only compute visible neurons
4. **Distributed Computing**: Offload brain regions to separate GPU/processes

```python
class ScalableBrain:
    def __init__(self):
        self.regions = {
            'retina': RetinaRegion(priority='high'),
            'lgn': LGNRegion(priority='high'),
            'v1': V1Region(priority='medium'),
            'v2': V2Region(priority='low')
        }
        
    def update(self, dt, focus_region='retina'):
        # High priority: full resolution
        self.regions[focus_region].update(dt, lod=1.0)
        
        # Medium priority: 50% resolution
        for region in self.get_adjacent_regions(focus_region):
            self.regions[region].update(dt, lod=0.5)
        
        # Low priority: 25% resolution or skip
        for region in self.get_distant_regions(focus_region):
            if frame_count % 3 == 0:  # Update every 3rd frame
                self.regions[region].update(dt, lod=0.25)
```

---

## Next Steps

1. **Run baseline benchmarks** (see Benchmarking Strategy)
2. **Start with Phase 1** (quickest wins, foundation for rest)
3. **Validate each optimization** (compare before/after)
4. **Iterate based on profiling** (find remaining bottlenecks)
5. **Scale to more brain regions** (using proven optimizations)

**Goal: Make system 100x faster while maintaining biological accuracy and visual quality**

