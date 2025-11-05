# Implementation Guide: Performance Optimizations

## Quick Start - Run Benchmarks

### 1. Install Dependencies
```bash
cd /Users/kylemathewson/.cursor/worktrees/Golem/cLC1n
source venv/bin/activate

# Already installed: numpy, scipy, flask, etc.
# No new dependencies needed for Phase 1!
```

### 2. Run Baseline Benchmark
```bash
python benchmark_performance.py
```

This will show you:
- Current performance metrics
- Where bottlenecks are
- How much speedup is needed

Expected output:
```
Current Performance:
  Full frame time: 150.0ms
  Effective FPS: 6.7
  Target FPS: 30

⚠️  Need 4.5x speedup to reach 30 FPS
```

---

## Phase 1 Implementation (Quick Wins)

### Step 1: Use Optimized Retina (Already Done! ✅)

The optimized retina is ready in `foveal_retina_optimized.py`. To use it:

```python
# In temporal_server.py, line 13, replace:
from foveal_retina import FovealRetina

# With:
from foveal_retina_optimized import FovealRetinaOptimized as FovealRetina
```

**Expected improvement**: 3-5x faster retina updates

---

### Step 2: Add Binary Encoding to Server

Edit `temporal_server.py`:

```python
# Add import at top (line 11)
from performance_utils import BinaryEncoder, benchmark_function

# Modify /api/state/current endpoint (starting at line 183)
@app.route('/api/state/current', methods=['GET'])
@benchmark_function('get_current_state')
def get_current_state():
    """Get current state of all layers."""
    if not neural_system['retina']:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        # Get activity maps
        left_retina_maps = {}
        right_retina_maps = {}
        
        # OPTIMIZATION: Use binary encoding instead of .tolist()
        encoder = BinaryEncoder()
        
        for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
            left_map = neural_system['retina'].get_activity_map('left', receptor_type)
            right_map = neural_system['retina'].get_activity_map('right', receptor_type)
            
            # Binary encode with float16 for 50% size reduction
            left_retina_maps[receptor_type] = encoder.encode_array(left_map, use_float16=True)
            right_retina_maps[receptor_type] = encoder.encode_array(right_map, use_float16=True)
        
        # Same for bipolar and ganglion...
        # (repeat pattern above)
        
        return jsonify({
            'status': 'success',
            'timestamp': time.time(),
            'encoding': 'binary_float16',  # Signal to frontend
            'retina': {
                'left': left_retina_maps,
                'right': right_retina_maps
            },
            # ... rest of response
        })
```

**Expected improvement**: 2-3x reduction in response size, 2x faster serialization

---

### Step 3: Update Frontend to Decode Binary

Edit `templates/temporal_index.html`:

Add base64 decoder (insert after line 550):

```javascript
// Base64 binary decoder for float16
function decodeFloat16Array(encoded) {
    const binary = atob(encoded.data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    
    // Convert float16 to float32
    const float16View = new Uint16Array(bytes.buffer);
    const float32Array = new Float32Array(float16View.length);
    
    for (let i = 0; i < float16View.length; i++) {
        float32Array[i] = float16ToFloat32(float16View[i]);
    }
    
    // Reshape to 2D array
    const shape = encoded.shape;
    const result = [];
    for (let i = 0; i < shape[0]; i++) {
        const row = [];
        for (let j = 0; j < shape[1]; j++) {
            row.push(float32Array[i * shape[1] + j]);
        }
        result.push(row);
    }
    return result;
}

function float16ToFloat32(h) {
    // Convert IEEE 754 float16 to float32
    const sign = (h & 0x8000) >> 15;
    const exponent = (h & 0x7C00) >> 10;
    const fraction = h & 0x03FF;
    
    if (exponent === 0) {
        // Subnormal or zero
        return (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
    } else if (exponent === 31) {
        // Infinity or NaN
        return fraction ? NaN : (sign ? -Infinity : Infinity);
    } else {
        // Normal
        return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }
}
```

Modify `updateVisualizations()` (line 638):

```javascript
async function updateVisualizations() {
    if (!systemInitialized) return;

    try {
        const response = await fetch('/api/state/current');
        const data = await response.json();
        
        // Check encoding type
        const isBinary = data.encoding === 'binary_float16';
        
        // Decode activity maps
        if (data.retina && data.retina.left) {
            const leftRetinaMaps = {};
            const rightRetinaMaps = {};
            
            for (const receptorType of ['rods', 'red_cones', 'green_cones', 'blue_cones']) {
                if (isBinary) {
                    leftRetinaMaps[receptorType] = decodeFloat16Array(data.retina.left[receptorType]);
                    rightRetinaMaps[receptorType] = decodeFloat16Array(data.retina.right[receptorType]);
                } else {
                    leftRetinaMaps[receptorType] = data.retina.left[receptorType];
                    rightRetinaMaps[receptorType] = data.retina.right[receptorType];
                }
            }
            
            // Use decoded maps
            drawActivityMap('retinaLeft', leftRetinaMaps[currentRetinaType], getRetinaColor(currentRetinaType), true, useLog);
            drawActivityMap('retinaRight', rightRetinaMaps[currentRetinaType], getRetinaColor(currentRetinaType), true, useLog);
        }
        
        // ... rest of function
    } catch (error) {
        console.error('Update error:', error);
    }
}
```

**Expected improvement**: 2-3x faster data transfer, lower network usage

---

### Step 4: Optimize Canvas Rendering with ImageData

Replace `drawActivityMap()` function (line 1003) with optimized version:

```javascript
function drawActivityMapOptimized(canvasId, activityMap, colorScheme, autoScale = true, useLogScale = false) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const size = activityMap.length;
    
    // Find min/max for autoscaling
    let minVal = Infinity, maxVal = -Infinity;
    if (autoScale) {
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const val = activityMap[i][j];
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }
        }
        if (maxVal - minVal < 0.001) maxVal = minVal + 0.001;
    } else {
        minVal = 0; maxVal = 1;
    }
    
    // OPTIMIZATION: Use ImageData for direct pixel manipulation
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;
    
    const scaleX = canvas.width / size;
    const scaleY = canvas.height / size;
    
    // Pre-compute colormap
    const colormap = new Array(256);
    for (let i = 0; i < 256; i++) {
        const normalized = i / 255;
        colormap[i] = getColorForScheme(normalized, colorScheme);
    }
    
    // Fill pixels (much faster than fillRect)
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const activity = activityMap[i][j];
            
            // Normalize
            let normalized;
            if (useLogScale && activity > 0) {
                const logMin = Math.log10(Math.max(minVal, 1e-6));
                const logMax = Math.log10(Math.max(maxVal, 1e-5));
                const logVal = Math.log10(Math.max(activity, 1e-6));
                normalized = (logVal - logMin) / (logMax - logMin);
                normalized = Math.max(0, Math.min(1, normalized));
            } else {
                normalized = (activity - minVal) / (maxVal - minVal);
            }
            
            const colorIdx = Math.floor(normalized * 255);
            const color = colormap[colorIdx];
            
            // Fill corresponding pixels in canvas
            const startX = Math.floor(j * scaleX);
            const startY = Math.floor(i * scaleY);
            const endX = Math.floor((j + 1) * scaleX);
            const endY = Math.floor((i + 1) * scaleY);
            
            for (let py = startY; py < endY; py++) {
                for (let px = startX; px < endX; px++) {
                    const idx = (py * canvas.width + px) * 4;
                    data[idx] = color.r;
                    data[idx + 1] = color.g;
                    data[idx + 2] = color.b;
                    data[idx + 3] = 255;
                }
            }
        }
    }
    
    // Draw all pixels at once (much faster)
    ctx.putImageData(imageData, 0, 0);
    
    return { min: minVal, max: maxVal };
}

function getColorForScheme(value, scheme) {
    const v = Math.floor(value * 255);
    
    if (scheme === 'grayscale') {
        return {r: v, g: v, b: v};
    } else if (scheme === 'red') {
        return {r: v, g: 0, b: 0};
    } else if (scheme === 'green') {
        return {r: 0, g: v, b: 0};
    } else if (scheme === 'blue') {
        return {r: 0, g: 0, b: v};
    } else if (scheme === 'viridis') {
        return {
            r: Math.floor(v * 0.7),
            g: Math.floor(v * 0.9),
            b: Math.floor(Math.max(0, 255 - v * 0.5))
        };
    } else if (scheme === 'plasma') {
        const t = value;
        return {
            r: Math.floor(255 * Math.pow(t, 0.5)),
            g: Math.floor(255 * Math.pow(t, 2)),
            b: Math.floor(255 * (1 - t))
        };
    }
    return {r: v, g: v, b: v};
}

// Update all calls to use optimized version
const drawActivityMap = drawActivityMapOptimized;
```

**Expected improvement**: 3-5x faster canvas rendering

---

### Step 5: Add Performance Monitoring

Add monitoring to server (in `temporal_server.py`):

```python
from performance_utils import perf_monitor

# Add endpoint to get performance stats
@app.route('/api/performance/stats', methods=['GET'])
def get_performance_stats():
    """Get performance statistics from monitor"""
    return jsonify(perf_monitor.get_stats())

# Add periodic reporting
import atexit

def print_performance_report():
    perf_monitor.print_report()

atexit.register(print_performance_report)
```

---

## Testing the Optimizations

### 1. Run benchmark again
```bash
python benchmark_performance.py
```

Expected results after Phase 1:
```
PERFORMANCE COMPARISON
======================================================================
Metric                   Original    Optimized     Speedup
----------------------------------------------------------------------
Initialization             2.50s       2.80s         0.9x  (slightly slower due to indexing)
Image Processing          15.30ms      3.20ms         4.8x  ✅
Neural Update             45.20ms     12.50ms         3.6x  ✅
Activity Maps (cold)      22.10ms      8.30ms         2.7x  ✅

Effective FPS:              6.7         22.1          3.3x  ✅
======================================================================
```

### 2. Test in browser
```bash
python temporal_server.py
# Open http://localhost:5001
# Click "Initialize System"
# Observe FPS counter
```

Expected: FPS should increase from ~6-10 to ~20-30

---

## Troubleshooting

### Issue: "No module named 'scipy'"
```bash
pip install scipy
```

### Issue: Float16 decoding errors in browser
Check console for errors. Float16 conversion is tricky. If issues persist, use float32:
```python
encoder.encode_array(array, use_float16=False)  # Fall back to float32
```

### Issue: Activity maps look wrong
The optimized version should produce identical output. If not:
1. Run `benchmark_performance.py` to compare numerically
2. Check cache invalidation is working
3. Verify interpolation weights are correct

---

## Next Steps (Phase 2 - GPU)

After Phase 1 is working, install GPU libraries:

```bash
# For Apple Silicon (M5)
pip install torch>=2.0.0  # Includes MPS (Metal Performance Shaders) support

# Test GPU
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

Then implement GPU-accelerated updates (see PERFORMANCE_OPTIMIZATION_STRATEGY.md Phase 2).

---

## Monitoring Performance in Production

Add this to your initialization:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Log slow operations
from performance_utils import benchmark_function

# All decorated functions will be timed
# Check performance report on shutdown
```

---

## Summary

Phase 1 optimizations should provide:
- **3-5x** speedup in retina updates (vectorization)
- **2-3x** reduction in network transfer (binary encoding)
- **3-5x** faster frontend rendering (ImageData)
- **Overall: 5-10x end-to-end speedup**

This should be enough to reach 30 FPS with current grid size (64×64).

For further speedup (100x+), proceed to Phase 2 (GPU acceleration).

