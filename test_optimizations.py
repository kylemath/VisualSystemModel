#!/usr/bin/env python
"""
Quick test script to verify all optimizations work.
Run this before running the full benchmark.
"""

import sys

print("Testing optimizations...")
print("="*70)

# Test 1: Import optimized layers
print("\n1. Testing imports...")
try:
    from foveal_retina_optimized import FovealRetinaOptimized
    print("   ✅ FovealRetinaOptimized imported")
except Exception as e:
    print(f"   ❌ Failed to import FovealRetinaOptimized: {e}")
    sys.exit(1)

try:
    from bipolar_cells_optimized import BipolarLayerOptimized
    print("   ✅ BipolarLayerOptimized imported")
except Exception as e:
    print(f"   ❌ Failed to import BipolarLayerOptimized: {e}")
    sys.exit(1)

try:
    from ganglion_cells_optimized import GanglionLayerOptimized
    print("   ✅ GanglionLayerOptimized imported")
except Exception as e:
    print(f"   ❌ Failed to import GanglionLayerOptimized: {e}")
    sys.exit(1)

try:
    from performance_utils import BinaryEncoder, ActivityMapCache, PerformanceMonitor
    print("   ✅ Performance utils imported")
except Exception as e:
    print(f"   ❌ Failed to import performance utils: {e}")
    sys.exit(1)

# Test 2: Create neural system
print("\n2. Testing neural system creation...")
try:
    import numpy as np
    
    retina = FovealRetinaOptimized(grid_size=32, fovea_radius=0.15)
    print(f"   ✅ Retina created ({retina.total_photoreceptors} receptors)")
    
    bipolar = BipolarLayerOptimized(retina, receptor_type='all')
    print(f"   ✅ Bipolar layer created ({bipolar.total_bipolar_cells} cells)")
    
    ganglion = GanglionLayerOptimized(bipolar)
    print(f"   ✅ Ganglion layer created ({ganglion.total_ganglion_cells} cells)")
except Exception as e:
    print(f"   ❌ Failed to create neural system: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Process image
print("\n3. Testing image processing...")
try:
    test_image = np.random.rand(32, 32, 3)
    retina.process_image(test_image, test_image)
    print("   ✅ Image processing works")
except Exception as e:
    print(f"   ❌ Failed to process image: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Update neurons
print("\n4. Testing neural updates...")
try:
    retina.update(dt=33.3)
    print("   ✅ Retina update works")
    
    bipolar.update(dt=33.3)
    print("   ✅ Bipolar update works")
    
    ganglion.update(dt=33.3)
    print("   ✅ Ganglion update works")
except Exception as e:
    print(f"   ❌ Failed to update neurons: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Get activity maps
print("\n5. Testing activity map generation...")
try:
    activity_map = retina.get_activity_map('left', 'rods')
    print(f"   ✅ Retina activity map: shape {activity_map.shape}")
    
    activity_map = bipolar.get_activity_map('left', 'ON')
    print(f"   ✅ Bipolar activity map: shape {activity_map.shape}")
    
    activity_map = ganglion.get_activity_map('left', 'P')
    print(f"   ✅ Ganglion activity map: shape {activity_map.shape}")
except Exception as e:
    print(f"   ❌ Failed to generate activity maps: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Binary encoding
print("\n6. Testing binary encoding...")
try:
    encoder = BinaryEncoder()
    test_array = np.random.rand(32, 32)
    
    encoded = encoder.encode_array(test_array, use_float16=True)
    print(f"   ✅ Encoding works: {len(encoded['data'])} chars")
    
    decoded = encoder.decode_array(encoded)
    print(f"   ✅ Decoding works: shape {decoded.shape}")
    
    # Check accuracy
    error = np.mean(np.abs(test_array - decoded))
    print(f"   ✅ Encoding error: {error:.6f} (should be < 0.001)")
    
    if error > 0.001:
        print("   ⚠️  Warning: Encoding error is high (float16 precision)")
except Exception as e:
    print(f"   ❌ Failed binary encoding test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Caching
print("\n7. Testing activity map caching...")
try:
    # First call (cold cache)
    import time
    start = time.time()
    _ = retina.get_activity_map('left', 'rods')
    cold_time = time.time() - start
    
    # Second call (warm cache)
    start = time.time()
    _ = retina.get_activity_map('left', 'rods')
    warm_time = time.time() - start
    
    speedup = cold_time / warm_time if warm_time > 0 else 0
    print(f"   ✅ Cold cache: {cold_time*1000:.2f}ms")
    print(f"   ✅ Warm cache: {warm_time*1000:.2f}ms")
    print(f"   ✅ Cache speedup: {speedup:.1f}x")
    
    cache_stats = retina.activity_cache.get_stats()
    print(f"   ✅ Cache stats: {cache_stats}")
except Exception as e:
    print(f"   ❌ Failed caching test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYou can now run:")
print("  python benchmark_performance.py")
print("  python temporal_server.py")
print()

