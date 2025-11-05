"""
Benchmark script to measure performance improvements.
Compare original vs optimized implementations.
"""

import numpy as np
import time
import sys
from typing import Dict, Any

# Import original
from foveal_retina import FovealRetina
from bipolar_cells import BipolarLayer
from ganglion_cells import GanglionLayer

# Import optimized
try:
    from foveal_retina_optimized import FovealRetinaOptimized
    OPTIMIZED_AVAILABLE = True
except ImportError:
    print("Warning: Optimized version not available")
    OPTIMIZED_AVAILABLE = False

from performance_utils import perf_monitor


def create_test_image(size: int = 64) -> np.ndarray:
    """Create test RGB image"""
    image = np.zeros((size, size, 3))
    
    # Create a gradient pattern
    for i in range(size):
        for j in range(size):
            image[i, j, 0] = i / size  # Red gradient
            image[i, j, 1] = j / size  # Green gradient
            image[i, j, 2] = (i + j) / (2 * size)  # Blue gradient
    
    return image


def benchmark_original(grid_size: int = 64, num_updates: int = 100) -> Dict[str, Any]:
    """Benchmark original implementation"""
    print(f"\n{'='*70}")
    print("BENCHMARKING ORIGINAL IMPLEMENTATION")
    print(f"{'='*70}")
    
    results = {}
    
    # Initialization
    print("\n1. Initialization...")
    start = time.perf_counter()
    retina = FovealRetina(grid_size=grid_size, fovea_radius=0.15)
    bipolar_layer = BipolarLayer(retina, receptor_type='all')
    ganglion_layer = GanglionLayer(bipolar_layer)
    init_time = time.perf_counter() - start
    results['init_time'] = init_time
    print(f"   Initialization: {init_time:.3f}s")
    
    # Create test images
    left_img = create_test_image(grid_size)
    right_img = create_test_image(grid_size)
    
    # Benchmark process_image
    print("\n2. Image Processing...")
    times = []
    for i in range(10):
        start = time.perf_counter()
        retina.process_image(left_img, right_img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if i == 0:
            print(f"   First call: {elapsed*1000:.2f}ms")
    
    results['process_image_mean'] = np.mean(times) * 1000
    results['process_image_std'] = np.std(times) * 1000
    print(f"   Mean: {results['process_image_mean']:.2f}ms ± {results['process_image_std']:.2f}ms")
    
    # Benchmark update
    print("\n3. Neural Updates...")
    times = []
    for i in range(num_updates):
        start = time.perf_counter()
        retina.update(dt=33.3)  # 30 FPS
        bipolar_layer.update(dt=33.3)
        ganglion_layer.update(dt=33.3)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if i == 0:
            print(f"   First update: {elapsed*1000:.2f}ms")
    
    results['update_mean'] = np.mean(times) * 1000
    results['update_std'] = np.std(times) * 1000
    results['update_fps'] = 1000 / results['update_mean']
    print(f"   Mean: {results['update_mean']:.2f}ms ± {results['update_std']:.2f}ms")
    print(f"   Effective FPS: {results['update_fps']:.1f}")
    
    # Benchmark activity map generation
    print("\n4. Activity Map Generation...")
    times = []
    for i in range(20):
        start = time.perf_counter()
        _ = retina.get_activity_map('left', 'rods')
        _ = retina.get_activity_map('left', 'red_cones')
        _ = retina.get_activity_map('left', 'green_cones')
        _ = retina.get_activity_map('left', 'blue_cones')
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if i == 0:
            print(f"   First call (4 maps): {elapsed*1000:.2f}ms")
    
    results['activity_map_mean'] = np.mean(times) * 1000
    results['activity_map_std'] = np.std(times) * 1000
    print(f"   Mean: {results['activity_map_mean']:.2f}ms ± {results['activity_map_std']:.2f}ms")
    
    # Benchmark full frame (image + update + visualization)
    print("\n5. Full Frame (Process + Update + Viz)...")
    times = []
    for i in range(50):
        start = time.perf_counter()
        
        # Process new image
        retina.process_image(left_img, right_img)
        
        # Update neurons
        retina.update(dt=33.3)
        bipolar_layer.update(dt=33.3)
        ganglion_layer.update(dt=33.3)
        
        # Generate activity maps (as server would)
        for eye in ['left', 'right']:
            for rtype in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
                _ = retina.get_activity_map(eye, rtype)
            for btype in ['ON', 'OFF']:
                _ = bipolar_layer.get_activity_map(eye, btype)
            for gtype in ['P', 'M']:
                _ = ganglion_layer.get_activity_map(eye, gtype)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if i == 0:
            print(f"   First frame: {elapsed*1000:.2f}ms")
    
    results['full_frame_mean'] = np.mean(times) * 1000
    results['full_frame_std'] = np.std(times) * 1000
    results['full_frame_fps'] = 1000 / results['full_frame_mean']
    print(f"   Mean: {results['full_frame_mean']:.2f}ms ± {results['full_frame_std']:.2f}ms")
    print(f"   Effective FPS: {results['full_frame_fps']:.1f}")
    
    # Memory usage estimate
    import sys
    total_receptors = retina.total_photoreceptors
    total_bipolar = bipolar_layer.total_bipolar_cells
    total_ganglion = ganglion_layer.total_ganglion_cells
    
    # Rough estimate: ~500 bytes per neuron object
    estimated_memory = (total_receptors + total_bipolar + total_ganglion) * 500 / (1024 * 1024)
    results['estimated_memory_mb'] = estimated_memory
    
    print(f"\n6. System Statistics:")
    print(f"   Photoreceptors: {total_receptors:,}")
    print(f"   Bipolar cells: {total_bipolar:,}")
    print(f"   Ganglion cells: {total_ganglion:,}")
    print(f"   Estimated memory: {estimated_memory:.1f} MB")
    
    return results


def benchmark_optimized(grid_size: int = 64, num_updates: int = 100) -> Dict[str, Any]:
    """Benchmark optimized implementation"""
    if not OPTIMIZED_AVAILABLE:
        print("Optimized version not available, skipping benchmark")
        return {}
    
    print(f"\n{'='*70}")
    print("BENCHMARKING OPTIMIZED IMPLEMENTATION (FULL SYSTEM)")
    print(f"{'='*70}")
    
    results = {}
    
    # Initialization
    print("\n1. Initialization...")
    start = time.perf_counter()
    retina = FovealRetinaOptimized(grid_size=grid_size, fovea_radius=0.15)
    
    # Import and use optimized bipolar and ganglion layers
    try:
        from bipolar_cells_optimized import BipolarLayerOptimized
        from ganglion_cells_optimized import GanglionLayerOptimized
        bipolar_layer = BipolarLayerOptimized(retina, receptor_type='all')
        ganglion_layer = GanglionLayerOptimized(bipolar_layer)
        has_full_system = True
        print(f"   ✅ Using FULL optimized system (retina + bipolar + ganglion)")
    except ImportError as e:
        print(f"   ⚠️  Could not import optimized bipolar/ganglion: {e}")
        print(f"   Testing retina only")
        bipolar_layer = None
        ganglion_layer = None
        has_full_system = False
    
    init_time = time.perf_counter() - start
    results['init_time'] = init_time
    print(f"   Initialization: {init_time:.3f}s")
    
    # Create test images
    left_img = create_test_image(grid_size)
    right_img = create_test_image(grid_size)
    
    # Benchmark process_image
    print("\n2. Image Processing...")
    times = []
    for i in range(10):
        start = time.perf_counter()
        retina.process_image(left_img, right_img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if i == 0:
            print(f"   First call: {elapsed*1000:.2f}ms")
    
    results['process_image_mean'] = np.mean(times) * 1000
    results['process_image_std'] = np.std(times) * 1000
    print(f"   Mean: {results['process_image_mean']:.2f}ms ± {results['process_image_std']:.2f}ms")
    
    # Benchmark update
    print("\n3. Neural Updates...")
    times = []
    for i in range(num_updates):
        start = time.perf_counter()
        retina.update(dt=33.3)
        if has_full_system:
            bipolar_layer.update(dt=33.3)
            ganglion_layer.update(dt=33.3)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if i == 0:
            if has_full_system:
                print(f"   First update (retina + bipolar + ganglion): {elapsed*1000:.2f}ms")
            else:
                print(f"   First update (retina only): {elapsed*1000:.2f}ms")
    
    results['update_mean'] = np.mean(times) * 1000
    results['update_std'] = np.std(times) * 1000
    results['update_fps'] = 1000 / results['update_mean']
    print(f"   Mean: {results['update_mean']:.2f}ms ± {results['update_std']:.2f}ms")
    print(f"   Effective FPS: {results['update_fps']:.1f}")
    
    # Benchmark activity map generation (with caching)
    print("\n4. Activity Map Generation (with caching)...")
    
    # First call (cold cache)
    print("   Cold cache (first call):")
    times_cold = []
    for i in range(5):
        retina.activity_cache.invalidate_all()  # Force cache miss
        if has_full_system:
            bipolar_layer.activity_cache.invalidate_all()
            ganglion_layer.activity_cache.invalidate_all()
        
        start = time.perf_counter()
        # Retina maps
        _ = retina.get_activity_map('left', 'rods')
        _ = retina.get_activity_map('left', 'red_cones')
        _ = retina.get_activity_map('left', 'green_cones')
        _ = retina.get_activity_map('left', 'blue_cones')
        # Bipolar maps
        if has_full_system:
            _ = bipolar_layer.get_activity_map('left', 'ON')
            _ = bipolar_layer.get_activity_map('left', 'OFF')
            _ = ganglion_layer.get_activity_map('left', 'P')
            _ = ganglion_layer.get_activity_map('left', 'M')
        elapsed = time.perf_counter() - start
        times_cold.append(elapsed)
    
    results['activity_map_cold_mean'] = np.mean(times_cold) * 1000
    if has_full_system:
        print(f"     Mean (8 maps): {results['activity_map_cold_mean']:.2f}ms")
    else:
        print(f"     Mean (4 maps, retina only): {results['activity_map_cold_mean']:.2f}ms")
    
    # Warm cache
    print("   Warm cache (cached calls):")
    times_warm = []
    for i in range(20):
        start = time.perf_counter()
        # Retina maps
        _ = retina.get_activity_map('left', 'rods')
        _ = retina.get_activity_map('left', 'red_cones')
        _ = retina.get_activity_map('left', 'green_cones')
        _ = retina.get_activity_map('left', 'blue_cones')
        # Bipolar and ganglion maps
        if has_full_system:
            _ = bipolar_layer.get_activity_map('left', 'ON')
            _ = bipolar_layer.get_activity_map('left', 'OFF')
            _ = ganglion_layer.get_activity_map('left', 'P')
            _ = ganglion_layer.get_activity_map('left', 'M')
        elapsed = time.perf_counter() - start
        times_warm.append(elapsed)
    
    results['activity_map_warm_mean'] = np.mean(times_warm) * 1000
    print(f"     Mean: {results['activity_map_warm_mean']:.2f}ms")
    if results['activity_map_warm_mean'] > 0:
        print(f"   Cache speedup: {results['activity_map_cold_mean'] / results['activity_map_warm_mean']:.1f}x")
    
    # Cache statistics
    cache_stats = retina.activity_cache.get_stats()
    print(f"\n5. Cache Statistics:")
    print(f"   Cached items: {cache_stats['num_cached']}")
    print(f"   Total size: {cache_stats['total_size_mb']:.2f} MB")
    
    # Performance monitor report
    print("\n6. Performance Monitor:")
    perf_monitor.print_report()
    
    return results


def compare_results(original: Dict, optimized: Dict):
    """Compare and print speedup metrics"""
    if not optimized:
        print("\nNo optimized results to compare")
        return
    
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}\n")
    
    metrics = [
        ('Initialization', 'init_time', 's'),
        ('Image Processing', 'process_image_mean', 'ms'),
        ('Neural Update', 'update_mean', 'ms'),
        ('Activity Maps (cold)', 'activity_map_cold_mean', 'ms'),
    ]
    
    print(f"{'Metric':<25} {'Original':>12} {'Optimized':>12} {'Speedup':>10}")
    print("-"*70)
    
    for name, key, unit in metrics:
        if key in original and key in optimized:
            orig_val = original[key]
            opt_val = optimized[key]
            speedup = orig_val / opt_val
            
            print(f"{name:<25} {orig_val:>11.2f}{unit} {opt_val:>11.2f}{unit} {speedup:>9.1f}x")
    
    # FPS comparison
    if 'update_fps' in original and 'update_fps' in optimized:
        print(f"\n{'Effective FPS:':<25} {original['update_fps']:>11.1f}    {optimized['update_fps']:>11.1f}    {optimized['update_fps']/original['update_fps']:>9.1f}x")
    
    print("\n" + "="*70)


def main():
    """Run benchmarks"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          NEURAL SYSTEM PERFORMANCE BENCHMARK SUITE               ║
║                                                                  ║
║  This benchmark compares original vs optimized implementations   ║
║  to measure the impact of Phase 1 optimizations.                 ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    grid_size = 64
    num_updates = 100
    
    print(f"Configuration:")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Number of updates: {num_updates}")
    print(f"  Target FPS: 30")
    print()
    
    # Run benchmarks
    original_results = benchmark_original(grid_size, num_updates)
    
    if OPTIMIZED_AVAILABLE:
        optimized_results = benchmark_optimized(grid_size, num_updates)
        compare_results(original_results, optimized_results)
    else:
        print("\nOptimized implementation not available yet.")
        print("Install scipy and run again after implementing optimizations.")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    target_fps = 30
    target_frame_time = 1000 / target_fps
    
    if 'full_frame_mean' in original_results:
        current_fps = original_results['full_frame_fps']
        frame_time = original_results['full_frame_mean']
        
        print(f"Current Performance:")
        print(f"  Full frame time: {frame_time:.1f}ms")
        print(f"  Effective FPS: {current_fps:.1f}")
        print(f"  Target FPS: {target_fps}")
        print()
        
        if current_fps < target_fps:
            speedup_needed = target_frame_time / frame_time
            print(f"⚠️  Need {1/speedup_needed:.1f}x speedup to reach {target_fps} FPS")
            print()
            print("Recommended optimizations:")
            print("  1. ✅ Use optimized retina (FovealRetinaOptimized)")
            print("  2. ⏳ Implement binary encoding for network transfer")
            print("  3. ⏳ Add activity map caching to server")
            print("  4. ⏳ Optimize frontend rendering (ImageData API)")
            print("  5. ⏳ Consider GPU acceleration (Phase 2)")
        else:
            print(f"✅ System meets {target_fps} FPS target!")
            print(f"   Consider increasing grid size or adding more brain areas")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()

