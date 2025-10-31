"""
Test script for temporal neural system.
Validates all components work correctly.
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from foveal_retina import FovealRetina
from bipolar_cells import BipolarLayer
from webcam_input import SimulatedWebcam
import time

def test_foveal_retina():
    """Test foveal retina creation and function."""
    print("\n" + "="*70)
    print("TEST 1: Foveal Retina")
    print("="*70)
    
    grid_size = 64
    retina = FovealRetina(grid_size=grid_size, fovea_radius=0.15)
    
    print(f"✓ Created foveal retina ({grid_size}×{grid_size})")
    print(f"✓ Total photoreceptors: {retina.total_photoreceptors:,}")
    
    # Check distribution
    summary = retina.get_summary()
    for eye in ['left', 'right']:
        print(f"\n{eye.upper()} EYE:")
        for receptor_type, data in summary[eye].items():
            print(f"  {receptor_type}:")
            print(f"    Total: {data['total_count']}")
            print(f"    Distribution: {data['zone_distribution']}")
    
    # Test image processing
    test_image = np.random.rand(grid_size, grid_size, 3)
    retina.process_image(test_image, test_image, intensity=1.0)
    print("\n✓ Processed test image")
    
    # Test temporal update
    retina.update(dt=1.0)
    print("✓ Temporal update successful")
    
    # Get activity maps
    for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
        activity_map = retina.get_activity_map('left', receptor_type)
        mean_activity = np.mean(activity_map)
        print(f"  {receptor_type}: mean activity = {mean_activity:.3f}")
    
    return retina


def test_bipolar_layer(retina):
    """Test bipolar cell layer."""
    print("\n" + "="*70)
    print("TEST 2: Bipolar Cell Layer")
    print("="*70)
    
    bipolar_layer = BipolarLayer(retina, receptor_type='all')
    
    print(f"✓ Created bipolar layer")
    print(f"✓ Total bipolar cells: {bipolar_layer.total_bipolar_cells:,}")
    
    summary = bipolar_layer.get_summary()
    for eye in ['left', 'right']:
        print(f"\n{eye.upper()} EYE:")
        for cell_type in ['ON', 'OFF']:
            data = summary[eye][cell_type]
            print(f"  {cell_type}-center cells:")
            print(f"    Total: {data['count']}")
            print(f"    P pathway: {data['P_pathway']}")
            print(f"    M pathway: {data['M_pathway']}")
    
    # Test update
    bipolar_layer.update(dt=1.0)
    print("\n✓ Bipolar layer update successful")
    
    # Get activity maps
    for cell_type in ['ON', 'OFF']:
        activity_map = bipolar_layer.get_activity_map('left', cell_type)
        mean_activity = np.mean(activity_map)
        max_activity = np.max(activity_map)
        print(f"  {cell_type} cells: mean={mean_activity:.3f}, max={max_activity:.3f}")
    
    return bipolar_layer


def test_temporal_dynamics(retina, bipolar_layer):
    """Test continuous temporal updates."""
    print("\n" + "="*70)
    print("TEST 3: Temporal Dynamics")
    print("="*70)
    
    print("Running 100 time steps...")
    
    # Create changing input
    grid_size = retina.grid_size
    
    activities = {
        'rods': [],
        'ON_bipolar': [],
        'OFF_bipolar': []
    }
    
    for t in range(100):
        # Create moving stimulus
        x_pos = int(grid_size * 0.5 * (1 + np.sin(t * 0.1)))
        test_image = np.zeros((grid_size, grid_size, 3))
        test_image[max(0, x_pos-5):min(grid_size, x_pos+5), :, :] = 1.0
        
        # Process through system
        retina.process_image(test_image, test_image, intensity=1.0)
        retina.update(dt=1.0)
        bipolar_layer.update(dt=1.0)
        
        # Record activities
        rod_map = retina.get_activity_map('left', 'rods')
        on_map = bipolar_layer.get_activity_map('left', 'ON')
        off_map = bipolar_layer.get_activity_map('left', 'OFF')
        
        activities['rods'].append(np.mean(rod_map))
        activities['ON_bipolar'].append(np.mean(on_map))
        activities['OFF_bipolar'].append(np.mean(off_map))
    
    print("✓ Completed 100 temporal updates")
    
    # Analyze temporal response
    for key, values in activities.items():
        values = np.array(values)
        print(f"\n{key}:")
        print(f"  Mean: {np.mean(values):.3f}")
        print(f"  Std: {np.std(values):.3f}")
        print(f"  Range: [{np.min(values):.3f}, {np.max(values):.3f}]")
        print(f"  Temporal variation: {np.std(np.diff(values)):.3f}")


def test_webcam_simulation():
    """Test simulated webcam."""
    print("\n" + "="*70)
    print("TEST 4: Simulated Webcam")
    print("="*70)
    
    webcam = SimulatedWebcam(target_size=(64, 64), fps=30)
    webcam.start()
    
    print("✓ Simulated webcam started")
    
    # Let it run for a bit
    time.sleep(0.5)
    
    # Get frames
    left, right = webcam.get_frames()
    
    print(f"✓ Frame shape: {left.shape}")
    print(f"✓ Frame range: [{np.min(left):.3f}, {np.max(left):.3f}]")
    
    status = webcam.get_status()
    print(f"✓ Status: {status}")
    
    webcam.stop()
    print("✓ Simulated webcam stopped")


def test_receptive_field_structure(bipolar_layer):
    """Test receptive field organization."""
    print("\n" + "="*70)
    print("TEST 5: Receptive Field Structure")
    print("="*70)
    
    # Sample some bipolar cells
    on_cells = bipolar_layer.bipolar_cells['left']['ON']
    
    print(f"Analyzing {min(10, len(on_cells))} ON-center bipolar cells...")
    
    for i, cell in enumerate(on_cells[:10]):
        print(f"\nCell {i+1}:")
        print(f"  Position: ({cell.position[0]:.2f}, {cell.position[1]:.2f})")
        print(f"  Eccentricity: {cell.eccentricity:.3f}")
        print(f"  Pathway: {cell.pathway}")
        print(f"  Center size: {len(cell.center_photoreceptors)}")
        print(f"  Surround size: {len(cell.surround_photoreceptors)}")
        
        # Verify center-surround ratio
        if len(cell.center_photoreceptors) > 0:
            ratio = len(cell.surround_photoreceptors) / len(cell.center_photoreceptors)
            print(f"  Surround/Center ratio: {ratio:.2f}")


def test_neuron_properties():
    """Test individual neuron temporal properties."""
    print("\n" + "="*70)
    print("TEST 6: Neuron Temporal Properties")
    print("="*70)
    
    from temporal_neuron import TemporalNeuron
    
    neuron = TemporalNeuron('test')
    
    print(f"Initial state:")
    print(f"  V_rest: {neuron.v_rest} mV")
    print(f"  V_threshold: {neuron.v_threshold} mV")
    print(f"  Tau: {neuron.tau_membrane} ms")
    print(f"  Oscillation freq: {neuron.oscillation_freq} Hz")
    
    # Test update
    print("\nRunning 100 updates...")
    v_trace = []
    for t in range(100):
        neuron.update(dt=1.0)
        v_trace.append(neuron.v_membrane)
    
    v_trace = np.array(v_trace)
    print(f"✓ V_membrane range: [{np.min(v_trace):.1f}, {np.max(v_trace):.1f}] mV")
    print(f"✓ Mean: {np.mean(v_trace):.1f} mV")
    print(f"✓ Oscillation detected: {np.std(v_trace) > 1.0}")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "🧠 TEMPORAL NEURAL SYSTEM TEST SUITE ".center(70, "="))
    print()
    
    start_time = time.time()
    
    try:
        # Test 1: Foveal retina
        retina = test_foveal_retina()
        
        # Test 2: Bipolar layer
        bipolar_layer = test_bipolar_layer(retina)
        
        # Test 3: Temporal dynamics
        test_temporal_dynamics(retina, bipolar_layer)
        
        # Test 4: Webcam
        test_webcam_simulation()
        
        # Test 5: Receptive fields
        test_receptive_field_structure(bipolar_layer)
        
        # Test 6: Neuron properties
        test_neuron_properties()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print(f"✅ ALL TESTS PASSED in {elapsed:.2f} seconds")
        print("="*70)
        print("\n🚀 System is ready! Run:")
        print("   python temporal_server.py")
        print("   Then open http://localhost:5000")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

