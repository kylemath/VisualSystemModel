"""
Flask server for temporal neural network with webcam input.
Supports continuous time dynamics and 3D visualization.
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import numpy as np
import json
import time
import threading

from foveal_retina import FovealRetina
from bipolar_cells import BipolarLayer
from ganglion_cells import GanglionLayer
from lateral_cells import LateralProcessingLayer
from foveal_input_system import FovealInputSystem
from visualization_2d import (NeuralVisualizer2D, generate_safe_json,
                             create_network_summary_2d)

app = Flask(__name__)
CORS(app)

# Global neural system
neural_system = {
    'retina': None,
    'bipolar_layer': None,
    'ganglion_layer': None,
    'lateral_layer': None,
    'foveal_input': None,  # Replaces 'webcam', includes eye movements
    'visualizer': NeuralVisualizer2D(),
    'is_running': False,
    'update_thread': None,
    'fps': 30
}


def update_loop():
    """Continuous update loop for temporal dynamics."""
    dt = 1000.0 / neural_system['fps']  # ms per frame
    
    while neural_system['is_running']:
        loop_start = time.time()
        
        # Update eye movements and get foveal frames
        if neural_system['foveal_input']:
            # Update eye movements (microsaccades, drift, tremor)
            neural_system['foveal_input'].update(dt)
            
            # Get foveal frames from eye positions
            left_frame, right_frame = neural_system['foveal_input'].get_foveal_frames()
            
            # Process through retina
            if neural_system['retina']:
                neural_system['retina'].process_image(left_frame, right_frame)
                neural_system['retina'].update(dt)
            
            # Update bipolar cells
            if neural_system['bipolar_layer']:
                neural_system['bipolar_layer'].update(dt)
            
            # Update lateral processing (horizontal and amacrine cells)
            if neural_system['lateral_layer']:
                neural_system['lateral_layer'].update(dt)
                neural_system['lateral_layer'].apply_lateral_modulation()
            
            # Update ganglion cells
            if neural_system['ganglion_layer']:
                neural_system['ganglion_layer'].update(dt)
        
        # Frame rate control
        elapsed = (time.time() - loop_start) * 1000
        sleep_time = max(0, dt - elapsed) / 1000.0
        if sleep_time > 0:
            time.sleep(sleep_time)


@app.route('/')
def index():
    """Serve main temporal interface."""
    return render_template('temporal_index.html')


@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize neural system."""
    data = request.json
    grid_size = data.get('grid_size', 64)
    fovea_radius = data.get('fovea_radius', 0.15)
    use_real_webcam = data.get('use_real_webcam', False)
    fps = data.get('fps', 30)
    
    neural_system['fps'] = fps
    
    try:
        # Create retina
        print(f"Creating foveal retina ({grid_size}x{grid_size})...")
        neural_system['retina'] = FovealRetina(grid_size=grid_size, 
                                               fovea_radius=fovea_radius)
        
        # Create bipolar layer
        print("Creating bipolar cell layer...")
        neural_system['bipolar_layer'] = BipolarLayer(
            neural_system['retina'],
            receptor_type='all'
        )
        
        # Create ganglion layer
        print("Creating ganglion cell layer...")
        neural_system['ganglion_layer'] = GanglionLayer(
            neural_system['bipolar_layer']
        )
        
        # Create lateral processing layer (horizontal and amacrine cells)
        print("Creating lateral processing layer...")
        neural_system['lateral_layer'] = LateralProcessingLayer(
            neural_system['retina'],
            neural_system['bipolar_layer'],
            neural_system['ganglion_layer']
        )
        
        # Initialize foveal input system (webcam + eye movements)
        print("Initializing foveal input system with eye movements...")
        neural_system['foveal_input'] = FovealInputSystem(
            use_real_webcam=use_real_webcam,
            webcam_size=(320, 240),  # Full visual field
            foveal_size=(grid_size, grid_size),  # High-res fovea
            camera_index=0
        )
        
        # Start update loop
        neural_system['is_running'] = True
        neural_system['update_thread'] = threading.Thread(
            target=update_loop,
            daemon=True
        )
        neural_system['update_thread'].start()
        
        # Get initial state
        retina_summary = neural_system['retina'].get_summary()
        bipolar_summary = neural_system['bipolar_layer'].get_summary()
        ganglion_summary = neural_system['ganglion_layer'].get_summary()
        lateral_summary = neural_system['lateral_layer'].get_summary()
        
        eye_state = neural_system['foveal_input'].get_eye_state()
        
        return jsonify({
            'status': 'success',
            'message': 'System initialized successfully (with eye movements)',
            'system_info': {
                'retina': retina_summary,
                'bipolar_layer': bipolar_summary,
                'ganglion_layer': ganglion_summary,
                'lateral_layer': lateral_summary,
                'eye_movements': eye_state
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/shutdown', methods=['POST'])
def shutdown_system():
    """Shutdown neural system."""
    try:
        neural_system['is_running'] = False
        
        if neural_system['foveal_input']:
            neural_system['foveal_input'].stop()
        
        if neural_system['update_thread']:
            neural_system['update_thread'].join(timeout=2.0)
        
        return jsonify({'status': 'success', 'message': 'System shut down'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/state/current', methods=['GET'])
def get_current_state():
    """Get current state of all layers."""
    if not neural_system['retina']:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        # Get activity maps
        left_retina_maps = {}
        right_retina_maps = {}
        
        for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
            left_retina_maps[receptor_type] = neural_system['retina'].get_activity_map(
                'left', receptor_type
            ).tolist()
            right_retina_maps[receptor_type] = neural_system['retina'].get_activity_map(
                'right', receptor_type
            ).tolist()
        
        # Get bipolar activity
        left_bipolar_on = None
        left_bipolar_off = None
        right_bipolar_on = None
        right_bipolar_off = None
        
        if neural_system['bipolar_layer']:
            left_bipolar_on = neural_system['bipolar_layer'].get_activity_map(
                'left', 'ON'
            ).tolist()
            left_bipolar_off = neural_system['bipolar_layer'].get_activity_map(
                'left', 'OFF'
            ).tolist()
            right_bipolar_on = neural_system['bipolar_layer'].get_activity_map(
                'right', 'ON'
            ).tolist()
            right_bipolar_off = neural_system['bipolar_layer'].get_activity_map(
                'right', 'OFF'
            ).tolist()
        
        # Get ganglion activity
        left_ganglion_p = None
        left_ganglion_m = None
        right_ganglion_p = None
        right_ganglion_m = None
        
        if neural_system['ganglion_layer']:
            left_ganglion_p = neural_system['ganglion_layer'].get_activity_map(
                'left', 'P'
            ).tolist()
            left_ganglion_m = neural_system['ganglion_layer'].get_activity_map(
                'left', 'M'
            ).tolist()
            right_ganglion_p = neural_system['ganglion_layer'].get_activity_map(
                'right', 'P'
            ).tolist()
            right_ganglion_m = neural_system['ganglion_layer'].get_activity_map(
                'right', 'M'
            ).tolist()
        
        # Get current input frames (retinal regions)
        left_input, right_input = None, None
        full_left_frame, full_right_frame = None, None
        eye_regions = None
        
        if neural_system['foveal_input']:
            try:
                left_input, right_input = neural_system['foveal_input'].get_foveal_frames()
                if left_input is not None and right_input is not None:
                    left_input = left_input.tolist()
                    right_input = right_input.tolist()
                
                # Get full frames for webcam visualization
                full_left_frame, full_right_frame = neural_system['foveal_input'].get_full_frames()
                if full_left_frame is not None and full_right_frame is not None:
                    full_left_frame = full_left_frame.tolist()
                    full_right_frame = full_right_frame.tolist()
                
                # Get eye bounding boxes
                eye_state = neural_system['foveal_input'].get_eye_state()
                if 'foveal_regions' in eye_state:
                    eye_regions = eye_state['foveal_regions']
                    
            except Exception as e:
                print(f"Error getting foveal frames: {e}")
                import traceback
                traceback.print_exc()
        
        return jsonify({
            'status': 'success',
            'timestamp': time.time(),
            'input': {
                'left': left_input,
                'right': right_input,
                'full_left': full_left_frame,
                'full_right': full_right_frame,
                'eye_regions': eye_regions
            },
            'retina': {
                'left': left_retina_maps,
                'right': right_retina_maps
            },
            'bipolar': {
                'left': {
                    'ON': left_bipolar_on,
                    'OFF': left_bipolar_off
                },
                'right': {
                    'ON': right_bipolar_on,
                    'OFF': right_bipolar_off
                }
            },
            'ganglion': {
                'left': {
                    'P': left_ganglion_p,
                    'M': left_ganglion_m
                },
                'right': {
                    'P': right_ganglion_p,
                    'M': right_ganglion_m
                }
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get summary statistics."""
    if not neural_system['retina']:
        return jsonify({'error': 'System not initialized'}), 400
    
    summary = {
        'retina': neural_system['retina'].get_summary(),
        'eye_movements': neural_system['foveal_input'].get_eye_state() if neural_system['foveal_input'] else None
    }
    
    if neural_system['bipolar_layer']:
        summary['bipolar_layer'] = neural_system['bipolar_layer'].get_summary()
    
    if neural_system['ganglion_layer']:
        summary['ganglion_layer'] = neural_system['ganglion_layer'].get_summary()
    
    if neural_system['lateral_layer']:
        summary['lateral_layer'] = neural_system['lateral_layer'].get_summary()
    
    return jsonify(summary)


@app.route('/api/visualize/2d/photoreceptors/<eye>', methods=['GET'])
def visualize_photoreceptors_2d(eye):
    """Get 2D visualization data for photoreceptor distribution."""
    if not neural_system['retina']:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        data = neural_system['visualizer'].visualize_photoreceptor_distribution(
            neural_system['retina'], eye
        )
        return Response(generate_safe_json(data), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize/2d/receptive_fields/<eye>', methods=['GET'])
def visualize_receptive_fields_2d(eye):
    """Get 2D visualization data for bipolar cell receptive fields."""
    if not neural_system['bipolar_layer']:
        return jsonify({'error': 'Bipolar layer not initialized'}), 400
    
    try:
        num_samples = request.args.get('samples', 10, type=int)
        data = neural_system['visualizer'].visualize_receptive_fields_2d(
            neural_system['bipolar_layer'], eye, num_samples
        )
        return Response(generate_safe_json(data), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize/2d/network_overview', methods=['GET'])
def visualize_network_overview_2d():
    """Get 2D network overview with layer information."""
    if not neural_system['retina']:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        # Create summary with all layers
        summary = {
            'layers': []
        }
        
        # Retina
        retina_summary = neural_system['retina'].get_summary()
        summary['layers'].append({
            'name': 'Retina (Photoreceptors)',
            'z_level': 0,
            'neuron_count': neural_system['retina'].total_photoreceptors,
            'types': list(retina_summary['left'].keys()),
            'stats': retina_summary
        })
        
        # Bipolar
        if neural_system['bipolar_layer']:
            bipolar_summary = neural_system['bipolar_layer'].get_summary()
            summary['layers'].append({
                'name': 'Bipolar Cells + Horizontal',
                'z_level': 1,
                'neuron_count': neural_system['bipolar_layer'].total_bipolar_cells,
                'types': ['ON', 'OFF'],
                'stats': bipolar_summary
            })
        
        # Ganglion
        if neural_system['ganglion_layer']:
            ganglion_summary = neural_system['ganglion_layer'].get_summary()
            summary['layers'].append({
                'name': 'Ganglion Cells (Optic Nerve) + Amacrine',
                'z_level': 2,
                'neuron_count': neural_system['ganglion_layer'].total_ganglion_cells,
                'types': ['P', 'M', 'ipRGC'],
                'stats': ganglion_summary
            })
        
        return Response(generate_safe_json(summary), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/neuron/<int:neuron_id>', methods=['GET'])
def get_neuron_details(neuron_id):
    """Get detailed state of specific neuron."""
    # Would search through all layers for neuron with given ID
    # Return detailed state information
    return jsonify({'status': 'not_implemented'})


@app.route('/api/set_expectation', methods=['POST'])
def set_expectation():
    """Set expectation (attention) for specific neurons or regions."""
    data = request.json
    # Would set preparatory depolarization for selected neurons
    return jsonify({'status': 'not_implemented'})


@app.route('/api/optic_nerve/<eye>', methods=['GET'])
def get_optic_nerve_output(eye):
    """
    Get optic nerve output (spiking ganglion cells).
    This is what goes to LGN.
    """
    if not neural_system['ganglion_layer']:
        return jsonify({'error': 'Ganglion layer not initialized'}), 400
    
    try:
        output = neural_system['ganglion_layer'].get_optic_nerve_output(eye)
        return Response(generate_safe_json(output), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eye_movements/state', methods=['GET'])
def get_eye_movement_state():
    """Get current eye movement state."""
    if not neural_system['foveal_input']:
        return jsonify({'error': 'Eye movement system not initialized'}), 400
    
    try:
        state = neural_system['foveal_input'].get_eye_state()
        return jsonify(state)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eye_movements/saccade', methods=['POST'])
def command_saccade():
    """
    Command a saccade to target location.
    
    POST body: {"x": float, "y": float}  (normalized coords [-1, 1])
    """
    if not neural_system['foveal_input']:
        return jsonify({'error': 'Eye movement system not initialized'}), 400
    
    try:
        data = request.json
        x = data.get('x', 0.0)
        y = data.get('y', 0.0)
        
        neural_system['foveal_input'].saccade_to(x, y)
        
        return jsonify({
            'status': 'success',
            'message': f'Saccade commanded to ({x:.2f}, {y:.2f})'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eye_movements/vergence', methods=['POST'])
def set_vergence():
    """
    Set vergence angle for depth.
    
    POST body: {"angle": float}  (degrees, 0=parallel/far, +15=converged/near)
    """
    if not neural_system['foveal_input']:
        return jsonify({'error': 'Eye movement system not initialized'}), 400
    
    try:
        data = request.json
        angle = data.get('angle', 0.0)
        
        neural_system['foveal_input'].set_vergence(angle)
        
        return jsonify({
            'status': 'success',
            'message': f'Vergence set to {angle:.1f} degrees'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/neuron/receptive_field', methods=['POST'])
def get_receptive_field():
    """
    Get receptive field information for a neuron at clicked position.
    
    POST body: {
        "layer": "bipolar" or "ganglion",
        "eye": "left" or "right",
        "x": float,  // Position in [-1, 1]
        "y": float,  // Position in [-1, 1]
        "cell_type": "ON"/"OFF" for bipolar, "P"/"M" for ganglion (optional)
    }
    """
    if not neural_system['retina']:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        data = request.json
        layer = data.get('layer')
        eye = data.get('eye')
        x = float(data.get('x'))
        y = float(data.get('y'))
        cell_type = data.get('cell_type')
        
        result = {
            'layer': layer,
            'eye': eye,
            'position': (x, y),
            'receptive_field': None,
            'output_targets': []
        }
        
        if layer == 'bipolar':
            if not neural_system['bipolar_layer']:
                return jsonify({'error': 'Bipolar layer not initialized'}), 400
            
            cell = neural_system['bipolar_layer'].find_neuron_at_position(
                eye, x, y, cell_type
            )
            
            if cell:
                rf_info = neural_system['bipolar_layer'].get_receptive_field_info(cell)
                # Subsample photoreceptor positions for performance
                # Limit to max 500 positions per region for fast rendering
                max_positions = 500
                
                if rf_info.get('center_photoreceptors'):
                    center = rf_info['center_photoreceptors']
                    if len(center) > max_positions:
                        step = len(center) // max_positions
                        rf_info['center_photoreceptors'] = center[::step]
                
                if rf_info.get('surround_photoreceptors'):
                    surround = rf_info['surround_photoreceptors']
                    if len(surround) > max_positions:
                        step = len(surround) // max_positions
                        rf_info['surround_photoreceptors'] = surround[::step]
                
                result['receptive_field'] = rf_info
                
                # Find ganglion cells that receive input from this bipolar
                if neural_system['ganglion_layer']:
                    result['output_targets'] = neural_system['ganglion_layer'].find_ganglions_connected_to_bipolar(cell)
        
        elif layer == 'ganglion':
            if not neural_system['ganglion_layer']:
                return jsonify({'error': 'Ganglion layer not initialized'}), 400
            
            cell = neural_system['ganglion_layer'].find_neuron_at_position(
                eye, x, y, cell_type
            )
            
            if cell:
                rf_info = neural_system['ganglion_layer'].get_receptive_field_info(cell)
                # Subsample bipolar positions for performance
                max_positions = 500
                
                if rf_info.get('on_bipolar_inputs'):
                    on_bipolars = rf_info['on_bipolar_inputs']
                    if len(on_bipolars) > max_positions:
                        step = len(on_bipolars) // max_positions
                        rf_info['on_bipolar_inputs'] = on_bipolars[::step]
                
                if rf_info.get('off_bipolar_inputs'):
                    off_bipolars = rf_info['off_bipolar_inputs']
                    if len(off_bipolars) > max_positions:
                        step = len(off_bipolars) // max_positions
                        rf_info['off_bipolar_inputs'] = off_bipolars[::step]
                
                result['receptive_field'] = rf_info
                # Ganglion cells are output layer, so no forward connections yet
        
        else:
            return jsonify({'error': f'Unknown layer: {layer}'}), 400
        
        if not result['receptive_field']:
            return jsonify({'error': 'No neuron found at position'}), 404
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("Bio-Inspired Temporal Neural Network Server")
    print("=" * 70)
    print("\nComplete Retinal System:")
    print("  ✓ Layer 0: Photoreceptors (Rods + RGB Cones, ~22K neurons)")
    print("  ✓         - Blind spot correctly positioned (temporal)")
    print("  ✓ Layer 1: Bipolar cells (ON/OFF, P/M pathways, ~3.7K neurons)")
    print("  ✓         + Horizontal cells (lateral inhibition, ~100 neurons)")
    print("  ✓ Layer 2: Ganglion cells (P/M/ipRGC, optic nerve, ~388 neurons)")
    print("  ✓         + Amacrine cells (motion/direction, ~100 neurons)")
    print("\nEye Movement System:")
    print("  ✓ Microsaccades (prevent adaptation, ~1 Hz)")
    print("  ✓ Ocular drift and tremor (realistic fixation)")
    print("  ✓ Voluntary saccades (API control)")
    print("  ✓ Vergence movements (binocular depth)")
    print("  ✓ Foveal extraction from large visual field")
    print("\nFeatures:")
    print("  ✓ Foveal structure with eccentricity-based pooling")
    print("  ✓ Tripartite synapses (horizontal & amacrine modulation)")
    print("  ✓ Continuous temporal dynamics with spiking")
    print("  ✓ Webcam input with active vision (eye movements)")
    print("  ✓ Lightweight 2D visualization")
    print("\nTotal: ~26,000 retinal neurons")
    print("Output: Optic nerve ready for LGN integration")
    print("\nStarting server at http://localhost:5001")
    print("Note: Using port 5001 to avoid AirPlay Receiver on port 5000")
    print("=" * 70)
    print()
    
    app.run(debug=True, port=5001, threaded=True, host='127.0.0.1')

