"""
Flask server for neural network visualization.
Provides API endpoints for the HTML interface.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import json

from retina import RetinaLayer

app = Flask(__name__)
CORS(app)

# Global neural system
neural_system = {
    'retina': None
}


@app.route('/')
def index():
    """Serve the main visualization interface."""
    return render_template('index.html')


@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the neural system with specified parameters."""
    data = request.json
    grid_size = data.get('grid_size', 32)
    
    # Create retinal layer
    neural_system['retina'] = RetinaLayer(grid_size=grid_size)
    
    return jsonify({
        'status': 'success',
        'message': f'Initialized retinal layer with {grid_size}×{grid_size} grid',
        'system_info': {
            'retina': neural_system['retina'].get_summary()
        }
    })


@app.route('/api/retina/process', methods=['POST'])
def process_retina_input():
    """Process visual input through retinal layer."""
    if neural_system['retina'] is None:
        return jsonify({'error': 'System not initialized'}), 400
    
    data = request.json
    
    # Get or generate test images
    grid_size = neural_system['retina'].grid_size
    
    if 'left_image' in data and 'right_image' in data:
        left_image = np.array(data['left_image'])
        right_image = np.array(data['right_image'])
    else:
        # Generate test pattern
        pattern = data.get('pattern', 'gradient')
        left_image, right_image = generate_test_pattern(grid_size, pattern)
    
    intensity = data.get('intensity', 1.0)
    
    # Process images
    neural_system['retina'].process_image(left_image, right_image, intensity)
    
    # Get activity maps
    left_maps = neural_system['retina'].get_activity_maps('left')
    right_maps = neural_system['retina'].get_activity_maps('right')
    
    return jsonify({
        'status': 'success',
        'left_eye': {
            receptor_type: activity_map.tolist()
            for receptor_type, activity_map in left_maps.items()
        },
        'right_eye': {
            receptor_type: activity_map.tolist()
            for receptor_type, activity_map in right_maps.items()
        },
        'summary': neural_system['retina'].get_summary()
    })


@app.route('/api/retina/state', methods=['GET'])
def get_retina_state():
    """Get current state of retinal layer."""
    if neural_system['retina'] is None:
        return jsonify({'error': 'System not initialized'}), 400
    
    return jsonify(neural_system['retina'].get_summary())


@app.route('/api/system/architecture', methods=['GET'])
def get_architecture():
    """Get the current neural architecture structure."""
    architecture = {
        'layers': [
            {
                'id': 'retina',
                'name': 'Retinal Layer',
                'type': 'input',
                'description': 'Photoreceptor layer with rods and RGB cones',
                'status': 'active' if neural_system['retina'] is not None else 'inactive',
                'neuron_count': neural_system['retina'].total_photoreceptors if neural_system['retina'] else 0,
                'connections_to': []
            }
        ],
        'connections': []
    }
    
    return jsonify(architecture)


def generate_test_pattern(size: int, pattern: str = 'gradient') -> tuple:
    """Generate test patterns for visualization."""
    if pattern == 'gradient':
        # Horizontal RGB gradient
        image = np.zeros((size, size, 3))
        for x in range(size):
            for y in range(size):
                image[x, y] = [x/size, y/size, (x+y)/(2*size)]
        left_image = image
        right_image = image * 0.9  # Slightly dimmer for right eye
        
    elif pattern == 'checkerboard':
        # Black and white checkerboard
        image = np.zeros((size, size, 3))
        for x in range(size):
            for y in range(size):
                if (x // 4 + y // 4) % 2 == 0:
                    image[x, y] = [1.0, 1.0, 1.0]
        left_image = image
        right_image = image
        
    elif pattern == 'color_bands':
        # Vertical color bands
        image = np.zeros((size, size, 3))
        band_width = size // 3
        image[:, :band_width, 0] = 1.0  # Red
        image[:, band_width:2*band_width, 1] = 1.0  # Green
        image[:, 2*band_width:, 2] = 1.0  # Blue
        left_image = image
        right_image = image
        
    else:
        # Random noise
        left_image = np.random.rand(size, size, 3)
        right_image = np.random.rand(size, size, 3)
    
    return left_image, right_image


if __name__ == '__main__':
    print("Starting Bio-Inspired Neural Network Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)

