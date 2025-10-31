"""
Base neuron classes for bio-inspired neural network.
Designed for scalability and biological realism.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import weakref


class Neuron:
    """
    Base neuron class - represents a single neural unit.
    Uses efficient storage for connections and weights.
    """
    _neuron_count = 0
    
    def __init__(self, neuron_type: str = "generic"):
        self.id = Neuron._neuron_count
        Neuron._neuron_count += 1
        
        self.neuron_type = neuron_type
        self.activity = 0.0  # Current activation level
        self.membrane_potential = 0.0
        
        # Efficient connection storage using indices
        self.input_neurons: List[weakref.ref] = []  # Weak references to prevent circular refs
        self.input_weights: np.ndarray = np.array([])
        self.output_neurons: List[weakref.ref] = []
        
        # Biological properties
        self.threshold = 0.5
        self.refractory_period = 0
        self.time_since_spike = 1000
        
    def add_input(self, neuron: 'Neuron', weight: float = 0.5):
        """Add an input connection from another neuron."""
        self.input_neurons.append(weakref.ref(neuron))
        self.input_weights = np.append(self.input_weights, weight)
        neuron.output_neurons.append(weakref.ref(self))
        
    def compute_activation(self) -> float:
        """Compute activation based on inputs."""
        if self.time_since_spike < self.refractory_period:
            self.time_since_spike += 1
            return 0.0
        
        total_input = 0.0
        for neuron_ref, weight in zip(self.input_neurons, self.input_weights):
            neuron = neuron_ref()
            if neuron is not None:
                total_input += neuron.activity * weight
        
        # Simple sigmoid activation
        self.membrane_potential = total_input
        self.activity = 1.0 / (1.0 + np.exp(-total_input))
        
        # Spike detection
        if self.activity > self.threshold:
            self.time_since_spike = 0
            
        return self.activity
    
    def get_state(self) -> Dict:
        """Return current neuron state for visualization."""
        return {
            'id': self.id,
            'type': self.neuron_type,
            'activity': float(self.activity),
            'membrane_potential': float(self.membrane_potential),
            'num_inputs': len(self.input_neurons),
            'num_outputs': len(self.output_neurons)
        }


class Photoreceptor(Neuron):
    """
    Photoreceptor cell - specialized for light detection.
    Base class for rods and cones.
    """
    
    def __init__(self, receptor_type: str, position: Tuple[int, int], eye: str = "left"):
        super().__init__(neuron_type=f"photoreceptor_{receptor_type}")
        self.receptor_type = receptor_type  # 'rod', 'red', 'green', 'blue'
        self.position = position  # (x, y) position in retina
        self.eye = eye  # 'left' or 'right'
        
        # Spectral sensitivity curves (peak wavelengths in nm)
        self.peak_wavelengths = {
            'rod': 498,    # Scotopic vision
            'red': 564,    # L-cone
            'green': 534,  # M-cone
            'blue': 420    # S-cone
        }
        self.peak_wavelength = self.peak_wavelengths.get(receptor_type, 550)
        
        # Sensitivity parameters
        self.dark_adaptation = 1.0
        self.saturation_level = 1.0
        
    def set_light_input(self, rgb_value: Tuple[float, float, float], intensity: float = 1.0):
        """
        Set the light input based on RGB values and overall intensity.
        Simulates spectral sensitivity of different photoreceptor types.
        """
        r, g, b = rgb_value
        
        if self.receptor_type == 'rod':
            # Rods are most sensitive to blue-green, respond to luminance
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            response = luminance * self.dark_adaptation
        elif self.receptor_type == 'red':
            # L-cones peak at red, but also respond to green/yellow
            response = 0.8 * r + 0.3 * g
        elif self.receptor_type == 'green':
            # M-cones peak at green
            response = 0.6 * g + 0.2 * r
        elif self.receptor_type == 'blue':
            # S-cones peak at blue
            response = 0.9 * b
        else:
            response = (r + g + b) / 3.0
        
        # Apply intensity and saturation
        response *= intensity
        response = min(response, self.saturation_level)
        
        # Set activity directly (photoreceptors are input neurons)
        self.activity = response
        return response
    
    def get_state(self) -> Dict:
        """Return photoreceptor state including position and type."""
        state = super().get_state()
        state.update({
            'receptor_type': self.receptor_type,
            'position': self.position,
            'eye': self.eye,
            'peak_wavelength': self.peak_wavelength
        })
        return state

