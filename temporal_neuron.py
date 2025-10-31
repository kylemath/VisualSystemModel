"""
Temporal neuron model with continuous time dynamics.
Implements oscillations, action potentials, refractory periods,
and preparatory depolarization for expectations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import weakref


class TemporalNeuron:
    """
    Biologically-realistic neuron with temporal dynamics.
    
    Features:
    - Continuous time membrane potential
    - Action potential generation
    - Absolute and relative refractory periods
    - Oscillatory dynamics (theta, alpha, gamma bands)
    - Preparatory depolarization (expectation)
    - Re-entrant feedback support
    """
    
    _neuron_count = 0
    _dt = 0.001  # Time step in seconds (1ms)
    
    def __init__(self, neuron_type: str = "generic"):
        self.id = TemporalNeuron._neuron_count
        TemporalNeuron._neuron_count += 1
        
        self.neuron_type = neuron_type
        
        # Membrane dynamics
        self.v_rest = -70.0      # Resting potential (mV)
        self.v_threshold = -55.0  # Spike threshold (mV)
        self.v_peak = 40.0       # Spike peak (mV)
        self.v_reset = -75.0     # Post-spike reset (mV)
        self.v_membrane = self.v_rest  # Current membrane potential
        
        # Time constants (ms)
        self.tau_membrane = 20.0  # Membrane time constant
        self.tau_refrac_abs = 2.0  # Absolute refractory period
        self.tau_refrac_rel = 5.0  # Relative refractory period
        
        # State variables
        self.time_since_spike = 1000.0  # ms
        self.is_spiking = False
        self.spike_rate = 0.0  # Instantaneous firing rate
        
        # Oscillatory components (can be modulated)
        self.oscillation_phase = np.random.rand() * 2 * np.pi
        self.oscillation_freq = 8.0  # Hz (default: theta band)
        self.oscillation_amplitude = 2.0  # mV
        
        # Preparatory depolarization (expectation/attention)
        self.expectation_bias = 0.0  # Additional depolarization
        
        # Connections
        self.input_neurons: List[weakref.ref] = []
        self.input_weights: np.ndarray = np.array([])
        self.input_delays: np.ndarray = np.array([])  # Conduction delays (ms)
        self.output_neurons: List[weakref.ref] = []
        
        # Feedback connections (re-entrant)
        self.feedback_neurons: List[weakref.ref] = []
        self.feedback_weights: np.ndarray = np.array([])
        
        # Input buffer for delayed inputs
        self.input_buffer: List[Tuple[float, float]] = []  # (time, input)
        
    def add_input(self, neuron: 'TemporalNeuron', weight: float = 0.5, delay: float = 1.0):
        """Add feedforward input connection."""
        self.input_neurons.append(weakref.ref(neuron))
        self.input_weights = np.append(self.input_weights, weight)
        self.input_delays = np.append(self.input_delays, delay)
        neuron.output_neurons.append(weakref.ref(self))
    
    def add_feedback(self, neuron: 'TemporalNeuron', weight: float = 0.2):
        """Add feedback (re-entrant) connection."""
        self.feedback_neurons.append(weakref.ref(neuron))
        self.feedback_weights = np.append(self.feedback_weights, weight)
    
    def set_expectation(self, bias: float):
        """Set preparatory depolarization (attention/expectation)."""
        self.expectation_bias = bias
    
    def update(self, dt: Optional[float] = None, current_time: float = 0.0):
        """
        Update neuron state for one time step.
        
        Args:
            dt: Time step in ms (uses class default if None)
            current_time: Current simulation time in ms
        """
        if dt is None:
            dt = self._dt * 1000  # Convert to ms
        
        # Check refractory period
        in_absolute_refrac = self.time_since_spike < self.tau_refrac_abs
        in_relative_refrac = self.time_since_spike < self.tau_refrac_rel
        
        if in_absolute_refrac:
            # Absolute refractory: no integration, stay at reset
            self.v_membrane = self.v_reset
            self.time_since_spike += dt
            self.is_spiking = False
            return self.v_membrane
        
        # Compute synaptic input
        synaptic_current = self._compute_synaptic_input()
        
        # Compute feedback input
        feedback_current = self._compute_feedback_input()
        
        # Add oscillatory component
        self.oscillation_phase += 2 * np.pi * self.oscillation_freq * dt / 1000.0
        oscillation = self.oscillation_amplitude * np.sin(self.oscillation_phase)
        
        # Total input current
        total_current = synaptic_current + feedback_current + oscillation + self.expectation_bias
        
        # Relative refractory period: reduced sensitivity
        if in_relative_refrac:
            total_current *= 0.5
        
        # Leaky integrate-and-fire dynamics
        # dV/dt = (V_rest - V + R*I) / tau
        dv = ((self.v_rest - self.v_membrane + total_current) / self.tau_membrane) * dt
        self.v_membrane += dv
        
        # Check for spike
        if self.v_membrane >= self.v_threshold:
            self._generate_spike()
        else:
            self.is_spiking = False
        
        self.time_since_spike += dt
        
        # Update firing rate (exponential moving average)
        if self.is_spiking:
            self.spike_rate = 1000.0 / max(self.time_since_spike, 1.0)  # Hz
        else:
            self.spike_rate *= 0.99  # Decay
        
        return self.v_membrane
    
    def _compute_synaptic_input(self) -> float:
        """Compute total synaptic input from feedforward connections."""
        total = 0.0
        for neuron_ref, weight in zip(self.input_neurons, self.input_weights):
            neuron = neuron_ref()
            if neuron is not None:
                # Simple model: weight * (spike_rate / 100)
                # In real implementation, use spike buffer with delays
                if neuron.is_spiking:
                    total += weight * 10.0  # Spike gives strong input
                else:
                    total += weight * (neuron.v_membrane - neuron.v_rest) / 10.0
        return total
    
    def _compute_feedback_input(self) -> float:
        """Compute feedback input from re-entrant connections."""
        total = 0.0
        for neuron_ref, weight in zip(self.feedback_neurons, self.feedback_weights):
            neuron = neuron_ref()
            if neuron is not None:
                total += weight * (neuron.v_membrane - neuron.v_rest) / 10.0
        return total
    
    def _generate_spike(self):
        """Generate action potential."""
        self.v_membrane = self.v_peak
        self.is_spiking = True
        self.time_since_spike = 0.0
        # Next step will reset to v_reset
    
    def get_activation(self) -> float:
        """Get normalized activation (0-1) for visualization."""
        # Map membrane potential to 0-1 range
        return np.clip((self.v_membrane - self.v_rest) / (self.v_threshold - self.v_rest), 0, 1)
    
    def get_state(self) -> Dict:
        """Get complete neuron state."""
        return {
            'id': self.id,
            'type': self.neuron_type,
            'v_membrane': float(self.v_membrane),
            'v_threshold': float(self.v_threshold),
            'is_spiking': self.is_spiking,
            'spike_rate': float(self.spike_rate),
            'activation': float(self.get_activation()),
            'time_since_spike': float(self.time_since_spike),
            'expectation_bias': float(self.expectation_bias),
            'oscillation_phase': float(self.oscillation_phase),
            'num_inputs': len(self.input_neurons),
            'num_outputs': len(self.output_neurons),
            'num_feedback': len(self.feedback_neurons)
        }


class TemporalPhotoreceptor(TemporalNeuron):
    """
    Photoreceptor with temporal dynamics.
    Responds continuously to light input with adaptation.
    """
    
    def __init__(self, receptor_type: str, position: Tuple[float, float], 
                 eye: str = "left", eccentricity: float = 0.0):
        super().__init__(neuron_type=f"photoreceptor_{receptor_type}")
        
        self.receptor_type = receptor_type
        self.position = position  # (x, y) continuous position
        self.eye = eye
        self.eccentricity = eccentricity  # Distance from fovea (0=fovea center)
        
        # Spectral sensitivity
        self.peak_wavelengths = {
            'rod': 498,
            'red': 564,
            'green': 534,
            'blue': 420
        }
        self.peak_wavelength = self.peak_wavelengths.get(receptor_type, 550)
        
        # Photoreceptor-specific dynamics
        self.light_input = 0.0
        self.adapted_response = 0.0
        self.tau_adaptation = 100.0  # Adaptation time constant (ms)
        
        # Photoreceptors hyperpolarize in response to light
        self.v_dark = -40.0  # Depolarized in dark
        self.v_light = -70.0  # Hyperpolarized in light
        self.v_membrane = self.v_dark
        
    def set_light_input(self, rgb_value: Tuple[float, float, float], intensity: float = 1.0):
        """Set light input (called once per frame)."""
        r, g, b = rgb_value
        
        if self.receptor_type == 'rod':
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            response = luminance * intensity
        elif self.receptor_type == 'red':
            response = (0.8 * r + 0.3 * g) * intensity
        elif self.receptor_type == 'green':
            response = (0.6 * g + 0.2 * r) * intensity
        elif self.receptor_type == 'blue':
            response = 0.9 * b * intensity
        else:
            response = (r + g + b) / 3.0 * intensity
        
        self.light_input = response
    
    def update(self, dt: Optional[float] = None, current_time: float = 0.0):
        """Update photoreceptor with light adaptation."""
        if dt is None:
            dt = self._dt * 1000
        
        # Adaptation: exponential approach to light input
        adaptation_rate = dt / self.tau_adaptation
        self.adapted_response += (self.light_input - self.adapted_response) * adaptation_rate
        
        # Photoreceptors hyperpolarize with light
        target_v = self.v_dark + (self.v_light - self.v_dark) * self.adapted_response
        
        # Move towards target
        self.v_membrane += (target_v - self.v_membrane) * (dt / self.tau_membrane)
        
        # Photoreceptors don't spike, but we track their state
        self.is_spiking = False
        
        return self.v_membrane
    
    def get_activation(self) -> float:
        """Get normalized activation for visualization."""
        # Inverted: more light = more hyperpolarized = more "active" for downstream
        return self.adapted_response
    
    def get_state(self) -> Dict:
        """Get photoreceptor state."""
        state = super().get_state()
        state.update({
            'receptor_type': self.receptor_type,
            'position': self.position,
            'eye': self.eye,
            'eccentricity': float(self.eccentricity),
            'peak_wavelength': self.peak_wavelength,
            'light_input': float(self.light_input),
            'adapted_response': float(self.adapted_response)
        })
        return state

