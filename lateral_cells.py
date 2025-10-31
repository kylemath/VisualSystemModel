"""
Lateral processing cells in the retina.
Horizontal cells (photoreceptor level) and Amacrine cells (bipolar level).
Provide contextual modulation and lateral inhibition.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from temporal_neuron import TemporalNeuron, TemporalPhotoreceptor
from bipolar_cells import BipolarCell
from ganglion_cells import GanglionCell


class HorizontalCell(TemporalNeuron):
    """
    Horizontal cell - lateral processing at photoreceptor→bipolar synapse.
    Forms tripartite synapses.
    Integrates over wide spatial area for context.
    """
    
    def __init__(self, position: Tuple[float, float], eye: str, eccentricity: float):
        """
        Args:
            position: (x, y) spatial position
            eye: 'left' or 'right'
            eccentricity: Distance from fovea
        """
        super().__init__(neuron_type="horizontal")
        
        self.position = position
        self.eye = eye
        self.eccentricity = eccentricity
        
        # Horizontal cells integrate from many photoreceptors
        self.photoreceptor_inputs: List[TemporalPhotoreceptor] = []
        
        # They modulate bipolar cells (lateral inhibition)
        self.modulated_bipolars: List[BipolarCell] = []
        
        # Wide receptive field
        self.receptive_field_radius = 0.15  # Large spatial integration
        
        # Horizontal cells are GABAergic (inhibitory)
        self.inhibitory = True
    
    def add_photoreceptor_input(self, receptor: TemporalPhotoreceptor):
        """Add photoreceptor to integrate from."""
        self.photoreceptor_inputs.append(receptor)
    
    def add_bipolar_target(self, bipolar: BipolarCell):
        """Add bipolar cell to modulate."""
        self.modulated_bipolars.append(bipolar)
    
    def compute_lateral_signal(self) -> float:
        """
        Compute average activity across wide receptive field.
        This represents the local context.
        """
        if len(self.photoreceptor_inputs) == 0:
            return 0.0
        
        total_activity = sum(r.get_activation() for r in self.photoreceptor_inputs)
        average_activity = total_activity / len(self.photoreceptor_inputs)
        
        return average_activity
    
    def get_modulation_strength(self) -> float:
        """
        Get the modulation strength to apply to bipolar cells.
        Implements lateral inhibition (subtractive or divisive).
        """
        lateral_signal = self.compute_lateral_signal()
        
        # Lateral inhibition: subtract average (surround suppression)
        # Stronger modulation = more lateral signal
        modulation = -0.3 * lateral_signal  # Inhibitory, scaled
        
        return modulation
    
    def update(self, dt: Optional[float] = None, current_time: float = 0.0):
        """Update horizontal cell."""
        if dt is None:
            dt = self._dt * 1000
        
        # Compute lateral signal
        lateral_signal = self.compute_lateral_signal()
        
        # Set membrane potential based on lateral signal
        # Horizontal cells have graded potentials
        target_v = self.v_rest + lateral_signal * 20.0
        
        # Smooth integration
        dv = ((target_v - self.v_membrane) / self.tau_membrane) * dt
        self.v_membrane += dv
        
        return self.v_membrane
    
    def get_state(self) -> Dict:
        """Get horizontal cell state."""
        state = super().get_state()
        state.update({
            'position': self.position,
            'eye': self.eye,
            'eccentricity': float(self.eccentricity),
            'num_photoreceptor_inputs': len(self.photoreceptor_inputs),
            'num_bipolar_targets': len(self.modulated_bipolars),
            'lateral_signal': float(self.compute_lateral_signal())
        })
        return state


class AmacrineCell(TemporalNeuron):
    """
    Amacrine cell - lateral processing at bipolar→ganglion synapse.
    Forms tripartite synapses.
    Many subtypes with different functions (motion, direction, etc).
    """
    
    def __init__(self, subtype: str, position: Tuple[float, float], 
                 eye: str, eccentricity: float):
        """
        Args:
            subtype: e.g., 'A2' (wide-field), 'starburst' (direction), 'A17' (feedback)
            position: (x, y) spatial position
            eye: 'left' or 'right'
            eccentricity: Distance from fovea
        """
        super().__init__(neuron_type=f"amacrine_{subtype}")
        
        self.subtype = subtype
        self.position = position
        self.eye = eye
        self.eccentricity = eccentricity
        
        # Amacrine cells integrate from bipolar cells
        self.bipolar_inputs: List[BipolarCell] = []
        
        # They modulate ganglion cells
        self.modulated_ganglions: List[GanglionCell] = []
        
        # Subtype-specific properties
        if subtype == 'A2':
            # Wide-field: Large receptive field, general inhibition
            self.receptive_field_radius = 0.2
            self.inhibitory = True
            self.function = 'lateral_inhibition'
            
        elif subtype == 'starburst':
            # Direction selective: Responds to motion direction
            self.receptive_field_radius = 0.1
            self.inhibitory = True  # And excitatory (releases ACh and GABA)
            self.function = 'direction_selectivity'
            self.preferred_direction = np.random.rand() * 2 * np.pi  # Random direction
            
        elif subtype == 'A17':
            # Feedback to rod bipolar cells
            self.receptive_field_radius = 0.05
            self.inhibitory = True
            self.function = 'feedback_inhibition'
        
        else:
            # Generic amacrine
            self.receptive_field_radius = 0.1
            self.inhibitory = True
            self.function = 'general'
    
    def add_bipolar_input(self, bipolar: BipolarCell):
        """Add bipolar cell to integrate from."""
        self.bipolar_inputs.append(bipolar)
    
    def add_ganglion_target(self, ganglion: GanglionCell):
        """Add ganglion cell to modulate."""
        self.modulated_ganglions.append(ganglion)
    
    def compute_lateral_signal(self) -> float:
        """
        Compute activity across bipolar cell inputs.
        Different for different subtypes.
        """
        if len(self.bipolar_inputs) == 0:
            return 0.0
        
        if self.function == 'direction_selectivity':
            # Starburst: Compute direction of motion (simplified)
            # Would need temporal history for real motion detection
            # For now, just sum activity
            total_activity = sum(b.get_activation() for b in self.bipolar_inputs)
            return total_activity / len(self.bipolar_inputs)
        
        else:
            # General: Average activity
            total_activity = sum(b.get_activation() for b in self.bipolar_inputs)
            return total_activity / len(self.bipolar_inputs)
    
    def get_modulation_strength(self) -> float:
        """
        Get modulation strength for ganglion cells.
        """
        lateral_signal = self.compute_lateral_signal()
        
        if self.inhibitory:
            # Lateral inhibition
            modulation = -0.4 * lateral_signal
        else:
            # Some amacrines are excitatory
            modulation = 0.3 * lateral_signal
        
        return modulation
    
    def update(self, dt: Optional[float] = None, current_time: float = 0.0):
        """Update amacrine cell."""
        if dt is None:
            dt = self._dt * 1000
        
        # Compute lateral signal
        lateral_signal = self.compute_lateral_signal()
        
        # Amacrine cells can spike (unlike horizontal cells)
        # Set input current based on lateral signal
        input_current = lateral_signal * 25.0
        
        # Use parent class update (integrate-and-fire)
        self.v_membrane += input_current * dt / self.tau_membrane
        
        # Check for spike
        if self.v_membrane >= self.v_threshold:
            self._generate_spike()
        
        self.time_since_spike += dt
        
        return self.v_membrane
    
    def get_state(self) -> Dict:
        """Get amacrine cell state."""
        state = super().get_state()
        state.update({
            'subtype': self.subtype,
            'position': self.position,
            'eye': self.eye,
            'eccentricity': float(self.eccentricity),
            'num_bipolar_inputs': len(self.bipolar_inputs),
            'num_ganglion_targets': len(self.modulated_ganglions),
            'lateral_signal': float(self.compute_lateral_signal()),
            'function': self.function
        })
        return state


class LateralProcessingLayer:
    """
    Combined layer of horizontal and amacrine cells.
    Provides lateral processing and context to retinal circuits.
    """
    
    def __init__(self, retina, bipolar_layer, ganglion_layer):
        """
        Create lateral processing cells.
        
        Args:
            retina: FovealRetina instance
            bipolar_layer: BipolarLayer instance
            ganglion_layer: GanglionLayer instance
        """
        self.retina = retina
        self.bipolar_layer = bipolar_layer
        self.ganglion_layer = ganglion_layer
        
        # Horizontal and amacrine cells for each eye
        self.horizontal_cells = {'left': [], 'right': []}
        self.amacrine_cells = {'left': [], 'right': []}
        
        # Create lateral cells
        self._create_lateral_cells()
        
        print(f"Created {sum(len(cells) for cells in self.horizontal_cells.values())} horizontal cells")
        print(f"Created {sum(len(cells) for cells in self.amacrine_cells.values())} amacrine cells")
    
    def _create_lateral_cells(self):
        """Create horizontal and amacrine cells."""
        for eye in ['left', 'right']:
            self._create_eye_lateral_cells(eye)
    
    def _create_eye_lateral_cells(self, eye: str):
        """Create lateral cells for one eye."""
        # Horizontal cells: Sparse, wide-field
        horizontal_positions = self._generate_sparse_positions(spacing=0.2)
        
        for position in horizontal_positions:
            x, y = position
            eccentricity = np.sqrt(x**2 + y**2)
            
            h_cell = HorizontalCell(position, eye, eccentricity)
            
            # Connect to nearby photoreceptors
            self._connect_horizontal_to_photoreceptors(h_cell, eye)
            
            # Connect to nearby bipolar cells (to modulate)
            self._connect_horizontal_to_bipolars(h_cell, eye)
            
            self.horizontal_cells[eye].append(h_cell)
        
        # Amacrine cells: More numerous, various types
        amacrine_positions = self._generate_sparse_positions(spacing=0.15)
        
        for position in amacrine_positions:
            x, y = position
            eccentricity = np.sqrt(x**2 + y**2)
            
            # Mix of subtypes
            subtype_rand = np.random.rand()
            if subtype_rand < 0.6:
                subtype = 'A2'  # Most common
            elif subtype_rand < 0.8:
                subtype = 'starburst'
            else:
                subtype = 'A17'
            
            a_cell = AmacrineCell(subtype, position, eye, eccentricity)
            
            # Connect to nearby bipolar cells
            self._connect_amacrine_to_bipolars(a_cell, eye)
            
            # Connect to nearby ganglion cells (to modulate)
            self._connect_amacrine_to_ganglions(a_cell, eye)
            
            self.amacrine_cells[eye].append(a_cell)
    
    def _generate_sparse_positions(self, spacing: float = 0.2) -> List[Tuple[float, float]]:
        """Generate sparse grid positions for lateral cells."""
        positions = []
        num_cells = int(2.0 / spacing)
        
        for i in range(num_cells):
            for j in range(num_cells):
                x = (i / (num_cells - 1)) * 2.0 - 1.0
                y = (j / (num_cells - 1)) * 2.0 - 1.0
                positions.append((x, y))
        
        return positions
    
    def _connect_horizontal_to_photoreceptors(self, h_cell: HorizontalCell, eye: str):
        """Connect horizontal cell to photoreceptors in its receptive field."""
        hx, hy = h_cell.position
        radius = h_cell.receptive_field_radius
        
        for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
            for receptor in self.retina.photoreceptors[eye][receptor_type]:
                rx, ry = receptor.position
                distance = np.sqrt((hx - rx)**2 + (hy - ry)**2)
                
                if distance < radius:
                    h_cell.add_photoreceptor_input(receptor)
    
    def _connect_horizontal_to_bipolars(self, h_cell: HorizontalCell, eye: str):
        """Connect horizontal cell to bipolar cells it will modulate."""
        hx, hy = h_cell.position
        radius = h_cell.receptive_field_radius * 0.5  # Modulates nearby bipolars
        
        for cell_type in ['ON', 'OFF']:
            for bipolar in self.bipolar_layer.bipolar_cells[eye][cell_type]:
                bx, by = bipolar.position
                distance = np.sqrt((hx - bx)**2 + (hy - by)**2)
                
                if distance < radius:
                    h_cell.add_bipolar_target(bipolar)
    
    def _connect_amacrine_to_bipolars(self, a_cell: AmacrineCell, eye: str):
        """Connect amacrine cell to bipolar cells in its receptive field."""
        ax, ay = a_cell.position
        radius = a_cell.receptive_field_radius
        
        for cell_type in ['ON', 'OFF']:
            for bipolar in self.bipolar_layer.bipolar_cells[eye][cell_type]:
                bx, by = bipolar.position
                distance = np.sqrt((ax - bx)**2 + (ay - by)**2)
                
                if distance < radius:
                    a_cell.add_bipolar_input(bipolar)
    
    def _connect_amacrine_to_ganglions(self, a_cell: AmacrineCell, eye: str):
        """Connect amacrine cell to ganglion cells it will modulate."""
        ax, ay = a_cell.position
        radius = a_cell.receptive_field_radius * 0.5
        
        for cell_type in ['P', 'M', 'ipRGC']:
            for ganglion in self.ganglion_layer.ganglion_cells[eye][cell_type]:
                gx, gy = ganglion.position
                distance = np.sqrt((ax - gx)**2 + (ay - gy)**2)
                
                if distance < radius:
                    a_cell.add_ganglion_target(ganglion)
    
    def update(self, dt: float = 1.0):
        """Update all lateral cells."""
        # Update horizontal cells first
        for eye_cells in self.horizontal_cells.values():
            for cell in eye_cells:
                cell.update(dt, self.retina.current_time)
        
        # Then amacrine cells
        for eye_cells in self.amacrine_cells.values():
            for cell in eye_cells:
                cell.update(dt, self.retina.current_time)
    
    def apply_lateral_modulation(self):
        """
        Apply lateral modulation from horizontal and amacrine cells.
        This modulates bipolar and ganglion cell responses.
        """
        # Horizontal cells modulate bipolar cells
        for eye in ['left', 'right']:
            for h_cell in self.horizontal_cells[eye]:
                modulation = h_cell.get_modulation_strength()
                
                for bipolar in h_cell.modulated_bipolars:
                    # Add modulation to expectation_bias (acts like top-down modulation)
                    bipolar.expectation_bias += modulation
            
            # Amacrine cells modulate ganglion cells
            for a_cell in self.amacrine_cells[eye]:
                modulation = a_cell.get_modulation_strength()
                
                for ganglion in a_cell.modulated_ganglions:
                    ganglion.expectation_bias += modulation
    
    def get_summary(self) -> Dict:
        """Get summary of lateral processing."""
        summary = {}
        
        for eye in ['left', 'right']:
            summary[eye] = {
                'horizontal_cells': len(self.horizontal_cells[eye]),
                'amacrine_cells': len(self.amacrine_cells[eye]),
                'amacrine_subtypes': {}
            }
            
            # Count amacrine subtypes
            for a_cell in self.amacrine_cells[eye]:
                subtype = a_cell.subtype
                if subtype not in summary[eye]['amacrine_subtypes']:
                    summary[eye]['amacrine_subtypes'][subtype] = 0
                summary[eye]['amacrine_subtypes'][subtype] += 1
        
        return summary

