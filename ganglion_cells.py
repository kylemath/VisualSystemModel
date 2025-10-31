"""
Retinal Ganglion Cell (RGC) layer - output neurons of the retina.
Three types: P cells, M cells, and ipRGCs.
Forms the optic nerve to LGN.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from temporal_neuron import TemporalNeuron
from bipolar_cells import BipolarCell, BipolarLayer


class GanglionCell(TemporalNeuron):
    """
    Retinal Ganglion Cell - output neuron of retina.
    Receives input from bipolar cells (and amacrine cells).
    """
    
    def __init__(self, cell_subtype: str, position: Tuple[float, float],
                 eye: str, eccentricity: float):
        """
        Args:
            cell_subtype: 'P', 'M', or 'ipRGC'
            position: (x, y) spatial position
            eye: 'left' or 'right'
            eccentricity: Distance from fovea
        """
        super().__init__(neuron_type=f"ganglion_{cell_subtype}")
        
        self.cell_subtype = cell_subtype
        self.position = position
        self.eye = eye
        self.eccentricity = eccentricity
        
        # Bipolar cell inputs (both ON and OFF)
        self.on_bipolar_inputs: List[BipolarCell] = []
        self.off_bipolar_inputs: List[BipolarCell] = []
        
        # Type-specific properties
        if cell_subtype == 'P':
            # Parvocellular: Small receptive field, sustained response, color
            self.response_type = 'sustained'
            self.temporal_frequency_pref = 'low'  # <10 Hz
            self.spatial_frequency_pref = 'high'  # Fine detail
            self.conduction_velocity = 'slow'  # ~10 m/s
            
        elif cell_subtype == 'M':
            # Magnocellular: Large receptive field, transient response, motion
            self.response_type = 'transient'
            self.temporal_frequency_pref = 'high'  # >10 Hz
            self.spatial_frequency_pref = 'low'  # Coarse features
            self.conduction_velocity = 'fast'  # ~20 m/s
            
        elif cell_subtype == 'ipRGC':
            # Intrinsically photosensitive: Melanopsin, circadian rhythm
            self.response_type = 'sustained_slow'
            self.temporal_frequency_pref = 'very_low'  # <1 Hz
            self.spatial_frequency_pref = 'very_low'  # Broad integration
            self.conduction_velocity = 'very_slow'  # ~5 m/s
            self.melanopsin_response = 0.0  # Direct light response
    
    def add_bipolar_input(self, bipolar: BipolarCell, weight: float = 0.5):
        """Add input from bipolar cell."""
        if bipolar.cell_type == 'ON':
            self.on_bipolar_inputs.append(bipolar)
        else:
            self.off_bipolar_inputs.append(bipolar)
        
        # Add as standard neural connection
        self.add_input(bipolar, weight)
    
    def compute_ganglion_input(self) -> float:
        """
        Compute total input from bipolar cells.
        P and M cells integrate differently.
        """
        total_input = 0.0
        
        # ON pathway
        for bipolar in self.on_bipolar_inputs:
            response = bipolar.get_activation()
            total_input += response
        
        # OFF pathway (typically antagonistic or complementary)
        for bipolar in self.off_bipolar_inputs:
            response = bipolar.get_activation()
            # Some RGCs are ON-center (OFF antagonistic)
            # Some are OFF-center (ON antagonistic)
            # For simplicity, add both positively for now
            total_input += response
        
        # Average
        total_bipolar = len(self.on_bipolar_inputs) + len(self.off_bipolar_inputs)
        if total_bipolar > 0:
            total_input /= total_bipolar
        
        # Type-specific processing
        if self.cell_subtype == 'M':
            # M cells: more transient, sensitive to changes
            # Simple model: emphasize temporal derivative (handled by temporal dynamics)
            pass
        elif self.cell_subtype == 'P':
            # P cells: sustained response
            pass
        elif self.cell_subtype == 'ipRGC':
            # ipRGCs: Add melanopsin response (direct light sensitivity)
            total_input += self.melanopsin_response
        
        # Increased gain for better visualization
        # Ganglion cells amplify weak retinal signals
        return total_input * 40.0  # Scale to mV (increased from 15.0 for visibility)
    
    def update(self, dt: Optional[float] = None, current_time: float = 0.0):
        """Update ganglion cell with spiking dynamics."""
        if dt is None:
            dt = self._dt * 1000
        
        # Get input from bipolar cells
        bipolar_input = self.compute_ganglion_input()
        
        # Total current
        total_current = bipolar_input + self.expectation_bias
        
        # Ganglion cells DO spike (unlike bipolar cells)
        # Use full integrate-and-fire from parent class
        return super().update(dt, current_time)
    
    def get_state(self) -> Dict:
        """Get ganglion cell state."""
        state = super().get_state()
        state.update({
            'cell_subtype': self.cell_subtype,
            'position': self.position,
            'eye': self.eye,
            'eccentricity': float(self.eccentricity),
            'on_inputs': len(self.on_bipolar_inputs),
            'off_inputs': len(self.off_bipolar_inputs),
            'response_type': self.response_type
        })
        return state


class GanglionLayer:
    """
    Layer of retinal ganglion cells with P, M, and ipRGC types.
    Output layer of retina, forms optic nerve.
    """
    
    def __init__(self, bipolar_layer: BipolarLayer):
        """
        Create ganglion cell layer from bipolar input.
        
        Args:
            bipolar_layer: BipolarLayer instance
        """
        self.bipolar_layer = bipolar_layer
        self.retina = bipolar_layer.retina
        
        # Ganglion cells for each eye
        self.ganglion_cells = {
            'left': {'P': [], 'M': [], 'ipRGC': []},
            'right': {'P': [], 'M': [], 'ipRGC': []}
        }
        
        # Create ganglion cells
        self._create_ganglion_cells()
        
        # Statistics
        self.total_ganglion_cells = sum(
            len(cells) for eye_data in self.ganglion_cells.values()
            for cells in eye_data.values()
        )
        
        print(f"Created {self.total_ganglion_cells} ganglion cells")
    
    def _create_ganglion_cells(self):
        """Create ganglion cells with appropriate pooling."""
        for eye in ['left', 'right']:
            self._create_eye_ganglion_cells(eye)
    
    def _create_eye_ganglion_cells(self, eye: str):
        """Create ganglion cells for one eye."""
        # Get bipolar cells for this eye
        on_bipolars = self.bipolar_layer.bipolar_cells[eye]['ON']
        off_bipolars = self.bipolar_layer.bipolar_cells[eye]['OFF']
        all_bipolars = on_bipolars + off_bipolars
        
        if len(all_bipolars) == 0:
            return
        
        # Generate ganglion positions (sparser than bipolar)
        ganglion_positions = self._generate_ganglion_positions()
        
        for position in ganglion_positions:
            x, y = position
            eccentricity = np.sqrt(x**2 + y**2)
            
            # Determine cell type based on eccentricity and probability
            # P cells: Dominant in fovea (70% in fovea, 30% periphery)
            # M cells: More in periphery (20% in fovea, 50% periphery)
            # ipRGC: Sparse everywhere (~5%)
            
            if eccentricity < self.retina.fovea_radius:
                # Fovea
                type_prob = np.random.rand()
                if type_prob < 0.70:
                    cell_type = 'P'
                elif type_prob < 0.95:
                    cell_type = 'M'
                else:
                    cell_type = 'ipRGC'
            else:
                # Periphery
                type_prob = np.random.rand()
                if type_prob < 0.30:
                    cell_type = 'P'
                elif type_prob < 0.95:
                    cell_type = 'M'
                else:
                    cell_type = 'ipRGC'
            
            # Create ganglion cell
            ganglion = GanglionCell(cell_type, position, eye, eccentricity)
            
            # Connect to bipolar cells based on type
            if cell_type == 'P':
                # P cells: Small pooling (1-2 bipolar cells, one-to-one in fovea)
                if eccentricity < self.retina.fovea_radius:
                    pool_radius = 0.02  # Very small
                else:
                    pool_radius = 0.05
            elif cell_type == 'M':
                # M cells: Larger pooling (double the bipolar pooling)
                if eccentricity < self.retina.fovea_radius:
                    pool_radius = 0.06
                else:
                    pool_radius = 0.15  # Large in periphery
            else:  # ipRGC
                # ipRGC: Very large pooling (broad integration)
                pool_radius = 0.3  # Huge receptive fields
            
            # Connect to nearby bipolar cells
            self._connect_to_bipolars(ganglion, all_bipolars, pool_radius)
            
            # Add to layer
            self.ganglion_cells[eye][cell_type].append(ganglion)
    
    def _generate_ganglion_positions(self) -> List[Tuple[float, float]]:
        """
        Generate positions for ganglion cells.
        Sparser than bipolar cells (~1:10 ratio in fovea, 1:5 in periphery).
        """
        positions = []
        
        # Fovea: P cells dominant, fairly dense
        fovea_grid = 12  # Less dense than bipolar (was 16)
        for i in range(fovea_grid):
            for j in range(fovea_grid):
                x = (i / (fovea_grid - 1)) * 2 * self.retina.fovea_radius - self.retina.fovea_radius
                y = (j / (fovea_grid - 1)) * 2 * self.retina.fovea_radius - self.retina.fovea_radius
                positions.append((x, y))
        
        # Periphery: M cells dominant, sparse
        periphery_grid = 8  # Sparser than bipolar (was 12)
        for i in range(periphery_grid):
            for j in range(periphery_grid):
                x = (i / (periphery_grid - 1)) * 2.0 - 1.0
                y = (j / (periphery_grid - 1)) * 2.0 - 1.0
                
                eccentricity = np.sqrt(x**2 + y**2)
                
                # Skip foveal region (already covered)
                if eccentricity > self.retina.fovea_radius:
                    positions.append((x, y))
        
        return positions
    
    def _connect_to_bipolars(self, ganglion: GanglionCell, 
                            bipolars: List[BipolarCell],
                            pool_radius: float):
        """Connect ganglion cell to nearby bipolar cells."""
        gx, gy = ganglion.position
        
        for bipolar in bipolars:
            bx, by = bipolar.position
            distance = np.sqrt((gx - bx)**2 + (gy - by)**2)
            
            if distance < pool_radius:
                # Weight based on distance (closer = stronger)
                weight = 1.0 - (distance / pool_radius)
                ganglion.add_bipolar_input(bipolar, weight)
    
    def update(self, dt: float = 1.0):
        """Update all ganglion cells."""
        for eye_data in self.ganglion_cells.values():
            for cell_list in eye_data.values():
                for cell in cell_list:
                    cell.update(dt, self.retina.current_time)
    
    def get_activity_map(self, eye: str, cell_type: str = 'P') -> np.ndarray:
        """Get activity map for ganglion cells."""
        grid_size = self.retina.grid_size
        activity_map = np.zeros((grid_size, grid_size))
        count_map = np.zeros((grid_size, grid_size))
        
        cells = self.ganglion_cells[eye][cell_type]
        
        for cell in cells:
            x, y = cell.position
            i = int((x + 1.0) / 2.0 * (grid_size - 1))
            j = int((y + 1.0) / 2.0 * (grid_size - 1))
            
            i = np.clip(i, 0, grid_size - 1)
            j = np.clip(j, 0, grid_size - 1)
            
            activity_map[i, j] += cell.get_activation()
            count_map[i, j] += 1
        
        mask = count_map > 0
        activity_map[mask] /= count_map[mask]
        
        return activity_map
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {
            'type': 'ganglion_layer',
            'total_ganglion_cells': self.total_ganglion_cells
        }
        
        for eye in ['left', 'right']:
            eye_summary = {}
            for cell_type in ['P', 'M', 'ipRGC']:
                cells = self.ganglion_cells[eye][cell_type]
                
                # Average activity and spike rate
                total_activity = sum(c.get_activation() for c in cells)
                total_spike_rate = sum(c.spike_rate for c in cells)
                
                mean_activity = total_activity / len(cells) if len(cells) > 0 else 0
                mean_spike_rate = total_spike_rate / len(cells) if len(cells) > 0 else 0
                
                # Count by eccentricity zones
                foveal = sum(1 for c in cells if c.eccentricity < self.retina.fovea_radius)
                peripheral = len(cells) - foveal
                
                eye_summary[cell_type] = {
                    'count': len(cells),
                    'foveal': foveal,
                    'peripheral': peripheral,
                    'mean_activity': mean_activity,
                    'mean_spike_rate': mean_spike_rate
                }
            
            summary[eye] = eye_summary
        
        return summary
    
    def get_optic_nerve_output(self, eye: str) -> List[Dict]:
        """
        Get optic nerve output (spike trains from all RGCs).
        This is what goes to LGN.
        """
        output = []
        
        for cell_type in ['P', 'M', 'ipRGC']:
            for cell in self.ganglion_cells[eye][cell_type]:
                if cell.is_spiking:
                    output.append({
                        'cell_id': cell.id,
                        'cell_type': cell_type,
                        'position': cell.position,
                        'spike_rate': cell.spike_rate,
                        'membrane_potential': cell.v_membrane
                    })
        
        return output

