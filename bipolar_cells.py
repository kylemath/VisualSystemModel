"""
Bipolar cell layer with center-surround receptive fields.
Variable pooling based on eccentricity (P vs M pathways).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from temporal_neuron import TemporalNeuron, TemporalPhotoreceptor
from foveal_retina import FovealRetina


class BipolarCell(TemporalNeuron):
    """
    Bipolar cell with center-surround receptive field.
    
    Two types:
    - ON-center: Excited by light in center, inhibited by surround
    - OFF-center: Inhibited by light in center, excited by surround
    """
    
    def __init__(self, cell_type: str, position: Tuple[float, float],
                 eye: str, eccentricity: float, pathway: str = 'P'):
        """
        Args:
            cell_type: 'ON' or 'OFF'
            position: (x, y) spatial position
            eye: 'left' or 'right'
            eccentricity: Distance from fovea
            pathway: 'P' (parvocellular) or 'M' (magnocellular)
        """
        super().__init__(neuron_type=f"bipolar_{cell_type}_{pathway}")
        
        self.cell_type = cell_type  # 'ON' or 'OFF'
        self.position = position
        self.eye = eye
        self.eccentricity = eccentricity
        self.pathway = pathway
        
        # Receptive field
        self.center_photoreceptors: List[TemporalPhotoreceptor] = []
        self.surround_photoreceptors: List[TemporalPhotoreceptor] = []
        
        # Center-surround weights (physiologically realistic)
        self.center_weight = 1.0
        self.surround_weight = -0.6  # Inhibitory surround
        
        # Polarity: ON cells depolarize to light, OFF cells hyperpolarize
        if cell_type == 'ON':
            self.polarity = 1.0
        else:
            self.polarity = -1.0
    
    def add_center_receptor(self, receptor: TemporalPhotoreceptor):
        """Add photoreceptor to receptive field center."""
        self.center_photoreceptors.append(receptor)
    
    def add_surround_receptor(self, receptor: TemporalPhotoreceptor):
        """Add photoreceptor to receptive field surround."""
        self.surround_photoreceptors.append(receptor)
    
    def compute_receptive_field_input(self) -> float:
        """
        Compute center-surround input.
        Simple: center_sum - surround_sum
        """
        # Center: average photoreceptor response
        center_sum = 0.0
        if len(self.center_photoreceptors) > 0:
            for receptor in self.center_photoreceptors:
                # Photoreceptors hyperpolarize with light
                # More light = more negative = more "response"
                # We want: light ON = positive input
                response = receptor.get_activation()
                center_sum += response
            center_sum /= len(self.center_photoreceptors)
        
        # Surround: average photoreceptor response
        surround_sum = 0.0
        if len(self.surround_photoreceptors) > 0:
            for receptor in self.surround_photoreceptors:
                response = receptor.get_activation()
                surround_sum += response
            surround_sum /= len(self.surround_photoreceptors)
        
        # Center-surround: simple subtraction
        # ON cells: excited by light in center
        # OFF cells: inhibited by light in center
        center_surround = self.polarity * (center_sum * self.center_weight + 
                                          surround_sum * self.surround_weight)
        
        return center_surround * 20.0  # Scale to mV range
    
    def update(self, dt: Optional[float] = None, current_time: float = 0.0):
        """Update bipolar cell with receptive field computation."""
        if dt is None:
            dt = self._dt * 1000
        
        # Get receptive field input
        rf_input = self.compute_receptive_field_input()
        
        # Add to expectation bias for total current
        total_current = rf_input + self.expectation_bias
        
        # Simple integration (bipolar cells don't spike much, graded potentials)
        dv = ((self.v_rest + total_current - self.v_membrane) / self.tau_membrane) * dt
        self.v_membrane += dv
        
        # Bipolar cells rarely spike, mostly graded responses
        # But we can still track threshold crossings
        if self.v_membrane > self.v_threshold:
            self.is_spiking = True
            self.time_since_spike = 0.0
        else:
            self.is_spiking = False
            self.time_since_spike += dt
        
        return self.v_membrane
    
    def get_state(self) -> Dict:
        """Get bipolar cell state."""
        state = super().get_state()
        state.update({
            'cell_type': self.cell_type,
            'position': self.position,
            'eye': self.eye,
            'eccentricity': float(self.eccentricity),
            'pathway': self.pathway,
            'center_size': len(self.center_photoreceptors),
            'surround_size': len(self.surround_photoreceptors)
        })
        return state


class BipolarLayer:
    """
    Layer of bipolar cells with variable pooling based on eccentricity.
    Creates both ON and OFF center cells.
    """
    
    def __init__(self, retina: FovealRetina, receptor_type: str = 'all'):
        """
        Create bipolar cell layer from retinal input.
        
        Args:
            retina: FovealRetina instance
            receptor_type: Which receptors to connect ('rods', 'cones', 'all')
        """
        self.retina = retina
        self.receptor_type = receptor_type
        
        # Bipolar cells for each eye
        self.bipolar_cells = {
            'left': {'ON': [], 'OFF': []},
            'right': {'ON': [], 'OFF': []}
        }
        
        # Create bipolar cells
        self._create_bipolar_cells()
        
        # Statistics
        self.total_bipolar_cells = sum(
            len(cells) for eye_data in self.bipolar_cells.values()
            for cells in eye_data.values()
        )
        
        print(f"Created {self.total_bipolar_cells} bipolar cells")
    
    def _create_bipolar_cells(self):
        """Create bipolar cells with receptive fields."""
        for eye in ['left', 'right']:
            self._create_eye_bipolar_cells(eye)
    
    def _create_eye_bipolar_cells(self, eye: str):
        """Create bipolar cells for one eye."""
        # Get all photoreceptors for this eye
        if self.receptor_type == 'rods':
            receptor_types = ['rods']
        elif self.receptor_type == 'cones':
            receptor_types = ['red_cones', 'green_cones', 'blue_cones']
        else:
            receptor_types = ['rods', 'red_cones', 'green_cones', 'blue_cones']
        
        # Collect all receptors with positions
        all_receptors = []
        for rtype in receptor_types:
            for receptor in self.retina.photoreceptors[eye][rtype]:
                all_receptors.append(receptor)
        
        # Create spatial grid for bipolar cell centers
        # Adaptive resolution based on eccentricity
        bipolar_positions = self._generate_bipolar_positions()
        
        for position in bipolar_positions:
            x, y = position
            eccentricity = np.sqrt(x**2 + y**2)
            
            # Get pooling parameters for this eccentricity
            pooling_params = self.retina.get_pooling_params(eccentricity)
            pathway = pooling_params['pathway']
            
            # Determine pooling radius based on receptor dominance
            # Cones: smaller receptive fields
            # Rods: larger receptive fields
            if self.receptor_type == 'rods':
                pool_radius = pooling_params['rod_pool_size'] * 0.05
            elif self.receptor_type == 'cones':
                pool_radius = pooling_params['cone_pool_size'] * 0.03
            else:
                # Mixed: intermediate
                pool_radius = pooling_params['cone_pool_size'] * 0.04
            
            # Create ON-center cell
            on_cell = BipolarCell('ON', position, eye, eccentricity, pathway)
            self._connect_receptive_field(on_cell, all_receptors, pool_radius)
            
            if len(on_cell.center_photoreceptors) > 0:  # Only add if connected
                self.bipolar_cells[eye]['ON'].append(on_cell)
            
            # Create OFF-center cell
            off_cell = BipolarCell('OFF', position, eye, eccentricity, pathway)
            self._connect_receptive_field(off_cell, all_receptors, pool_radius)
            
            if len(off_cell.center_photoreceptors) > 0:
                self.bipolar_cells[eye]['OFF'].append(off_cell)
    
    def _generate_bipolar_positions(self) -> List[Tuple[float, float]]:
        """
        Generate positions for bipolar cells (adaptive density).
        IMPROVED: Better tiling with overlapping receptive fields.
        """
        positions = []
        
        # Fovea: DENSE sampling with overlap (increased from 16 to 24)
        fovea_grid = 24
        for i in range(fovea_grid):
            for j in range(fovea_grid):
                x = (i / (fovea_grid - 1)) * 2 * self.retina.fovea_radius - self.retina.fovea_radius
                y = (j / (fovea_grid - 1)) * 2 * self.retina.fovea_radius - self.retina.fovea_radius
                positions.append((x, y))
        
        # Periphery: DENSER sampling (increased from 12 to 20)
        periphery_grid = 20
        for i in range(periphery_grid):
            for j in range(periphery_grid):
                x = (i / (periphery_grid - 1)) * 2.0 - 1.0
                y = (j / (periphery_grid - 1)) * 2.0 - 1.0
                
                eccentricity = np.sqrt(x**2 + y**2)
                
                # Skip foveal region (already covered)
                if eccentricity > self.retina.fovea_radius:
                    positions.append((x, y))
        
        return positions
    
    def _connect_receptive_field(self, bipolar: BipolarCell, 
                                 receptors: List[TemporalPhotoreceptor],
                                 pool_radius: float):
        """Connect photoreceptors to bipolar cell's receptive field."""
        bx, by = bipolar.position
        
        # Center: receptors within pool_radius
        # Surround: receptors within pool_radius * 2 but outside center
        surround_radius = pool_radius * 2.0
        
        for receptor in receptors:
            rx, ry = receptor.position
            distance = np.sqrt((bx - rx)**2 + (by - ry)**2)
            
            if distance < pool_radius:
                # Center
                bipolar.add_center_receptor(receptor)
            elif distance < surround_radius:
                # Surround
                bipolar.add_surround_receptor(receptor)
    
    def update(self, dt: float = 1.0):
        """Update all bipolar cells."""
        for eye_data in self.bipolar_cells.values():
            for cell_list in eye_data.values():
                for cell in cell_list:
                    cell.update(dt, self.retina.current_time)
    
    def get_activity_map(self, eye: str, cell_type: str = 'ON') -> np.ndarray:
        """Get activity map for bipolar cells."""
        grid_size = self.retina.grid_size
        activity_map = np.zeros((grid_size, grid_size))
        count_map = np.zeros((grid_size, grid_size))
        
        cells = self.bipolar_cells[eye][cell_type]
        
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
            'type': 'bipolar_layer',
            'receptor_type': self.receptor_type,
            'total_bipolar_cells': self.total_bipolar_cells
        }
        
        for eye in ['left', 'right']:
            eye_summary = {}
            for cell_type in ['ON', 'OFF']:
                cells = self.bipolar_cells[eye][cell_type]
                
                # Count by pathway
                p_cells = sum(1 for c in cells if c.pathway == 'P')
                m_cells = sum(1 for c in cells if c.pathway == 'M')
                
                # Average activity
                total_activity = sum(c.get_activation() for c in cells)
                mean_activity = total_activity / len(cells) if len(cells) > 0 else 0
                
                eye_summary[cell_type] = {
                    'count': len(cells),
                    'P_pathway': p_cells,
                    'M_pathway': m_cells,
                    'mean_activity': mean_activity
                }
            
            summary[eye] = eye_summary
        
        return summary
    
    def get_state(self) -> Dict:
        """Get complete state."""
        return {
            'summary': self.get_summary(),
            'bipolar_cells': {
                eye: {
                    cell_type: [c.get_state() for c in cells]
                    for cell_type, cells in eye_data.items()
                }
                for eye, eye_data in self.bipolar_cells.items()
            }
        }

