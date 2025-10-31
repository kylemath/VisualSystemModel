"""
Efficient 2D visualization system for neural layers and activity.
Uses color to represent depth/activity without heavy 3D rendering.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json


class NeuralVisualizer2D:
    """
    2D visualization system for neural network architecture.
    Uses color-coded 2D plots instead of heavy 3D rendering.
    """
    
    def __init__(self):
        self.figures = {}
    
    def visualize_photoreceptor_distribution(self, retina, eye: str = 'left') -> Dict:
        """
        Visualize 2D distribution of photoreceptors by type.
        Color represents eccentricity or activity.
        
        Returns: Simple dict for canvas rendering (not Plotly)
        """
        receptor_types = ['rods', 'red_cones', 'green_cones', 'blue_cones']
        
        data = {}
        for receptor_type in receptor_types:
            receptors = retina.photoreceptors[eye][receptor_type]
            
            if len(receptors) == 0:
                data[receptor_type] = {'positions': [], 'activities': [], 'eccentricities': []}
                continue
            
            positions = [r.position for r in receptors]
            activities = [r.get_activation() for r in receptors]
            eccentricities = [r.eccentricity for r in receptors]
            
            data[receptor_type] = {
                'positions': positions,
                'activities': activities,
                'eccentricities': eccentricities,
                'count': len(receptors)
            }
        
        return data
    
    def visualize_receptive_fields_2d(self, bipolar_layer, eye: str = 'left', 
                                     num_samples: int = 10) -> Dict:
        """
        Visualize receptive fields in 2D.
        Returns data for canvas rendering.
        """
        on_cells = bipolar_layer.bipolar_cells[eye]['ON']
        off_cells = bipolar_layer.bipolar_cells[eye]['OFF']
        
        if len(on_cells) == 0:
            return {'cells': []}
        
        # Sample cells
        sample_cells = []
        for cell_list, cell_type in [(on_cells, 'ON'), (off_cells, 'OFF')]:
            if len(cell_list) > num_samples // 2:
                indices = np.linspace(0, len(cell_list)-1, num_samples//2, dtype=int)
                for idx in indices:
                    cell = cell_list[idx]
                    
                    # Get center and surround positions
                    center_positions = [r.position for r in cell.center_photoreceptors]
                    surround_positions = [r.position for r in cell.surround_photoreceptors]
                    
                    sample_cells.append({
                        'type': cell_type,
                        'position': cell.position,
                        'center_positions': center_positions,
                        'surround_positions': surround_positions,
                        'activity': float(cell.get_activation()),
                        'pathway': cell.pathway,
                        'eccentricity': float(cell.eccentricity)
                    })
        
        return {'cells': sample_cells}
    
    def create_layer_heatmap_data(self, activity_map: np.ndarray) -> Dict:
        """
        Create heatmap data from activity map.
        Simple format for canvas rendering.
        """
        # Ensure no NaN or inf values
        activity_map = np.nan_to_num(activity_map, nan=0.0, posinf=1.0, neginf=0.0)
        
        return {
            'data': activity_map.tolist(),
            'width': activity_map.shape[1],
            'height': activity_map.shape[0],
            'min': float(np.min(activity_map)),
            'max': float(np.max(activity_map)),
            'mean': float(np.mean(activity_map))
        }
    
    def visualize_network_overview(self, retina, bipolar_layer=None) -> Dict:
        """
        Create 2D network overview showing all layers.
        Uses color to indicate layer and activity.
        """
        layers_data = []
        
        # Layer 0: Photoreceptors
        for eye in ['left', 'right']:
            for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
                receptors = retina.photoreceptors[eye][receptor_type]
                if len(receptors) > 0:
                    positions = [(r.position[0], r.position[1], 0) for r in receptors]  # z=0
                    activities = [r.get_activation() for r in receptors]
                    
                    layers_data.append({
                        'layer': 'photoreceptors',
                        'type': receptor_type,
                        'eye': eye,
                        'positions': positions[:1000],  # Limit for performance
                        'activities': activities[:1000],
                        'count': len(receptors)
                    })
        
        # Layer 1: Bipolar cells
        if bipolar_layer:
            for eye in ['left', 'right']:
                for cell_type in ['ON', 'OFF']:
                    cells = bipolar_layer.bipolar_cells[eye][cell_type]
                    if len(cells) > 0:
                        positions = [(c.position[0], c.position[1], 1) for c in cells]  # z=1
                        activities = [c.get_activation() for c in cells]
                        
                        layers_data.append({
                            'layer': 'bipolar',
                            'type': cell_type,
                            'eye': eye,
                            'positions': positions[:1000],
                            'activities': activities[:1000],
                            'count': len(cells)
                        })
        
        return {'layers': layers_data}


def generate_safe_json(data: Dict) -> str:
    """
    Convert data to JSON, handling NaN and inf values safely.
    """
    def clean_value(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return obj
        elif isinstance(obj, dict):
            return {k: clean_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_value(v) for v in obj]
        return obj
    
    cleaned_data = clean_value(data)
    return json.dumps(cleaned_data)


def create_network_summary_2d(retina, bipolar_layer=None) -> Dict:
    """
    Create simple 2D summary for network architecture.
    Returns statistics and simplified positions for rendering.
    """
    summary = {
        'layers': []
    }
    
    # Retina summary
    retina_summary = retina.get_summary()
    summary['layers'].append({
        'name': 'Retina',
        'z_level': 0,
        'neuron_count': retina.total_photoreceptors,
        'types': list(retina_summary['left'].keys()),
        'stats': retina_summary
    })
    
    # Bipolar summary
    if bipolar_layer:
        bipolar_summary = bipolar_layer.get_summary()
        summary['layers'].append({
            'name': 'Bipolar Cells',
            'z_level': 1,
            'neuron_count': bipolar_layer.total_bipolar_cells,
            'types': ['ON', 'OFF'],
            'stats': bipolar_summary
        })
    
    return summary


# Future: Brain view concept
"""
FUTURE IMPLEMENTATION: Physical 3D Brain View

Concept: Show neurons as physical objects in 3D space
- Neurons: Small spheres at physical positions
- Axons: Lines connecting neurons
- Layers: Organized like "eggs in a carton"
- Can zoom/rotate to see structure

Implementation approach:
- Use Three.js or WebGL (not Plotly)
- Efficient instanced rendering for many neurons
- LOD (Level of Detail) - show fewer neurons when zoomed out
- Chunk loading - only render visible regions

Not needed now, but keep in mind for future.
"""

