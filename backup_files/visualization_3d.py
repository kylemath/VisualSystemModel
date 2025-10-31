"""
3D visualization system for neural layers and activity.
Interactive visualizations using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
import json


class NeuralVisualizer3D:
    """
    3D visualization system for neural network architecture.
    """
    
    def __init__(self):
        self.figures = {}
    
    def visualize_photoreceptor_distribution(self, retina, eye: str = 'left') -> go.Figure:
        """
        Visualize 3D distribution of photoreceptors by type.
        Shows spatial layout and eccentricity-based density.
        """
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Rods', 'Red Cones', 'Green Cones', 'Blue Cones')
        )
        
        receptor_types = ['rods', 'red_cones', 'green_cones', 'blue_cones']
        colors = ['gray', 'red', 'green', 'blue']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for receptor_type, color, (row, col) in zip(receptor_types, colors, positions):
            receptors = retina.photoreceptors[eye][receptor_type]
            
            if len(receptors) == 0:
                continue
            
            # Extract positions and activities
            xs, ys, activities, eccentricities = [], [], [], []
            for receptor in receptors:
                x, y = receptor.position
                xs.append(x)
                ys.append(y)
                activities.append(receptor.get_activation())
                eccentricities.append(receptor.eccentricity)
            
            # Z-axis represents activity
            zs = activities
            
            # Color by eccentricity
            scatter = go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='markers',
                marker=dict(
                    size=3,
                    color=eccentricities,
                    colorscale='Viridis',
                    showscale=(row == 1 and col == 1),
                    colorbar=dict(title="Eccentricity") if (row == 1 and col == 1) else None,
                    opacity=0.8
                ),
                name=receptor_type,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'X: %{x:.2f}<br>Y: %{y:.2f}<br>' +
                             'Activity: %{z:.2f}<br>' +
                             '<extra></extra>'
            )
            
            fig.add_trace(scatter, row=row, col=col)
            
            # Update axes
            fig.update_scenes(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Activity',
                row=row, col=col
            )
        
        fig.update_layout(
            title=f'Photoreceptor Distribution ({eye.capitalize()} Eye)',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def visualize_receptive_fields(self, bipolar_layer, eye: str = 'left', 
                                   num_samples: int = 10) -> go.Figure:
        """
        Visualize receptive fields of bipolar cells in 3D.
        Shows center-surround organization.
        """
        fig = go.Figure()
        
        # Sample random bipolar cells
        on_cells = bipolar_layer.bipolar_cells[eye]['ON']
        off_cells = bipolar_layer.bipolar_cells[eye]['OFF']
        
        if len(on_cells) == 0:
            return fig
        
        # Sample cells from different eccentricities
        sample_cells = []
        for cell_list, cell_type in [(on_cells, 'ON'), (off_cells, 'OFF')]:
            if len(cell_list) > num_samples // 2:
                indices = np.linspace(0, len(cell_list)-1, num_samples//2, dtype=int)
                for idx in indices:
                    sample_cells.append((cell_list[idx], cell_type))
        
        # Visualize each cell's receptive field
        for i, (cell, cell_type) in enumerate(sample_cells):
            # Center receptors
            if len(cell.center_photoreceptors) > 0:
                center_x = [r.position[0] for r in cell.center_photoreceptors]
                center_y = [r.position[1] for r in cell.center_photoreceptors]
                center_z = [1.0] * len(center_x)  # Height = 1 for center
                
                color = 'green' if cell_type == 'ON' else 'red'
                
                fig.add_trace(go.Scatter3d(
                    x=center_x,
                    y=center_y,
                    z=center_z,
                    mode='markers',
                    marker=dict(size=4, color=color, opacity=0.8),
                    name=f'{cell_type} Center {i}',
                    legendgroup=f'cell_{i}',
                    showlegend=(i == 0)
                ))
            
            # Surround receptors
            if len(cell.surround_photoreceptors) > 0:
                surround_x = [r.position[0] for r in cell.surround_photoreceptors]
                surround_y = [r.position[1] for r in cell.surround_photoreceptors]
                surround_z = [0.5] * len(surround_x)  # Height = 0.5 for surround
                
                color = 'lightgreen' if cell_type == 'ON' else 'lightcoral'
                
                fig.add_trace(go.Scatter3d(
                    x=surround_x,
                    y=surround_y,
                    z=surround_z,
                    mode='markers',
                    marker=dict(size=2, color=color, opacity=0.5),
                    name=f'{cell_type} Surround {i}',
                    legendgroup=f'cell_{i}',
                    showlegend=False
                ))
            
            # Cell body position
            bx, by = cell.position
            fig.add_trace(go.Scatter3d(
                x=[bx],
                y=[by],
                z=[1.5],  # Above receptors
                mode='markers',
                marker=dict(size=8, color='blue' if cell_type == 'ON' else 'orange',
                           symbol='diamond', opacity=1.0),
                name=f'{cell_type} Cell {i}',
                legendgroup=f'cell_{i}',
                showlegend=False,
                hovertemplate=f'<b>{cell_type} Cell</b><br>' +
                             f'Position: ({bx:.2f}, {by:.2f})<br>' +
                             f'Eccentricity: {cell.eccentricity:.2f}<br>' +
                             f'Pathway: {cell.pathway}<br>' +
                             f'Center size: {len(cell.center_photoreceptors)}<br>' +
                             f'Surround size: {len(cell.surround_photoreceptors)}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Bipolar Cell Receptive Fields ({eye.capitalize()} Eye)<br>' +
                  '<sub>Blue/Orange diamonds = ON/OFF bipolar cells, ' +
                  'Green/Red = center receptors, Lighter = surround</sub>',
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Layer',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            height=700
        )
        
        return fig
    
    def visualize_activity_over_time(self, activity_history: Dict) -> go.Figure:
        """
        Visualize neural activity over time as animated 3D volume.
        
        Args:
            activity_history: Dict with keys 'time', 'layer_name', 'activity_maps'
        """
        fig = go.Figure()
        
        # This would show temporal evolution of activity
        # Implement based on recorded activity traces
        
        return fig
    
    def visualize_layer_connectivity(self, layers: List[Dict]) -> go.Figure:
        """
        Visualize connectivity between layers in 3D.
        Shows information flow through the network.
        """
        fig = go.Figure()
        
        # Each layer at different Z height
        z_positions = {layer['name']: i * 2.0 for i, layer in enumerate(layers)}
        
        for layer in layers:
            # Placeholder: would show actual neurons
            # For now, show layer as a plane
            pass
        
        fig.update_layout(
            title='Neural Network Architecture - Layer Connectivity',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Layer Depth'
            ),
            height=800
        )
        
        return fig
    
    def visualize_input_image_3d(self, image: np.ndarray) -> go.Figure:
        """
        Visualize input image in 3D with RGB channels.
        """
        height, width, _ = image.shape
        
        fig = go.Figure()
        
        # Create mesh for image surface
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Use luminance for Z
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        
        # Create RGB color array for surface
        # Flatten and format for plotly
        colors = []
        for i in range(height):
            row_colors = []
            for j in range(width):
                r, g, b = image[i, j]
                row_colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
            colors.append(row_colors)
        
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=luminance,
            surfacecolor=luminance,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Luminance'),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Luminance: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Input Image 3D Representation<br><sub>Height = luminance</sub>',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Luminance',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            height=600
        )
        
        return fig
    
    def create_interactive_neuron_inspector(self, neurons: List) -> go.Figure:
        """
        Create interactive visualization for inspecting individual neurons.
        Click on neuron to see detailed state.
        """
        fig = go.Figure()
        
        # Plot neurons in 3D space
        xs, ys, zs, activities, ids = [], [], [], [], []
        
        for neuron in neurons:
            if hasattr(neuron, 'position'):
                x, y = neuron.position
                z = neuron.eccentricity if hasattr(neuron, 'eccentricity') else 0
                
                xs.append(x)
                ys.append(y)
                zs.append(z)
                activities.append(neuron.get_activation())
                ids.append(neuron.id)
        
        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers',
            marker=dict(
                size=5,
                color=activities,
                colorscale='Hot',
                showscale=True,
                colorbar=dict(title='Activity'),
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=[f'Neuron {id}' for id in ids],
            hovertemplate='<b>%{text}</b><br>' +
                         'Position: (%{x:.2f}, %{y:.2f})<br>' +
                         'Activity: %{marker.color:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive Neuron Inspector<br><sub>Click neurons to inspect</sub>',
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Eccentricity',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            height=700
        )
        
        return fig


def generate_plotly_json(fig: go.Figure) -> str:
    """Convert Plotly figure to JSON for web embedding."""
    return json.dumps(fig.to_dict())


def create_network_architecture_3d(retina, bipolar_layer=None) -> go.Figure:
    """
    Create complete 3D visualization of network architecture.
    Shows all layers and connections.
    """
    fig = go.Figure()
    
    # Layer 1: Photoreceptors (z=0)
    for eye in ['left', 'right']:
        x_offset = -0.5 if eye == 'left' else 0.5
        
        for receptor_type in ['rods', 'red_cones', 'green_cones', 'blue_cones']:
            receptors = retina.photoreceptors[eye][receptor_type]
            
            if len(receptors) == 0:
                continue
            
            xs = [r.position[0] + x_offset for r in receptors]
            ys = [r.position[1] for r in receptors]
            zs = [0.0] * len(xs)
            activities = [r.get_activation() for r in receptors]
            
            color_map = {
                'rods': 'gray',
                'red_cones': 'red',
                'green_cones': 'green',
                'blue_cones': 'blue'
            }
            
            fig.add_trace(go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='markers',
                marker=dict(
                    size=2,
                    color=activities,
                    colorscale=[[0, 'black'], [1, color_map[receptor_type]]],
                    opacity=0.6
                ),
                name=f'{eye} {receptor_type}',
                showlegend=True
            ))
    
    # Layer 2: Bipolar cells (z=1.0)
    if bipolar_layer:
        for eye in ['left', 'right']:
            x_offset = -0.5 if eye == 'left' else 0.5
            
            for cell_type in ['ON', 'OFF']:
                cells = bipolar_layer.bipolar_cells[eye][cell_type]
                
                if len(cells) == 0:
                    continue
                
                xs = [c.position[0] + x_offset for c in cells]
                ys = [c.position[1] for c in cells]
                zs = [1.0] * len(xs)
                activities = [c.get_activation() for c in cells]
                
                color = 'cyan' if cell_type == 'ON' else 'magenta'
                
                fig.add_trace(go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=activities,
                        colorscale=[[0, 'black'], [1, color]],
                        opacity=0.8,
                        symbol='diamond'
                    ),
                    name=f'{eye} {cell_type}-bipolar',
                    showlegend=True
                ))
    
    fig.update_layout(
        title='Complete Neural Network Architecture (3D)<br>' +
              '<sub>Z=0: Photoreceptors, Z=1: Bipolar Cells</sub>',
        scene=dict(
            xaxis_title='X Position (Left/Right Eye)',
            yaxis_title='Y Position',
            zaxis_title='Layer',
            camera=dict(
                eye=dict(x=2.0, y=2.0, z=1.5)
            ),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=0.5)
        ),
        height=800,
        showlegend=True
    )
    
    return fig

