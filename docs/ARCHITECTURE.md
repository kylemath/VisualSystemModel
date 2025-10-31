# Neural Architecture Documentation

## Overview

This is a modular, biologically-inspired neural network system that mirrors the human brain's sensory and motor processing pathways. We're building it piece by piece, starting with the visual system.

## Design Philosophy

### 1. Neuron-as-Object
Each neuron is represented as a Python object with:
- Unique ID
- Activation state
- Membrane potential
- Input/output connections with weights
- Biological properties (threshold, refractory period)

### 2. Efficient Scalability
- Weak references prevent circular reference issues
- NumPy arrays for efficient weight storage
- Connection indices instead of duplicating neuron data
- Modular layer architecture

### 3. Biological Realism
We implement simplified but biologically-plausible mechanisms:
- Photoreceptor spectral sensitivity curves
- Membrane potential and threshold-based activation
- Refractory periods
- Proper layer connectivity patterns

## Current Implementation

### Layer 1: Retinal Photoreceptors

The foundation of our visual system.

#### Structure
- **Two eyes** (binocular vision)
- **n×n grid** of photoreceptors per eye (configurable size)
- **Four photoreceptor types** at each grid position:

1. **Rods** (Scotopic vision)
   - Peak sensitivity: 498nm (blue-green)
   - Respond to luminance
   - High sensitivity, no color discrimination

2. **Red Cones (L-cones)**
   - Peak sensitivity: 564nm
   - Long wavelength sensitive
   - Primary red detection with some green/yellow response

3. **Green Cones (M-cones)**
   - Peak sensitivity: 534nm
   - Medium wavelength sensitive
   - Green detection with some red response

4. **Blue Cones (S-cones)**
   - Peak sensitivity: 420nm
   - Short wavelength sensitive
   - Blue detection

#### Biological Accuracy

The photoreceptors implement:
- **Spectral sensitivity curves**: Different response curves for each receptor type
- **Saturation**: Maximum response limits
- **Dark adaptation**: Sensitivity adjustment (placeholder for future enhancement)
- **Spatial arrangement**: Grid-based positioning (simplified from actual retinal mosaic)

#### Input Processing

Images are processed as RGB values at each spatial position:
- Each photoreceptor type responds according to its spectral sensitivity
- Output is activation level (0.0 to 1.0)
- Activity maps can be visualized independently for each receptor type

## Next Layers (Planned)

### Layer 2: Bipolar Cells
- First processing layer after photoreceptors
- ON-center and OFF-center cells
- Initial edge detection and contrast enhancement
- Spatial pooling (multiple photoreceptors → one bipolar cell)

### Layer 3: Horizontal Cells
- Lateral inhibition
- Contrast enhancement
- Local gain control

### Layer 4: Ganglion Cells
- Further feature extraction
- Motion detection (magnocellular pathway)
- Fine detail processing (parvocellular pathway)
- Output neurons that form the optic nerve

### Layer 5: Lateral Geniculate Nucleus (LGN)
- Relay station in thalamus
- Attention modulation
- Feedback from cortex

### Later: Visual Cortex Layers
- V1: Orientation, spatial frequency
- V2-V5: Complex features, motion, color
- Integration with other sensory systems

## Code Organization

```
neuron.py          # Base neuron classes
retina.py          # Retinal photoreceptor layer
server.py          # Flask API server
templates/
  index.html       # Web visualization interface
test_retina.py     # Testing and validation
```

## Visualization

The web interface provides:
- **Flowchart view**: High-level architecture overview
- **Layer detail views**: Expandable detailed visualization
- **Activity maps**: Real-time neuron activation patterns
- **Statistics**: Layer-specific metrics and analytics

## Technical Details

### Connection Storage

Connections are stored efficiently:
```python
# Instead of full neuron objects:
self.input_neurons: List[weakref.ref]  # Weak references
self.input_weights: np.ndarray         # NumPy array

# This allows:
# - Garbage collection of unused neurons
# - Fast vectorized weight operations
# - Minimal memory overhead
```

### Activity Computation

```python
def compute_activation(self):
    # Sum weighted inputs
    total_input = sum(neuron.activity * weight 
                     for neuron, weight in connections)
    
    # Sigmoid activation
    activity = 1.0 / (1.0 + exp(-total_input))
    
    # Threshold detection for spiking
    if activity > threshold:
        spike()
```

### Photoreceptor Response

```python
# Simulate spectral sensitivity
if receptor_type == 'red':
    response = 0.8 * R + 0.3 * G  # Overlapping sensitivities

# Apply intensity and saturation
response *= intensity
response = min(response, saturation_level)
```

## Future Enhancements

### Near-term
1. Add bipolar cell layer with center-surround receptive fields
2. Implement temporal dynamics (not just spatial)
3. Add real image input (file upload or camera)
4. Implement saccadic eye movements

### Medium-term
1. Complete retinal processing chain
2. Add LGN and V1 cortical layers
3. Implement attention mechanisms
4. Add learning rules (Hebbian, STDP)

### Long-term
1. Full dorsal and ventral visual streams
2. Integration with motor control
3. Memory systems (hippocampus)
4. Multi-modal sensory integration
5. Executive function (prefrontal cortex)

## Testing

Run the test suite:
```bash
python test_retina.py
```

This validates:
- Neuron creation and connectivity
- Photoreceptor response curves
- Image processing pipeline
- Activity map generation
- System statistics

## Web Interface Usage

1. **Initialize System**: Set grid size and initialize the neural network
2. **Select Pattern**: Choose a test pattern (or upload an image in future versions)
3. **Process Input**: Run the input through the retinal layer
4. **Click Layer Cards**: Expand to see detailed visualizations
5. **Inspect Activity**: View individual photoreceptor type responses

## Performance Considerations

Current implementation can handle:
- Grid sizes up to 64×64 (16,384 photoreceptors per eye)
- Real-time processing for most operations
- Interactive visualization without lag

For larger scales:
- Consider GPU acceleration (CuPy/PyTorch)
- Implement sparse connectivity patterns
- Use batch processing for multiple images
- Add progressive rendering for visualization

## Discussion Points

Let's talk through:
1. **Bipolar cell implementation**: How should we organize center-surround receptive fields?
2. **Connection patterns**: Random, structured, or learned?
3. **Temporal dynamics**: Frame-by-frame processing or continuous time?
4. **Learning mechanisms**: When and how should we add plasticity?
5. **Scale**: What grid sizes do we want to support?

---

**Current Status**: ✅ Retinal layer complete and functional
**Next Step**: Design and implement bipolar cell layer

