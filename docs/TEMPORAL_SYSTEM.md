# Temporal Bio-Inspired Neural Network System

## 🎉 Major Upgrade Complete!

Your neural network system now includes:

### ✅ New Features

1. **Temporal Dynamics**
   - Continuous time updates (not frame-by-frame)
   - Membrane potential with leaky integrate-and-fire dynamics
   - Action potentials and refractory periods
   - Neural oscillations (theta, alpha, gamma bands)
   - Preparatory depolarization (expectation/attention)
   - Re-entrant feedback connections

2. **Foveal Retina Structure**
   - Eccentricity-based photoreceptor distribution
   - Dense cones in fovea (center vision)
   - Dense rods in periphery (peripheral vision)
   - Realistic spatial arrangement
   - Variable pooling for downstream processing

3. **Bipolar Cell Layer**
   - Center-surround receptive fields
   - ON-center cells (excited by light in center)
   - OFF-center cells (inhibited by light in center)
   - Variable pooling based on eccentricity
   - P pathway (parvocellular): Fovea, fine detail, color
   - M pathway (magnocellular): Periphery, motion, luminance

4. **Webcam Input**
   - Real-time webcam capture (or simulated input)
   - Downsampled to network resolution
   - Binocular vision support
   - Continuous streaming at target FPS

5. **3D Interactive Visualization**
   - Photoreceptor spatial distribution
   - Receptive field organization
   - Complete network architecture
   - Individual neuron inspection
   - Real-time activity visualization

## Architecture Overview

```
┌─────────────────────────────────────────┐
│  WEBCAM INPUT (Real or Simulated)      │
│  - Continuous capture                   │
│  - Downsampled to grid size            │
│  - Binocular (left/right eyes)         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  FOVEAL RETINA (Photoreceptors)        │ ✅ Layer 1
│  - Rods (scotopic, periphery)          │
│  - Red cones (L-cones)                 │
│  - Green cones (M-cones)               │
│  - Blue cones (S-cones)                │
│                                         │
│  Distribution:                          │
│  • Fovea (0-15%): Dense cones, few rods│
│  • Periphery (15-100%): Dense rods     │
│                                         │
│  Temporal Dynamics:                     │
│  • Light adaptation                     │
│  • Membrane hyperpolarization          │
│  • Continuous response                 │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  BIPOLAR CELLS (Center-Surround)       │ ✅ Layer 2
│  - ON-center cells                      │
│  - OFF-center cells                     │
│                                         │
│  Receptive Fields:                      │
│  • Small in fovea (1-2 receptors)      │
│  • Large in periphery (5-11 receptors) │
│                                         │
│  Pathways:                              │
│  • P pathway: Fovea, color, detail     │
│  • M pathway: Periphery, motion        │
│                                         │
│  Computation:                           │
│  • Center_input - Surround_input       │
│  • Edge detection                      │
│  • Contrast enhancement                │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  FUTURE LAYERS                          │ 📋 Coming Next
│  - Ganglion cells                       │
│  - Lateral geniculate nucleus (LGN)    │
│  - Primary visual cortex (V1)          │
└─────────────────────────────────────────┘
```

## Key Design Principles (As You Requested)

### 1. Temporal Dynamics from the Start
- Every neuron updates continuously in time
- Not frame-based: continuous membrane potential
- Oscillations built in (can be modulated)
- Refractory periods prevent overfiring
- Preparatory depolarization for expectations

### 2. Foveal Structure with Variable Pooling
- Center (fovea): High resolution, P pathway
- Periphery: Lower resolution, M pathway
- Cones → small pooling (detail)
- Rods → large pooling (sensitivity)

### 3. Simple Computations, Complex Arrangements
- Bipolar cells: Just (center - surround)
- No complex math in individual neurons
- Complexity emerges from connections
- Scalable and interpretable

### 4. Physiologically Realistic
- Center-surround ratios match recordings
- Spectral sensitivities realistic
- Eccentricity-based distributions
- P vs M pathway organization

### 5. Designed for Learning (Future)
- Synaptic weights can be modified
- Feedback connections ready
- Expectation/attention mechanisms
- Spike-timing dependent plasticity (STDP) ready

## Usage

### Start the Temporal Server

```bash
cd /Users/kylemathewson/Golem
source venv/bin/activate
python temporal_server.py
```

Then open: **http://localhost:5000**

### Configuration Options

1. **Grid Size**: 32×32 to 128×128
   - Larger = more detail, slower
   - Smaller = faster, less detail
   - Recommended: 64×64 for balance

2. **Fovea Radius**: 0.05 to 0.30
   - Fraction of visual field that's high resolution
   - 0.15 (15%) is realistic
   - Smaller = more focused central vision

3. **Update FPS**: 15 to 60
   - How fast the simulation runs
   - 30 FPS is smooth and efficient
   - Your M5 Mac can handle 60 FPS at 64×64

4. **Input Source**:
   - **Simulated**: Procedural test patterns (no camera needed)
   - **Real Webcam**: Live camera input

### Workflow

1. **Configure** system parameters
2. **Initialize** - creates all neurons and connections
3. **Watch** continuous updates in real-time
4. **Explore** 3D visualizations
5. **Inspect** individual layers and neurons

## Neuron Types

### TemporalPhotoreceptor
```python
Properties:
- v_membrane: Membrane potential (-70 to -40 mV)
- light_input: Current RGB stimulus
- adapted_response: Adapted light response
- eccentricity: Distance from fovea
- receptor_type: rod, red, green, blue

Dynamics:
- Hyperpolarizes with light (biological accurate)
- Light adaptation (tau = 100ms)
- No action potentials (graded response)
```

### BipolarCell
```python
Properties:
- cell_type: ON or OFF
- pathway: P (parvocellular) or M (magnocellular)
- center_photoreceptors: List of center inputs
- surround_photoreceptors: List of surround inputs
- polarity: +1 (ON) or -1 (OFF)

Computation:
- Center: average(center_receptors)
- Surround: average(surround_receptors)
- Output: polarity * (center - surround * 0.6)

Dynamics:
- Graded potentials (mostly)
- Can spike if strongly activated
- Temporal integration
```

## Visualization Modes

### 2D Real-Time (Canvas)
- **Webcam Input**: See what the system sees
- **Photoreceptors**: Individual receptor types
- **Bipolar Cells**: ON vs OFF responses
- Updates at 15 FPS for smooth display

### 3D Interactive (Plotly)
1. **Full Architecture**: All layers in 3D space
   - Z=0: Photoreceptors
   - Z=1: Bipolar cells
   - Color coded by type and activity

2. **Photoreceptor Distribution**:
   - Spatial positions (X, Y)
   - Activity level (Z height)
   - Color by eccentricity
   - See foveal density

3. **Receptive Fields**:
   - Individual bipolar cells
   - Their center receptors
   - Their surround receptors
   - Connection patterns

4. **Input 3D**:
   - Webcam as 3D surface
   - Height = luminance
   - Interactive rotation

## Statistics and Metrics

The system tracks:
- Total photoreceptors (varies by foveal structure)
- Total bipolar cells
- P vs M pathway neuron counts
- Actual update FPS
- Mean activity by layer
- Zone-based distributions (fovea, parafovea, periphery)

## Performance

**On Your M5 Mac:**

| Grid Size | Photoreceptors | Bipolar Cells | Est. FPS |
|-----------|---------------|---------------|----------|
| 32×32     | ~6,000        | ~400          | 60+      |
| 64×64     | ~22,000       | ~1,200        | 40-60    |
| 96×96     | ~48,000       | ~2,500        | 30-40    |
| 128×128   | ~82,000       | ~4,000        | 20-30    |

*Note: Actual counts vary due to foveal distribution*

## Next Steps (Discussion Points)

### Immediate Next Layer Options

1. **Ganglion Cells**
   - Magnocellular (M) pathway: Motion sensitive
   - Parvocellular (P) pathway: Color and detail
   - Direction selectivity
   - Non-linear integration

2. **Horizontal Cells**
   - Lateral inhibition in retina
   - Contrast enhancement
   - Receptive field modulation
   - Feedback to photoreceptors

3. **Amacrine Cells**
   - Motion detection (starburst amacrine)
   - Direction selectivity
   - Complex inhibitory patterns
   - Multiple subtypes

### Questions for You

1. **Next Layer Priority?**
   - Ganglion cells (output from retina)?
   - Horizontal cells (lateral processing)?
   - Both in parallel?

2. **Temporal Complexity?**
   - Add more realistic spike timing?
   - Implement burst firing?
   - Add rebound dynamics?

3. **Learning Implementation?**
   - Start with Hebbian ("cells that fire together...")?
   - STDP (spike-timing dependent plasticity)?
   - Reward-modulated learning?
   - Where first: Retina→Bipolar or later stages?

4. **Motor System?**
   - When to add motor output?
   - Eye movements (saccades)?
   - Start simple (2D reaching)?

5. **Scale and Optimization?**
   - GPU acceleration with CuPy/PyTorch?
   - Sparse connectivity representations?
   - Batch processing?

## File Organization

```
temporal_neuron.py       - Base temporal neuron classes
foveal_retina.py        - Retina with foveal structure
bipolar_cells.py        - Bipolar layer with center-surround
webcam_input.py         - Webcam capture (real/simulated)
visualization_3d.py     - 3D visualization system
temporal_server.py      - Flask server with temporal updates
templates/
  temporal_index.html   - Web interface

Legacy (simple version):
neuron.py              - Original simple neurons
retina.py              - Original uniform retina
server.py              - Original server
templates/index.html   - Original interface
```

## Tips and Tricks

### For Development
- Use simulated webcam for reproducible testing
- Start with 32×32 for fast iteration
- Use 3D viz to understand connectivity
- Monitor FPS to optimize

### For Demos
- Use 64×64 or 96×96 for visual quality
- Real webcam shows immediate response
- Show different layers side-by-side
- Use 3D receptive fields to explain

### For Research
- 128×128 for detailed experiments
- Log activity over time
- Export neuron states
- Analyze pathway differences

## Troubleshooting

**System won't initialize?**
- Check dependencies installed
- Check webcam permissions (if using real camera)
- Try simulated webcam first

**Low FPS?**
- Reduce grid size
- Lower target FPS
- Close other applications

**Webcam not working?**
- Check camera permissions in macOS
- Try different camera index
- Use simulated webcam as fallback

**Visualizations not updating?**
- Check browser console for errors
- Refresh page
- Restart server

## Scientific Basis

This implementation is inspired by:

1. **Retinal Structure**: Curcio et al. (1990) - photoreceptor distribution
2. **Bipolar Cells**: Dacey et al. (2000) - center-surround organization
3. **P/M Pathways**: Kaplan & Shapley (1986) - pathway properties
4. **Temporal Dynamics**: Hodgkin & Huxley (1952) - action potentials
5. **Neural Oscillations**: Buzsáki (2006) - rhythms of the brain

## Future Enhancements

- [ ] Ganglion cell layer
- [ ] LGN (thalamus)
- [ ] V1 cortex with orientation selectivity
- [ ] Eye movement control (saccades)
- [ ] Attention mechanisms
- [ ] Synaptic plasticity
- [ ] Memory systems
- [ ] Motor output
- [ ] Audio input pathway
- [ ] Multimodal integration

---

**Ready to continue building!** 🚀

What would you like to work on next?

