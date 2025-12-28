# Bio-Inspired Neural Network System

A modular, scalable neural network architecture based on the human brain's sensory and motor systems with **temporal dynamics**, **foveal structure**, and **real-time processing**.

## 🚀 Quick Start

```bash
cd /Users/kylemathewson/VisualSystemModel
source venv/bin/activate
python temporal_server.py
```

Then open **http://localhost:5000** in your browser!

👉 **See [START_HERE.md](START_HERE.md) for complete guide**

## ✨ Key Features

### Temporal Neural Dynamics
- **Continuous time** updates (not frame-based)
- Membrane potentials, action potentials, refractory periods
- Neural **oscillations** (theta, alpha, gamma bands)
- **Preparatory depolarization** for attention/expectations
- **Re-entrant feedback** connections

### Biologically Realistic Vision System

**Layer 1: Foveal Retina** (~11,000 to 80,000+ neurons)
- Eccentricity-based photoreceptor distribution
- Dense **cones in fovea** (center, high-resolution)
- Dense **rods in periphery** (edge vision, motion)
- 4 receptor types: Rods, Red/Green/Blue cones
- Light adaptation and temporal dynamics

**Layer 2: Bipolar Cells** (~1,200 to 4,000+ neurons)
- Center-surround receptive fields
- **ON-center** and **OFF-center** cells
- Variable pooling by eccentricity (1:1 to 11:1)
- **P pathway** (parvocellular): Color, detail, fovea
- **M pathway** (magnocellular): Motion, periphery

### Real-Time Input & Visualization
- **Webcam input** (real or simulated)
- **2D real-time** activity maps
- **3D interactive** network visualization
- **Individual neuron** inspection
- Runs at **15-60 FPS** on modern hardware

## Current Architecture

```
WEBCAM INPUT (Binocular)
       ↓
RETINA (4 photoreceptor types, foveal structure)
       ↓
BIPOLAR CELLS (Center-surround, P/M pathways)
       ↓
[Coming: Ganglion → LGN → V1 → Motor...]
```

## Current System

### Temporal System (Production)
- Full temporal dynamics
- Foveal retina structure  
- Bipolar cell layer
- Webcam input
- 3D visualization

```bash
python temporal_server.py
```

*Legacy simple system moved to `backup_files/` for reference*

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Testing

```bash
# Test temporal system (comprehensive)
python test_files/test_temporal_system.py

# Test original retina (legacy)
python test_files/test_retina.py
```

## Design Philosophy

✅ **Temporal from start**: Continuous time, not discrete frames
✅ **Foveal structure**: Variable pooling by eccentricity
✅ **Simple computations**: Center-surround, pooling, averaging
✅ **Complex arrangements**: Sophistication from connectivity
✅ **Physiologically realistic**: Based on neural recordings
✅ **Massively scalable**: Object-oriented, efficient
✅ **Interpretable**: Inspect individual neurons/layers
✅ **Ready for learning**: Synaptic weights modifiable

## Documentation

📚 **[View Complete Documentation Index](docs/index.html)** - Organized documentation hub

Quick Links:
- **[START_HERE.md](docs/START_HERE.md)** - Quick start guide
- **[TEMPORAL_SYSTEM.md](docs/TEMPORAL_SYSTEM.md)** - Complete technical docs
- **[SYSTEM_OVERVIEW.md](docs/SYSTEM_OVERVIEW.md)** - System summary & status
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Design principles
- **[QUICKSTART.md](docs/QUICKSTART.md)** - Original system guide

## Next Steps

See [START_HERE.md](docs/START_HERE.md) for discussion of:
- Ganglion cells (M/P pathways)
- LGN and V1 cortex
- Learning and plasticity
- Eye movements
- Motor output
- Scaling and optimization

