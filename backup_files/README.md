# Backup Files - Legacy Code

This folder contains **original/superseded code** that has been replaced by newer implementations.

## Why These Files Are Here

These files were the initial implementations that have been superseded by the temporal system. They're kept for reference and backward compatibility if needed.

## Legacy System (Original - Simple Retina)

### Core Files
- **`neuron.py`** - Original simple neuron classes
  - Replaced by: `temporal_neuron.py` (with temporal dynamics)
  
- **`retina.py`** - Uniform retinal grid (no foveal structure)
  - Replaced by: `foveal_retina.py` (eccentricity-based)
  
- **`server.py`** - Flask server for simple system
  - Replaced by: `temporal_server.py` (continuous updates)

### Interface
- **`templates/index.html`** - Original web interface
  - Replaced by: `templates/temporal_index.html` (advanced features)

### Visualization
- **`visualization_3d.py`** - Heavy Plotly 3D visualizations
  - Replaced by: `visualization_2d.py` (lightweight 2D canvas)
  - Reason: Performance issues, NaN errors, high RAM/GPU usage
  - Future: Will implement efficient WebGL "brain view" when needed

## When to Use Legacy Files

❌ **Don't use for new development** - Use the temporal system instead

✅ **Use for:**
- Understanding the evolution of the codebase
- Learning simpler concepts before temporal dynamics
- Quick demos without temporal complexity
- Backward compatibility testing

## Migration Path

If you need the simple system:
```bash
# Copy from backup to main directory
cp backup_files/server.py .
cp backup_files/neuron.py .
cp backup_files/retina.py .
cp backup_files/templates/index.html templates/

# Run simple system
python server.py
```

## Differences: Legacy vs Temporal

| Feature | Legacy (Backup) | Temporal (Current) |
|---------|----------------|-------------------|
| Neuron model | Simple activation | Membrane potential, spikes |
| Time | Discrete frames | Continuous time |
| Retina | Uniform grid | Foveal structure |
| Photoreceptors | Equal distribution | Eccentricity-based |
| Bipolar cells | ❌ Not implemented | ✅ Center-surround |
| Input | Static patterns | Webcam streaming |
| Visualization | 2D canvas | 2D + 3D interactive |
| Performance | ~10K neurons | ~20K+ neurons |

## Best Practice Going Forward

When creating improved versions of existing files:

**Option 1: Update in place** (Preferred for minor changes)
- Edit the existing file
- Git tracks the changes
- Clean history

**Option 2: Create new + Move old to backup** (For major rewrites)
- Create new file with improved implementation
- Move old file to `backup_files/`
- Update imports/references
- Clear which is current

**❌ Avoid:** Having multiple "current" versions in main directory

