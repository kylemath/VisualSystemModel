# Test Files

All test scripts and validation code for the neural network system.

## Test Files

### `test_temporal_system.py` ⭐ (Primary)
**Comprehensive test suite for the complete temporal system**

Tests:
- Foveal retina creation and receptor distribution
- Bipolar cell layer with P/M pathways
- Temporal dynamics (100 time steps)
- Simulated webcam input
- Receptive field structure
- Neuron temporal properties

Run:
```bash
cd /Users/kylemathewson/VisualSystemModel
source venv/bin/activate
python test_files/test_temporal_system.py
```

### `test_retina.py` (Legacy)
**Test suite for original simple retinal system**

Tests:
- Basic retinal layer functionality
- Uniform photoreceptor distribution
- Simple image processing
- Activity map generation

Run:
```bash
python test_files/test_retina.py
```

## Running Tests

### Quick Test (Temporal System)
```bash
python test_files/test_temporal_system.py
```

Expected output: All 6 tests pass in ~10-15 seconds

### Legacy Test (Simple System)
```bash
python test_files/test_retina.py
```

## Test Coverage

✅ **Currently Tested:**
- Photoreceptor creation and distribution
- Foveal structure (eccentricity-based)
- Bipolar cells with center-surround
- Temporal dynamics and continuous time
- Webcam simulation
- Receptive field connectivity

⏳ **Future Tests Needed:**
- Ganglion cells (when implemented)
- LGN layer (when implemented)
- Learning mechanisms (when added)
- Full integration tests
- Performance benchmarks
- Stress tests at large scales

## Adding New Tests

When adding new features, add corresponding tests to `test_temporal_system.py`:

1. Create test function: `def test_new_feature():`
2. Add to `run_all_tests()` function
3. Run full test suite to verify
4. Document in this README

## Test Best Practices

- ✅ Test each layer independently
- ✅ Test integration between layers
- ✅ Validate temporal dynamics
- ✅ Check biological realism
- ✅ Verify performance metrics
- ❌ Don't commit broken tests
- ❌ Don't skip tests before deployment

