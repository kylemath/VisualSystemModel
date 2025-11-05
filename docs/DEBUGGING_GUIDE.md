# Debugging Guide for Temporal Server

## Problem
After refactoring `temporal_server.py`, webcam and simulated input are working, but the three cell layers (retinal, bipolar, and ganglion) are showing no activity - only black squares.

## Debugging Enhancements Added

### 1. Initialization Logging (`/api/initialize`)

**Location**: Lines 177-271

**What it logs**:
- Layer creation confirmation with neuron counts
- Verification of layer connections (retina→bipolar→ganglion)
- Test frame processing at initialization
- Initial activity ranges for all three layers

**What to look for**:
```
[INIT] ✓ Retina created: XXXXX photoreceptors
[INIT] ✓ Bipolar layer connected to retina
[INIT] Test retina output: range=[X.XXX, X.XXX]
[INIT] ✓ Retina producing activity  (or WARNING if zeros)
```

### 2. Update Loop Logging

**Location**: Lines 68-155

**What it logs** (every 30 frames ≈ 1 second):
- Input frame shapes and value ranges
- Retina activity after processing
- Bipolar layer ON cell activity
- Ganglion layer P cell activity

**What to look for**:
```
[UPDATE] Frame 0: Left shape=(64, 64), range=[0.000, 1.000]
[UPDATE] Retina activity: range=[0.000, 0.850], mean=0.234
[UPDATE] Bipolar ON activity: range=[0.000, 0.654], mean=0.145
[UPDATE] Ganglion P activity: range=[0.000, 0.432], mean=0.087
```

**Warning signs**:
- `range=[0.000, 0.000]` = All zeros, no activity
- `mean=0.000` = Layer not responding
- `WARNING - Frames are None!` = Input system broken
- `WARNING - X activity is None!` = Layer method failing

### 3. API State Endpoint Logging

**Location**: Lines 264-372

**What it logs** (on every `/api/state/current` request):
- Binary encoding mode
- Activity maps for all receptor types (rods, red/green/blue cones)
- Bipolar ON/OFF activity for both eyes
- Ganglion P/M activity for both eyes

**What to look for**:
```
[API] get_current_state called, binary=true
[API] Retina rods LEFT: shape=(64, 64), range=[0.000, 0.850]
[API] Bipolar ON LEFT: shape=(64, 64), range=[0.000, 0.654]
[API] Ganglion P LEFT: shape=(64, 64), range=[0.000, 0.432]
```

## Common Issues and Solutions

### Issue 1: All Zeros in Retina Output
**Symptoms**: 
- `[INIT] ⚠️ WARNING: Retina output is all zeros!`
- Retina range is `[0.000, 0.000]`

**Possible causes**:
1. Input frames are all black
2. `process_image()` not being called
3. Retina's `update()` method not applying dynamics

**Debug steps**:
- Check input frame ranges in `[UPDATE]` logs
- Verify retina's photoreceptor count is > 0
- Check if optimized retina imports correctly

### Issue 2: Bipolar Layer All Zeros
**Symptoms**:
- Retina has activity, but bipolar is zeros
- `[INIT] ⚠️ WARNING: Bipolar output is all zeros!`

**Possible causes**:
1. Bipolar layer not connected to retina properly
2. `update()` method not reading from retina
3. Receptive field connections broken

**Debug steps**:
- Look for `[INIT] ✓ Bipolar layer connected to retina`
- Check if bipolar layer's `update()` method fetches retina activity
- Verify optimized bipolar imports correctly

### Issue 3: Ganglion Layer All Zeros
**Symptoms**:
- Retina and bipolar have activity, but ganglion is zeros
- `[INIT] ⚠️ WARNING: Ganglion output is all zeros!`

**Possible causes**:
1. Ganglion not connected to bipolar layer
2. Ganglion's `update()` not reading bipolar input
3. Spiking threshold too high

**Debug steps**:
- Look for `[INIT] ✓ Ganglion layer connected to bipolar layer`
- Check ganglion update method reads from bipolar
- Verify spike generation logic

### Issue 4: Data Lost Between Server and Frontend
**Symptoms**:
- Console shows activity (non-zero ranges)
- But frontend displays black squares

**Possible causes**:
1. Binary encoding/decoding mismatch
2. Data not being sent in response
3. Frontend not receiving/decoding properly

**Debug steps**:
- Check `[API]` logs show non-zero data being sent
- Try disabling binary encoding: add `?binary=false` to API URL
- Check browser console for JavaScript errors
- Verify frontend is requesting correct endpoint

## How to Use This Debugging System

1. **Start the server** with terminal visible:
   ```bash
   python temporal_server.py
   ```

2. **Watch initialization logs** - should see:
   - Layer creation confirmations
   - Test frame processing
   - Initial activity verification

3. **Watch update loop logs** - every ~1 second:
   - Input frame info
   - Activity at each layer
   - Any WARNING messages

4. **Watch API logs** - when frontend requests state:
   - What data is being sent
   - Ranges of activity maps

5. **Compare logs** to identify break point:
   - If input frames are non-zero but retina is zero → retina issue
   - If retina is non-zero but bipolar is zero → bipolar issue
   - If bipolar is non-zero but ganglion is zero → ganglion issue
   - If all are non-zero but frontend shows black → transmission issue

## Quick Test Commands

### Test with simulated input (no webcam needed):
```bash
curl -X POST http://localhost:5001/api/initialize \
  -H "Content-Type: application/json" \
  -d '{"grid_size": 64, "use_real_webcam": false, "fps": 30}'
```

### Get current state:
```bash
curl http://localhost:5001/api/state/current
```

### Get current state (no binary encoding):
```bash
curl http://localhost:5001/api/state/current?binary=false
```

## Expected Normal Output

### At initialization:
```
[INIT] Creating FovealRetina with grid_size=64, fovea_radius=0.15
[INIT] ✓ Retina created: 22528 photoreceptors
[INIT] Creating BipolarLayer...
[INIT] ✓ Bipolar layer created: 3712 cells
[INIT] ✓ Bipolar layer connected to retina
[INIT] Creating GanglionLayer...
[INIT] ✓ Ganglion layer created: 928 cells
[INIT] ✓ Ganglion layer connected to bipolar layer
[INIT] Test frames: left=(64, 64) range=[0.235, 0.867]
[INIT] Test retina output: range=[0.123, 0.745], mean=0.342
[INIT] ✓ Retina producing activity
[INIT] Test bipolar output: range=[0.045, 0.534], mean=0.187
[INIT] ✓ Bipolar layer producing activity
[INIT] Test ganglion output: range=[0.012, 0.387], mean=0.098
[INIT] ✓ Ganglion layer producing activity
```

### During operation (every ~1 second):
```
[UPDATE] Frame 30: Left shape=(64, 64), range=[0.123, 0.945]
[UPDATE] Retina activity: range=[0.087, 0.823], mean=0.387
[UPDATE] Bipolar ON activity: range=[0.034, 0.612], mean=0.201
[UPDATE] Ganglion P activity: range=[0.015, 0.445], mean=0.112
```

### On API request:
```
[API] get_current_state called, binary=true
[API] Retina rods LEFT: shape=(64, 64), range=[0.087, 0.823], mean=0.387
[API] Bipolar ON LEFT: shape=(64, 64), range=[0.034, 0.612], mean=0.201
[API] Ganglion P LEFT: shape=(64, 64), range=[0.015, 0.445], mean=0.112
```

## Next Steps Based on Findings

1. **If all layers show activity in logs but frontend is black**:
   - Issue is in data transmission or frontend rendering
   - Check browser console for errors
   - Try `?binary=false` to test without binary encoding
   - Check frontend visualization code

2. **If specific layer shows zeros**:
   - Read that layer's source code (optimized version)
   - Verify its `update()` and `get_activity_map()` methods
   - Check if it's reading from the previous layer correctly

3. **If input frames are all zeros**:
   - Issue is in `FovealInputSystem`
   - Check webcam initialization or simulated input generation

4. **If errors appear in console**:
   - Full stack traces are printed
   - Look for import errors or method call failures

## Temporary Disabling of Lateral Cells

**Note**: Lateral processing layer (horizontal and amacrine cells) is currently disabled (line 213, 261-264). The optimized bipolar and ganglion layers are designed to work without lateral modulation for performance.

This should **not** cause the black square issue, as the lateral layer only modulates existing activity - it doesn't create it.

