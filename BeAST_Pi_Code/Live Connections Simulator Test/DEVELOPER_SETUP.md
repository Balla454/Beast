# BeAST Live Data Monitor - Developer Setup Guide

## 📋 Overview

This test environment allows you to simulate the BeAST warfighter monitoring system with an Arduino emulator feeding data through a Python WebSocket service to a web dashboard.

**System Flow:**
```
Arduino Simulator → Serial USB → Python Service → WebSocket → Web Dashboard
   (Hardware data)              (Data relay)                  (Visualization)
```

---

## 📦 What's Included

### 1. Arduino Simulator (`BeAST_MultiMode.ino`)
- Simulates 3 data modes: L1 (EEG Raw), L2 (EEG Bands), L3 (Clinical)
- Sends JSON data over serial at 115200 baud
- Responds to mode commands: L1, L2, L3, S (stop)

### 2. Python WebSocket Service (`beast_live_stream_only.py`)
- Reads serial data from Arduino
- Broadcasts to dashboard via WebSocket (port 8765)
- Forwards mode commands from dashboard to Arduino

### 3. Web Dashboard (Lovable Project)
- Real-time visualization of all data modes
- Mode control buttons
- Time series charts for EEG data
- Clinical metrics display

### 4. Test Control Panel (`beast_control_panel.html`)
- Standalone HTML for testing WebSocket commands
- Useful for debugging without the main dashboard

### 5. Documentation
- Data format specifications
- Multi-mode guide
- API documentation

---

## 🛠️ Prerequisites

### Hardware
- **Arduino Uno** (or compatible board)
- **USB cable**
- **Computer** (Mac, Windows, or Linux)

### Software
- **Arduino IDE** (v1.8.x or v2.x)
  - Download: https://www.arduino.cc/en/software
- **Python 3.8+**
  - Download: https://www.python.org/downloads/
- **Web Browser** (Chrome, Firefox, Safari, or Edge)
- **Code Editor** (VS Code recommended but not required)

### Python Packages
```bash
pip install pyserial websockets
```

---

## 📥 Installation Steps

### Step 1: Download All Files

You should have received:
```
BeAST_Developer_Package/
├── Arduino/
│   └── BeAST_MultiMode.ino
├── Python/
│   └── beast_live_stream_only.py
├── Web/
│   └── beast_control_panel.html
├── Documentation/
│   ├── DEVELOPER_SETUP.md (this file)
│   ├── DATA_FORMATS.md
│   ├── MULTIMODE_GUIDE.md
│   └── API_REFERENCE.md
└── Lovable/
    └── LOVABLE_EXPORT_INSTRUCTIONS.txt
```

### Step 2: Install Python Dependencies

```bash
# Navigate to Python folder
cd BeAST_Developer_Package/Python

# Install required packages
pip install pyserial websockets

# Verify installation
python3 -c "import serial; import websockets; print('✅ Dependencies installed')"
```

### Step 3: Upload Arduino Code

1. **Connect Arduino** to your computer via USB
2. **Open Arduino IDE**
3. **Open** `BeAST_MultiMode.ino`
4. **Select Board:**
   - Tools → Board → Arduino Uno
5. **Select Port:**
   - Tools → Port → (select your Arduino port)
   - Mac: `/dev/cu.usbmodem*`
   - Windows: `COM3`, `COM4`, etc.
   - Linux: `/dev/ttyACM0` or `/dev/ttyUSB0`
6. **Upload:**
   - Click Upload button (→)
   - Wait for "Done uploading"
7. **Test:**
   - Tools → Serial Monitor
   - Set baud rate to **115200**
   - You should see: `{"status":"BeAST Arduino Ready",...}`
   - Type `L3` and press Enter
   - You should see JSON data flowing
8. **Close Serial Monitor** (important for next step!)

### Step 4: Configure Python Service

1. **Find your Arduino port:**
   ```bash
   # Mac
   ls /dev/cu.usbmodem*
   
   # Windows (in Command Prompt)
   mode
   
   # Linux
   ls /dev/ttyACM* /dev/ttyUSB*
   ```

2. **Edit** `beast_live_stream_only.py` (line 26):
   ```python
   # Mac example
   SERIAL_PORT = '/dev/cu.usbmodem14101'
   
   # Windows example
   SERIAL_PORT = 'COM3'
   
   # Linux example
   SERIAL_PORT = '/dev/ttyACM0'
   ```

3. **Save** the file

### Step 5: Get Lovable Dashboard

**Option A: If you have the Lovable project link:**
1. Open the Lovable project URL shared with you
2. The dashboard should load immediately
3. Note: Lovable is a web-based IDE, no installation needed

**Option B: If you have exported code:**
1. Extract the Lovable export ZIP file
2. Install dependencies:
   ```bash
   cd lovable-export
   npm install
   ```
3. Start development server:
   ```bash
   npm run dev
   ```
4. Open browser to `http://localhost:5173` (or shown URL)

---

## 🚀 Running the System

### Complete Startup Sequence

**1. Start Arduino** (already done if uploaded in Step 3)
   - Arduino should be plugged in and powered
   - LED should be on

**2. Start Python Service**
   ```bash
   cd BeAST_Developer_Package/Python
   python3 beast_live_stream_only.py
   ```
   
   **Expected output:**
   ```
   ✅ WebSocket server started on ws://localhost:8765
   ✅ Connected to Arduino on /dev/cu.usbmodem14101
   📡 Waiting for dashboard to send mode command (L1/L2/L3)
   ============================================================
   BeAST Live Data Stream - Ready
   ============================================================
   Dashboard can send commands: L1, L2, L3, S
   ```

**3. Open Dashboard**
   - **Lovable:** Open project URL in browser
   - **Exported:** Navigate to `http://localhost:5173`
   
   **Expected behavior:**
   - Connection indicator shows: **"● CONNECTED"** (green)
   - Four mode buttons visible: L1, L2, L3, STOP
   - Display shows: "Select a mode to start streaming data"

**4. Test Mode Switching**
   
   Click each button and verify behavior:
   
   **L3 - Clinical Mode:**
   - Python shows: `📤 Sent command to Arduino: L3`
   - Dashboard displays: Gauges, metrics, zone durations
   - Data updates every second
   
   **L1 - EEG Raw Mode:**
   - Python shows: `📤 Sent command to Arduino: L1`
   - Dashboard displays: 12 scrolling line charts (L1-L6, R1-R6)
   - Charts scroll right-to-left in real-time
   
   **L2 - EEG Bands Mode:**
   - Python shows: `📤 Sent command to Arduino: L2`
   - Dashboard displays: 5 colored band charts (Delta, Theta, Alpha, Beta, Gamma)
   - Charts update every second
   
   **STOP:**
   - Python shows: `📤 Sent command to Arduino: S`
   - Dashboard displays: "Stream stopped - Click a mode to start"
   - Data flow stops

---

## 🧪 Testing & Validation

### Test Sequence 1: Basic Connectivity

**Test:** Verify complete data pipeline

1. Start Python service
2. Open dashboard
3. Open browser console (F12)
4. Click L3 button
5. **Verify:**
   - [ ] Python terminal shows: `📨 Received from dashboard: {'command': 'L3'}`
   - [ ] Python terminal shows: `📤 Sent command to Arduino: L3`
   - [ ] Python terminal shows: `📊 Stats: X packets received...`
   - [ ] Browser console shows: `📦 Received: {type: "data_batch",...}`
   - [ ] Dashboard displays clinical metrics

**Expected data rate:** ~8 packets/second in L3 mode

---

### Test Sequence 2: L1 Mode - EEG Raw Data

**Test:** Validate 12-channel raw EEG streaming

1. Click L1 button
2. **Verify:**
   - [ ] Dashboard shows 12 separate charts (3 columns × 4 rows)
   - [ ] Charts labeled: L1, L2, L3, L4, L5, L6, R1, R2, R3, R4, R5, R6
   - [ ] Left ear channels (L1-L6) in **blue**
   - [ ] Right ear channels (R1-R6) in **green**
   - [ ] Waveforms scroll **right to left**
   - [ ] Y-axis range: -100 to +100
   - [ ] Data flows smoothly without compression

**Expected data format (browser console):**
```javascript
{
  type: "data_batch",
  batch: [
    {type: "eeg_raw", channel: "L1", samples: [12.3, 14.5, 11.2, ...]},
    {type: "eeg_raw", channel: "L2", samples: [...]},
    // ... 12 total
  ]
}
```

**Expected data rate:** ~120 samples/second (12 channels × 10 samples/sec)

---

### Test Sequence 3: L2 Mode - EEG Band Powers

**Test:** Validate frequency band analysis

1. Click L2 button
2. **Verify:**
   - [ ] Dashboard shows 5 stacked charts
   - [ ] Charts labeled with frequency ranges:
     - Delta (0.5-4 Hz) - Blue
     - Theta (4-8 Hz) - Purple
     - Alpha (8-13 Hz) - Green
     - Beta (13-30 Hz) - Orange
     - Gamma (30+ Hz) - Red
   - [ ] Each chart shows single line updating over time
   - [ ] Band values change dynamically (not static)
   - [ ] Charts scroll right to left

**Expected data format (browser console):**
```javascript
{
  type: "data_batch",
  batch: [
    {
      type: "eeg_bands",
      delta: 18.3,
      theta: 22.1,
      alpha: 20.5,
      beta: 35.2,
      gamma: 12.8
    }
  ]
}
```

**Expected data rate:** 1 update/second (5 values per update)

---

### Test Sequence 4: L3 Mode - Clinical Full

**Test:** Validate comprehensive clinical monitoring

1. Click L3 button
2. **Verify all data types appear:**

**Clinical Metrics:**
- [ ] Alertness (0-100%)
- [ ] Cognitive Load (0-100%)
- [ ] Fatigue (0-100%)
- [ ] Alert Level (GREEN/YELLOW/ORANGE/RED banner)
- [ ] Zone durations (cumulative seconds in each zone)

**Hydration:**
- [ ] Value (0-100%)
- [ ] Current zone (GREEN/YELLOW/ORANGE/RED)
- [ ] Zone durations (seconds in each zone)

**PPG (Cardiovascular):**
- [ ] Heart Rate (BPM)
- [ ] HR zone (GREEN/YELLOW/ORANGE/RED)
- [ ] HR zone durations
- [ ] SpO2 (%)
- [ ] SpO2 zone (GREEN/YELLOW/RED)
- [ ] SpO2 zone durations

**Temperature:**
- [ ] Core temperature (°C)
- [ ] Core zone classification
- [ ] Ambient temperature (°C)

**Impact Monitoring:**
- [ ] Current magnitude (g-force)
- [ ] Zone counts (low/medium/high/critical)

**Expected data format (browser console):**
```javascript
{
  type: "data_batch",
  batch: [
    {type: "clinical_metrics", alertness: 85.3, ...},
    {type: "hydration", value: 84.2, zone_durations: {...}},
    {type: "ppg", heart_rate: 78.5, hr_zone_durations: {...}, spo2_zone_durations: {...}},
    {type: "core_temp", temperature_c: 37.15, zone: "GREEN"},
    {type: "ambient_temp", temperature_c: 22.3},
    {type: "impact", current_magnitude_g: 0.0, zone_counts: {...}}
  ]
}
```

**Expected data rate:** 6 packets/second

---

### Test Sequence 5: Mode Switching

**Test:** Verify clean transitions between modes

1. Start in L3 mode (wait 10 seconds)
2. Switch to L1 mode
3. **Verify:**
   - [ ] L3 data stops immediately
   - [ ] L1 charts appear and start scrolling
   - [ ] No mixed data (no L3 packets in L1 mode)
4. Switch to L2 mode
5. **Verify:**
   - [ ] L1 charts disappear
   - [ ] L2 charts appear
   - [ ] Band values start from fresh state
6. Switch back to L3
7. **Verify:**
   - [ ] Zone durations reset to zero (new session)
   - [ ] All metrics display correctly

---

### Test Sequence 6: Performance & Stability

**Test:** Long-duration stability

1. Click L1 button
2. **Let run for 5 minutes**
3. **Verify:**
   - [ ] Charts continue scrolling smoothly (no freezing)
   - [ ] Memory usage stable in browser (check Task Manager)
   - [ ] Python terminal shows steady packet rate
   - [ ] No error messages in console
4. Switch to L2 mode
5. **Let run for 5 minutes**
6. **Verify same stability**
7. Switch to L3 mode
8. **Let run for 5 minutes**
9. **Verify:**
   - [ ] Zone durations continuously increasing
   - [ ] Impact counts incrementing (occasionally)
   - [ ] No data corruption or freezing

---

## 🔍 Troubleshooting

### Issue: Python can't connect to Arduino

**Symptoms:**
```
❌ Failed to connect to Arduino: [Errno 2] could not open port /dev/cu.usbmodem14101
```

**Solutions:**
1. Check Arduino is plugged in (LED should be on)
2. Close Arduino Serial Monitor if open
3. Find correct port:
   ```bash
   # Mac
   ls /dev/cu.usbmodem*
   
   # Windows
   # Check Device Manager → Ports (COM & LPT)
   
   # Linux
   ls /dev/ttyACM* /dev/ttyUSB*
   ```
4. Update `SERIAL_PORT` in Python script
5. On Linux, may need permissions:
   ```bash
   sudo usermod -a -G dialout $USER
   # Then log out and back in
   ```

---

### Issue: Dashboard shows "DISCONNECTED"

**Symptoms:**
- Red connection indicator
- No mode buttons working

**Solutions:**
1. Check Python service is running (should see "WebSocket server started")
2. Check firewall isn't blocking port 8765
3. Open browser console (F12) and check for WebSocket errors
4. Try accessing from same machine as Python service
5. If Lovable project, ensure it's not trying to connect to wrong host

---

### Issue: Mode commands not switching Arduino

**Symptoms:**
- Click L1 button but still receiving L3 data
- Python shows "Sent command" but Arduino doesn't change

**Solutions:**
1. Check Python terminal for "Sent command to Arduino: L1"
2. Stop Python and test Arduino directly with Serial Monitor:
   - Open Serial Monitor (115200 baud)
   - Type `L1` and press Enter
   - Should see: `{"status":"live_started","mode":"EEG_Raw",...}`
   - Should see: `{"batch":[{"type":"eeg_raw",...}]}`
3. If Serial Monitor works but Python doesn't:
   - Check Python is using correct serial port
   - Restart both Python and Arduino
4. Make sure Arduino code is the latest version (with String-based command handling)

---

### Issue: Charts compressing/not scrolling

**Symptoms:**
- L1/L2 charts show all data compressed
- Doesn't scroll right-to-left

**Solutions:**
1. Verify Lovable code uses `.slice(-100)` for L1 data
2. Verify Lovable code uses `.slice(-60)` for L2 data
3. Check browser console for JavaScript errors
4. Make sure `isAnimationActive={false}` is set on Line components

---

### Issue: No data appearing in charts

**Symptoms:**
- Dashboard shows mode active but charts are empty
- Browser console shows data arriving

**Solutions:**
1. Check browser console for data format:
   ```javascript
   console.log('EEG Raw History:', eegRawHistory);
   console.log('EEG Bands History:', eegBandsHistory);
   ```
2. Verify state is updating (should see arrays with values)
3. Check chart data prop is correctly mapped
4. Clear browser cache and hard refresh (Ctrl+Shift+R)

---

## 📊 Data Format Reference

### Quick Reference Table

| Mode | Packet Type | Update Rate | Key Fields |
|------|-------------|-------------|------------|
| L1 | `eeg_raw` | 1 Hz (12 packets) | `channel`, `samples` (10 values) |
| L2 | `eeg_bands` | 1 Hz (1 packet) | `delta`, `theta`, `alpha`, `beta`, `gamma` |
| L3 | `clinical_metrics` | 1 Hz (1 packet) | `alertness`, `cognitive_load`, `fatigue`, `zone_durations` |
| L3 | `hydration` | 1 Hz (1 packet) | `value`, `zone`, `zone_durations` |
| L3 | `ppg` | 1 Hz (1 packet) | `heart_rate`, `spo2`, `hr_zone_durations`, `spo2_zone_durations` |
| L3 | `core_temp` | 1 Hz (1 packet) | `temperature_c`, `zone` |
| L3 | `ambient_temp` | 1 Hz (1 packet) | `temperature_c` |
| L3 | `impact` | 1 Hz (1 packet) | `current_magnitude_g`, `zone_counts` |

### Full Data Format Specifications

See `DATA_FORMATS.md` for complete JSON schemas and field descriptions.

---

## 📝 Notes for Hardware Integration

### When Replacing Arduino Simulator with Real Hardware

The real hardware developer should match these exact specifications:

**Serial Communication:**
- Baud rate: **115200**
- Format: **JSON strings**, one per line, ending with `\n`
- Encoding: **UTF-8**

**Command Protocol:**
- Receive commands: `L1`, `L2`, `L3`, `S`
- Commands end with `\n`
- Respond with: `{"status":"live_started","mode":"<mode_name>",...}`

**Data Protocol:**
- Send data in batches: `{"type":"data_batch","batch":[...]}`
- Each packet in batch has `type` field identifying packet type
- All field names must match exactly (case-sensitive)
- Zone names are uppercase: "GREEN", "YELLOW", "ORANGE", "RED"
- Zone durations are objects: `{green: 123, yellow: 45, orange: 5, red: 0}`

**Testing Real Hardware:**
1. Upload Arduino simulator code to verify data format
2. Compare Arduino simulator output to real hardware output
3. Use `beast_control_panel.html` to test commands
4. Watch Python terminal to see exact data received
5. Use browser console to verify dashboard parsing

---

## 🎯 Success Criteria

Your setup is working correctly when:

✅ **Python service connects to Arduino** without errors
✅ **Dashboard shows "CONNECTED"** status
✅ **L1 mode displays 12 scrolling EEG charts** (blue and green)
✅ **L2 mode displays 5 colored band power charts**
✅ **L3 mode displays all clinical metrics with zone durations**
✅ **Mode switching works instantly** (no mixed data)
✅ **Charts scroll smoothly** without compression
✅ **Zone durations increment** over time in L3 mode
✅ **System runs for 5+ minutes** without errors or crashes
✅ **Browser console shows correct data formats** for each mode
✅ **Python terminal shows steady packet rates**

---

## 📞 Support & Questions

If you encounter issues not covered in this guide:

1. **Check all files are latest version** (check version numbers in code comments)
2. **Review browser console** for JavaScript errors
3. **Review Python terminal** for connection/data errors
4. **Test Arduino independently** with Serial Monitor
5. **Document exact error messages** and steps to reproduce
6. **Check DATA_FORMATS.md** for data structure questions

---

## 📚 Additional Documentation

- **DATA_FORMATS.md** - Complete data format specifications with examples
- **MULTIMODE_GUIDE.md** - Detailed explanation of L1/L2/L3 modes
- **API_REFERENCE.md** - WebSocket API and command reference
- **ARDUINO_SPECS.md** - Hardware integration specifications

---

**Version:** 1.0
**Last Updated:** November 2025
**Maintained By:** BeAST Development Team
