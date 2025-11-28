# BeAST Data Format Specification

## 📋 Overview

This document defines the exact data formats used in the BeAST Live Data Monitor system. All data flows from Arduino → Python → Dashboard as JSON.

**Protocol:** Serial (Arduino to Python) + WebSocket (Python to Dashboard)
**Format:** JSON strings, one per line
**Encoding:** UTF-8

---

## 🎯 Command Protocol (Dashboard → Arduino)

### Command Format

**From Dashboard to Python:**
```json
{
  "command": "L1"
}
```

**Valid Commands:**
- `"L1"` - Switch to EEG Raw mode
- `"L2"` - Switch to EEG Bands mode
- `"L3"` - Switch to Clinical Full mode
- `"S"` - Stop streaming

**From Python to Arduino:**
```
L1\n
```
(Simple text command followed by newline)

---

### Command Acknowledgment

**From Arduino to Python:**
```json
{
  "status": "live_started",
  "mode": "EEG_Raw",
  "session_id": "LIVE-1234567890"
}
```

**From Python to Dashboard:**
```json
{
  "type": "command_ack",
  "command": "L1",
  "status": "queued"
}
```

Then:
```json
{
  "type": "command_sent",
  "command": "L1",
  "timestamp": "2025-11-17T19:30:00.000Z"
}
```

---

## 📊 Data Batch Protocol

All data from Arduino is wrapped in a batch structure:

```json
{
  "type": "data_batch",
  "batch": [
    { ... packet 1 ... },
    { ... packet 2 ... },
    { ... packet N ... }
  ]
}
```

When forwarded to dashboard, Python adds metadata:

```json
{
  "type": "data_batch",
  "batch": [ ... ],
  "timestamp": "2025-11-17T19:30:01.000Z",
  "stats": {
    "packets_received": 120,
    "packets_broadcast": 120,
    "errors": 0
  }
}
```

---

## 📡 Mode L1: EEG Raw Data

### Packet Structure

```json
{
  "type": "eeg_raw",
  "channel": "L1",
  "samples": [12.3, 14.5, 11.2, 15.8, 13.1, 16.9, 12.7, 14.2, 15.5, 13.8]
}
```

### Field Definitions

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `type` | string | Packet identifier | `"eeg_raw"` |
| `channel` | string | EEG channel name | `"L1"`, `"L2"`, `"L3"`, `"L4"`, `"L5"`, `"L6"`, `"R1"`, `"R2"`, `"R3"`, `"R4"`, `"R5"`, `"R6"` |
| `samples` | array[number] | Raw voltage samples | Array of 10 numbers, range: -100 to +100 µV |

### Complete L1 Batch Example

```json
{
  "type": "data_batch",
  "batch": [
    {
      "type": "eeg_raw",
      "channel": "L1",
      "samples": [12.3, 14.5, 11.2, 15.8, 13.1, 16.9, 12.7, 14.2, 15.5, 13.8]
    },
    {
      "type": "eeg_raw",
      "channel": "L2",
      "samples": [8.2, 9.1, 7.5, 10.3, 8.9, 11.2, 9.5, 8.7, 10.1, 9.3]
    },
    {
      "type": "eeg_raw",
      "channel": "L3",
      "samples": [15.6, 17.2, 14.8, 18.1, 16.3, 19.4, 15.9, 17.5, 18.3, 16.7]
    },
    {
      "type": "eeg_raw",
      "channel": "L4",
      "samples": [11.4, 12.8, 10.9, 13.5, 11.7, 14.2, 12.1, 11.9, 13.1, 12.5]
    },
    {
      "type": "eeg_raw",
      "channel": "L5",
      "samples": [9.8, 10.5, 9.2, 11.1, 10.3, 12.0, 9.9, 10.7, 11.3, 10.6]
    },
    {
      "type": "eeg_raw",
      "channel": "L6",
      "samples": [14.2, 15.8, 13.5, 16.4, 14.9, 17.1, 14.6, 15.3, 16.0, 15.1]
    },
    {
      "type": "eeg_raw",
      "channel": "R1",
      "samples": [13.5, 15.1, 12.8, 16.0, 14.2, 17.3, 13.9, 14.8, 15.7, 14.5]
    },
    {
      "type": "eeg_raw",
      "channel": "R2",
      "samples": [10.1, 11.3, 9.7, 12.0, 10.8, 12.9, 10.4, 11.0, 11.8, 11.2]
    },
    {
      "type": "eeg_raw",
      "channel": "R3",
      "samples": [16.8, 18.4, 15.9, 19.2, 17.1, 20.5, 16.5, 18.0, 19.0, 17.8]
    },
    {
      "type": "eeg_raw",
      "channel": "R4",
      "samples": [12.7, 14.1, 11.9, 14.8, 13.0, 15.5, 12.4, 13.6, 14.4, 13.3]
    },
    {
      "type": "eeg_raw",
      "channel": "R5",
      "samples": [11.0, 12.2, 10.5, 12.8, 11.5, 13.6, 11.2, 11.9, 12.5, 11.8]
    },
    {
      "type": "eeg_raw",
      "channel": "R6",
      "samples": [15.3, 17.0, 14.6, 17.8, 16.0, 18.9, 15.7, 16.5, 17.4, 16.3]
    }
  ]
}
```

### Update Rate
- **Frequency:** 1 Hz (one batch per second)
- **Total samples per batch:** 120 (12 channels × 10 samples)
- **Effective sampling rate:** 10 samples/second per channel

### Channel Layout
- **L1-L6:** Left ear sensors (6 channels)
- **R1-R6:** Right ear sensors (6 channels)
- **Total:** 12 channels (bilateral dual-ear configuration)

---

## 📈 Mode L2: EEG Band Powers

### Packet Structure

```json
{
  "type": "eeg_bands",
  "delta": 18.3,
  "theta": 22.1,
  "alpha": 20.5,
  "beta": 35.2,
  "gamma": 12.8
}
```

### Field Definitions

| Field | Type | Description | Valid Values | Frequency Range |
|-------|------|-------------|--------------|-----------------|
| `type` | string | Packet identifier | `"eeg_bands"` | - |
| `delta` | number | Delta band power | 0-50 µV² | 0.5-4 Hz |
| `theta` | number | Theta band power | 0-50 µV² | 4-8 Hz |
| `alpha` | number | Alpha band power | 0-50 µV² | 8-13 Hz |
| `beta` | number | Beta band power | 0-50 µV² | 13-30 Hz |
| `gamma` | number | Gamma band power | 0-50 µV² | 30+ Hz |

### Band Power Interpretation

| Band | State Indicators |
|------|------------------|
| **Delta** | High: Deep sleep, unconsciousness. Low: Alert, awake |
| **Theta** | High: Drowsiness, meditation, memory encoding. Low: Fully alert |
| **Alpha** | High: Relaxed, eyes closed, calm. Low: Eyes open, active thinking |
| **Beta** | High: Active thinking, focus, stress. Low: Relaxed state |
| **Gamma** | High: Complex processing, attention. Low: Reduced cognitive activity |

### Simulated Behavior

The simulator varies band powers based on internal state:
- **Delta/Theta** increase with `fatigueLevel`
- **Alpha** decreases with `cognitiveLoad`
- **Beta** increases with `cognitiveLoad`
- **Gamma** increases with `stressLevel`

### Complete L2 Batch Example

```json
{
  "type": "data_batch",
  "batch": [
    {
      "type": "eeg_bands",
      "delta": 18.3,
      "theta": 22.1,
      "alpha": 20.5,
      "beta": 35.2,
      "gamma": 12.8
    }
  ]
}
```

### Update Rate
- **Frequency:** 1 Hz (one batch per second)
- **Packets per batch:** 1
- **Values per packet:** 5 (one per band)

---

## 🎯 Mode L3: Clinical Full

Mode L3 contains 6 different packet types in each batch.

### Packet Type 1: Clinical Metrics

```json
{
  "type": "clinical_metrics",
  "alertness": 85.3,
  "cognitive_load": 45.2,
  "fatigue": 18.7,
  "alert_level": "GREEN",
  "zone_durations": {
    "green": 120,
    "yellow": 30,
    "orange": 5,
    "red": 0
  }
}
```

**Field Definitions:**

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `type` | string | Packet identifier | `"clinical_metrics"` |
| `alertness` | number | Alertness level | 0-100 (%) |
| `cognitive_load` | number | Mental workload | 0-100 (%) |
| `fatigue` | number | Fatigue level | 0-100 (%) |
| `alert_level` | string | Overall status | `"GREEN"`, `"YELLOW"`, `"ORANGE"`, `"RED"` |
| `zone_durations` | object | Time in each zone | Seconds (cumulative) |

**Zone Thresholds (Alertness-based):**
- **GREEN:** Alertness > 70% (Optimal performance)
- **YELLOW:** Alertness 50-70% (Reduced performance)
- **ORANGE:** Alertness 30-50% (Compromised performance)
- **RED:** Alertness < 30% (Non-mission-capable)

---

### Packet Type 2: Hydration

```json
{
  "type": "hydration",
  "value": 84.2,
  "zone": "GREEN",
  "zone_durations": {
    "green": 110,
    "yellow": 35,
    "orange": 10,
    "red": 0
  }
}
```

**Field Definitions:**

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `type` | string | Packet identifier | `"hydration"` |
| `value` | number | Hydration level | 0-100 (%) |
| `zone` | string | Current zone | `"GREEN"`, `"YELLOW"`, `"ORANGE"`, `"RED"` |
| `zone_durations` | object | Time in each zone | Seconds (cumulative) |

**Zone Thresholds:**
- **GREEN:** > 85% (Well hydrated)
- **YELLOW:** 75-85% (Mild dehydration)
- **ORANGE:** 65-75% (Moderate dehydration)
- **RED:** < 65% (Severe dehydration)

---

### Packet Type 3: PPG (Photoplethysmography)

```json
{
  "type": "ppg",
  "heart_rate": 78.5,
  "spo2": 97.2,
  "hr_zone": "GREEN",
  "spo2_zone": "GREEN",
  "hr_zone_durations": {
    "green": 130,
    "yellow": 20,
    "orange": 5,
    "red": 0
  },
  "spo2_zone_durations": {
    "green": 150,
    "yellow": 5,
    "red": 0
  }
}
```

**Field Definitions:**

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `type` | string | Packet identifier | `"ppg"` |
| `heart_rate` | number | Heart rate | 40-220 BPM |
| `spo2` | number | Blood oxygen saturation | 85-100 (%) |
| `hr_zone` | string | HR status | `"GREEN"`, `"YELLOW"`, `"ORANGE"`, `"RED"` |
| `spo2_zone` | string | SpO2 status | `"GREEN"`, `"YELLOW"`, `"RED"` |
| `hr_zone_durations` | object | HR time in zones | Seconds (cumulative) |
| `spo2_zone_durations` | object | SpO2 time in zones | Seconds (cumulative) |

**Heart Rate Zone Thresholds:**
- **GREEN:** < 100 BPM (Normal resting)
- **YELLOW:** 100-120 BPM (Elevated)
- **ORANGE:** 120-150 BPM (High)
- **RED:** > 150 BPM (Critical)

**SpO2 Zone Thresholds:**
- **GREEN:** > 95% (Normal)
- **YELLOW:** 90-95% (Low)
- **RED:** < 90% (Critical)

---

### Packet Type 4: Core Temperature

```json
{
  "type": "core_temp",
  "temperature_c": 37.15,
  "zone": "GREEN"
}
```

**Field Definitions:**

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `type` | string | Packet identifier | `"core_temp"` |
| `temperature_c` | number | Core body temperature | 35.0-42.0°C |
| `zone` | string | Temperature status | `"GREEN"`, `"YELLOW"`, `"ORANGE"`, `"RED"` |

**Zone Thresholds:**
- **GREEN:** 36.5-37.5°C (Normal)
- **YELLOW:** 37.5-38.5°C (Heat strain)
- **ORANGE:** 38.5-39.5°C (Heat exhaustion)
- **RED:** > 39.5°C (Heat stroke risk)

---

### Packet Type 5: Ambient Temperature

```json
{
  "type": "ambient_temp",
  "temperature_c": 22.3
}
```

**Field Definitions:**

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `type` | string | Packet identifier | `"ambient_temp"` |
| `temperature_c` | number | Environmental temperature | -40 to +85°C |

**No zones** - informational only

---

### Packet Type 6: Impact

```json
{
  "type": "impact",
  "current_magnitude_g": 0.0,
  "zone_counts": {
    "low": 5,
    "medium": 2,
    "high": 0,
    "critical": 0
  }
}
```

**Field Definitions:**

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `type` | string | Packet identifier | `"impact"` |
| `current_magnitude_g` | number | Recent impact force | 0-20+ g-force |
| `zone_counts` | object | Cumulative impact counts | Integer counts |

**Zone Thresholds (Counts, not durations):**
- **Low:** < 2g (Minor bumps)
- **Medium:** 2-5g (Moderate impacts)
- **High:** 5-10g (Severe impacts, TBI screening needed)
- **Critical:** > 10g (Life-threatening, immediate medical attention)

---

### Complete L3 Batch Example

```json
{
  "type": "data_batch",
  "batch": [
    {
      "type": "clinical_metrics",
      "alertness": 85.3,
      "cognitive_load": 45.2,
      "fatigue": 18.7,
      "alert_level": "GREEN",
      "zone_durations": {
        "green": 120,
        "yellow": 30,
        "orange": 5,
        "red": 0
      }
    },
    {
      "type": "hydration",
      "value": 84.2,
      "zone": "GREEN",
      "zone_durations": {
        "green": 110,
        "yellow": 35,
        "orange": 10,
        "red": 0
      }
    },
    {
      "type": "ppg",
      "heart_rate": 78.5,
      "spo2": 97.2,
      "hr_zone": "GREEN",
      "spo2_zone": "GREEN",
      "hr_zone_durations": {
        "green": 130,
        "yellow": 20,
        "orange": 5,
        "red": 0
      },
      "spo2_zone_durations": {
        "green": 150,
        "yellow": 5,
        "red": 0
      }
    },
    {
      "type": "core_temp",
      "temperature_c": 37.15,
      "zone": "GREEN"
    },
    {
      "type": "ambient_temp",
      "temperature_c": 22.3
    },
    {
      "type": "impact",
      "current_magnitude_g": 0.0,
      "zone_counts": {
        "low": 5,
        "medium": 2,
        "high": 0,
        "critical": 0
      }
    }
  ]
}
```

### Update Rate
- **Frequency:** 1 Hz (one batch per second)
- **Packets per batch:** 6
- **Total data points:** ~30 per second

---

## 🔄 Connection Lifecycle

### 1. Initial Connection

**Python → Dashboard:**
```json
{
  "type": "connected",
  "message": "Connected to BeAST live data stream",
  "timestamp": "2025-11-17T19:30:00.000Z"
}
```

### 2. Command Sent

**Dashboard → Python:**
```json
{
  "command": "L3"
}
```

**Python → Dashboard:**
```json
{
  "type": "command_ack",
  "command": "L3",
  "status": "queued"
}
```

### 3. Command Forwarded

**Python → Arduino:**
```
L3\n
```

**Arduino → Python:**
```json
{
  "status": "live_started",
  "mode": "Clinical_Full",
  "session_id": "LIVE-1234567890"
}
```

**Python → Dashboard:**
```json
{
  "type": "command_sent",
  "command": "L3",
  "timestamp": "2025-11-17T19:30:00.123Z"
}
```

### 4. Data Streaming

**Arduino → Python → Dashboard:**
```json
{
  "type": "data_batch",
  "batch": [...],
  "timestamp": "2025-11-17T19:30:01.456Z",
  "stats": {
    "packets_received": 6,
    "packets_broadcast": 6,
    "errors": 0
  }
}
```

### 5. Stop Command

**Dashboard → Python:**
```json
{
  "command": "S"
}
```

**Arduino → Python:**
```json
{
  "status": "stopped"
}
```

---

## ⚠️ Important Notes

### Case Sensitivity
- All packet `type` values are **lowercase** with underscores: `"eeg_raw"`, `"clinical_metrics"`
- All zone names are **UPPERCASE**: `"GREEN"`, `"YELLOW"`, `"ORANGE"`, `"RED"`
- All zone duration keys are **lowercase**: `green`, `yellow`, `orange`, `red`
- Commands are **case-insensitive** (Arduino converts to uppercase)

### Field Naming
- Use **snake_case** for field names: `temperature_c`, `heart_rate`, `zone_durations`
- Never camelCase: ~~`temperatureC`~~, ~~`heartRate`~~, ~~`zoneDurations`~~

### Data Types
- **Numeric values:** Always numbers, never strings (`78.5` not `"78.5"`)
- **Zone names:** Always strings (`"GREEN"` not just `GREEN`)
- **Arrays:** Always arrays, even for single values (`samples` is always 10-element array)

### Timestamps
- **ISO 8601 format:** `"2025-11-17T19:30:00.000Z"`
- **Always UTC timezone**
- Added by Python service, not Arduino

### Error Handling
- Invalid commands should be ignored by Arduino
- Malformed JSON should log error but not crash service
- Missing fields should use defaults where appropriate

---

## 🧪 Validation

### JSON Schema Validators

You can validate data against these formats using JSON Schema tools.

Example for `eeg_raw`:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["type", "channel", "samples"],
  "properties": {
    "type": {
      "type": "string",
      "const": "eeg_raw"
    },
    "channel": {
      "type": "string",
      "enum": ["L1", "L2", "L3", "L4", "L5", "L6", "R1", "R2", "R3", "R4", "R5", "R6"]
    },
    "samples": {
      "type": "array",
      "minItems": 10,
      "maxItems": 10,
      "items": {
        "type": "number",
        "minimum": -100,
        "maximum": 100
      }
    }
  }
}
```

---

**Version:** 1.0
**Last Updated:** November 2025
