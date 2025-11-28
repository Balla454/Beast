/*
 * BeAST System Arduino Emulator - Multi-Mode Version
 * Memory-efficient simulator with 3 separate data modes
 * 
 * Commands:
 *   L1 - EEG Raw Data (time series)
 *   L2 - EEG Processed Band Powers (time series)
 *   L3 - Clinical Metrics, Zones, PPG, Temp, Impacts
 *   S  - Stop streaming
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

enum OperatingMode {
  MODE_IDLE = 0,
  MODE_EEG_RAW = 1,      // L1: Raw EEG time series
  MODE_EEG_BANDS = 2,    // L2: EEG band powers
  MODE_CLINICAL = 3      // L3: Clinical metrics + zones
};

OperatingMode currentMode = MODE_IDLE;

// Timing
unsigned long lastUpdateTime = 0;
const unsigned long updateInterval = 1000; // 1 Hz
unsigned long sessionStartTime = 0;

// State variables
float fatigueLevel = 0.0;
float stressLevel = 0.3;
float cognitiveLoad = 0.4;
float hydrationLevel = 0.85;

// Zone duration counters (seconds spent in each zone)
unsigned long greenZoneDuration = 0;
unsigned long yellowZoneDuration = 0;
unsigned long orangeZoneDuration = 0;
unsigned long redZoneDuration = 0;

// HR zone durations
unsigned long hrGreenDuration = 0;
unsigned long hrYellowDuration = 0;
unsigned long hrOrangeDuration = 0;
unsigned long hrRedDuration = 0;

// SpO2 zone durations
unsigned long spo2GreenDuration = 0;
unsigned long spo2YellowDuration = 0;
unsigned long spo2RedDuration = 0;

// Hydration zone durations
unsigned long hydrationGreenDuration = 0;
unsigned long hydrationYellowDuration = 0;
unsigned long hydrationOrangeDuration = 0;
unsigned long hydrationRedDuration = 0;

// Impact counters
int impactCountLow = 0;      // <2g
int impactCountMedium = 0;   // 2-5g
int impactCountHigh = 0;     // 5-10g
int impactCountCritical = 0; // >10g

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(115200);
  
  while(Serial.available() > 0) {
    Serial.read();
  }
  
  randomSeed(analogRead(A0));
  delay(2000);
  Serial.flush();
  
  Serial.println(F("{\"status\":\"BeAST Arduino Ready\",\"version\":\"3.0\",\"modes\":\"L1=EEG_Raw,L2=EEG_Bands,L3=Clinical\"}"));
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  if (Serial.available() > 0) {
    handleCommand();
  }
  
  unsigned long currentTime = millis();
  
  if (currentMode != MODE_IDLE && currentTime - lastUpdateTime >= updateInterval) {
    lastUpdateTime = currentTime;
    updateStates();
    
    switch (currentMode) {
      case MODE_EEG_RAW:
        sendEEGRawBatch();
        break;
      case MODE_EEG_BANDS:
        sendEEGBandsBatch();
        break;
      case MODE_CLINICAL:
        sendClinicalBatch();
        break;
      default:
        break;
    }
  }
}

// ============================================================================
// COMMAND HANDLING
// ============================================================================

void handleCommand() {
  // Read entire command string
  String command = "";
  
  // Wait a bit for complete command to arrive
  delay(10);
  
  // Read all available characters
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c != '\n' && c != '\r') {
      command += c;
    }
  }
  
  // Process command
  if (command.length() > 0) {
    command.toUpperCase(); // Make case-insensitive
    
    if (command == "L1") {
      startMode(MODE_EEG_RAW, F("EEG_Raw"));
    } else if (command == "L2") {
      startMode(MODE_EEG_BANDS, F("EEG_Bands"));
    } else if (command == "L3") {
      startMode(MODE_CLINICAL, F("Clinical_Full"));
    } else if (command == "L") {
      // Default to clinical if just 'L'
      startMode(MODE_CLINICAL, F("Clinical_Full"));
    } else if (command == "S") {
      stopStreaming();
    } else {
      // Unknown command - send error
      Serial.print(F("{\"error\":\"unknown_command\",\"received\":\""));
      Serial.print(command);
      Serial.println(F("\"}"));
    }
  }
}

void startMode(OperatingMode mode, const __FlashStringHelper* modeName) {
  currentMode = mode;
  sessionStartTime = millis();
  
  // Reset state
  fatigueLevel = 0.0;
  stressLevel = 0.3;
  cognitiveLoad = 0.4;
  hydrationLevel = 0.85;
  
  // Reset zone durations
  greenZoneDuration = 0;
  yellowZoneDuration = 0;
  orangeZoneDuration = 0;
  redZoneDuration = 0;
  
  // Reset HR zone durations
  hrGreenDuration = 0;
  hrYellowDuration = 0;
  hrOrangeDuration = 0;
  hrRedDuration = 0;
  
  // Reset SpO2 zone durations
  spo2GreenDuration = 0;
  spo2YellowDuration = 0;
  spo2RedDuration = 0;
  
  // Reset Hydration zone durations
  hydrationGreenDuration = 0;
  hydrationYellowDuration = 0;
  hydrationOrangeDuration = 0;
  hydrationRedDuration = 0;
  
  // Reset impact counts
  impactCountLow = 0;
  impactCountMedium = 0;
  impactCountHigh = 0;
  impactCountCritical = 0;
  
  Serial.print(F("{\"status\":\"live_started\",\"mode\":\""));
  Serial.print(modeName);
  Serial.print(F("\",\"session_id\":\"LIVE-"));
  Serial.print(millis());
  Serial.println(F("\"}"));
}

void stopStreaming() {
  currentMode = MODE_IDLE;
  Serial.println(F("{\"status\":\"stopped\"}"));
}

// ============================================================================
// STATE UPDATES
// ============================================================================

void updateStates() {
  unsigned long elapsed = (millis() - sessionStartTime) / 1000;
  
  // Gradual fatigue increase
  fatigueLevel = min(90.0, (float)elapsed * 0.5);
  
  // Oscillating stress
  stressLevel = 30.0 + 20.0 * sin((float)elapsed * 0.1);
  
  // Varying cognitive load
  cognitiveLoad = 40.0 + 30.0 * cos((float)elapsed * 0.15);
  
  // Hydration slowly decreases
  hydrationLevel = max(70.0, 95.0 - (float)elapsed * 0.1);
  
  // Calculate current values for zone tracking
  float alertness = max(10.0, 90.0 - fatigueLevel);
  float hr = 72.0 + stressLevel * 0.3;
  float spo2 = 96.0 + random(0, 4);
  
  // Update alertness zone durations
  if (alertness > 70) {
    greenZoneDuration++;
  } else if (alertness > 50) {
    yellowZoneDuration++;
  } else if (alertness > 30) {
    orangeZoneDuration++;
  } else {
    redZoneDuration++;
  }
  
  // Update HR zone durations
  if (hr < 100) {
    hrGreenDuration++;
  } else if (hr < 120) {
    hrYellowDuration++;
  } else if (hr < 150) {
    hrOrangeDuration++;
  } else {
    hrRedDuration++;
  }
  
  // Update SpO2 zone durations
  if (spo2 > 95) {
    spo2GreenDuration++;
  } else if (spo2 > 90) {
    spo2YellowDuration++;
  } else {
    spo2RedDuration++;
  }
  
  // Update Hydration zone durations
  if (hydrationLevel > 85) {
    hydrationGreenDuration++;
  } else if (hydrationLevel > 75) {
    hydrationYellowDuration++;
  } else if (hydrationLevel > 65) {
    hydrationOrangeDuration++;
  } else {
    hydrationRedDuration++;
  }
  
  // Simulate random impacts (low probability)
  if (random(100) < 2) { // 2% chance per second
    int impactLevel = random(100);
    if (impactLevel < 70) {
      impactCountLow++;
    } else if (impactLevel < 90) {
      impactCountMedium++;
    } else if (impactLevel < 98) {
      impactCountHigh++;
    } else {
      impactCountCritical++;
    }
  }
}

// ============================================================================
// MODE 1: EEG RAW DATA (Time Series) - 12 CHANNELS
// ============================================================================

void sendEEGRawBatch() {
  Serial.print(F("{\"type\":\"data_batch\",\"batch\":["));
  
  // 12 channels for bilateral dual-ear setup
  const char* channels[] = {
    "L1", "L2", "L3", "L4", "L5", "L6",  // Left ear
    "R1", "R2", "R3", "R4", "R5", "R6"   // Right ear
  };
  
  for (int ch = 0; ch < 12; ch++) {
    Serial.print(F("{\"type\":\"eeg_raw\",\"channel\":\""));
    Serial.print(channels[ch]);
    Serial.print(F("\",\"samples\":["));
    
    // Generate 10 synthetic samples per channel
    for (int i = 0; i < 10; i++) {
      float baseSignal = sin((millis() + i * 100 + ch * 50) * 0.01) * 50.0;
      float noise = random(-10, 10);
      float sample = baseSignal + noise;
      
      Serial.print(sample, 1);
      if (i < 9) Serial.print(F(","));
    }
    
    Serial.print(F("]}"));
    if (ch < 11) Serial.print(F(","));
  }
  
  Serial.println(F("]}"));
}

// ============================================================================
// MODE 2: EEG BAND POWERS (5 bands reporting values over time)
// ============================================================================

void sendEEGBandsBatch() {
  Serial.print(F("{\"type\":\"data_batch\",\"batch\":["));
  
  // Generate aggregate band powers across all channels
  // These represent the overall brain state at this moment
  float delta = 15.0 + fatigueLevel * 0.2 + random(-3, 3);
  float theta = 20.0 + fatigueLevel * 0.3 + random(-3, 3);
  float alpha = 25.0 - cognitiveLoad * 0.2 + random(-3, 3);
  float beta = 30.0 + cognitiveLoad * 0.3 + random(-3, 3);
  float gamma = 10.0 + stressLevel * 0.2 + random(-2, 2);
  
  Serial.print(F("{\"type\":\"eeg_bands\","));
  Serial.print(F("\"delta\":"));
  Serial.print(delta, 1);
  Serial.print(F(",\"theta\":"));
  Serial.print(theta, 1);
  Serial.print(F(",\"alpha\":"));
  Serial.print(alpha, 1);
  Serial.print(F(",\"beta\":"));
  Serial.print(beta, 1);
  Serial.print(F(",\"gamma\":"));
  Serial.print(gamma, 1);
  Serial.print(F("}"));
  
  Serial.println(F("]}"));
}

// ============================================================================
// MODE 3: CLINICAL METRICS + ZONES + PPG + TEMP + IMPACTS
// ============================================================================

void sendClinicalBatch() {
  float alertness = max(10.0, 90.0 - fatigueLevel);
  float hr = 72.0 + stressLevel * 0.3 + random(-5, 5);
  float spo2 = 96.0 + random(0, 4);
  float coreTemp = 37.0 + stressLevel * 0.01 + random(-10, 10) / 100.0;
  float ambientTemp = 22.0 + random(-20, 30) / 10.0;
  
  // Determine alert level
  const char* alertLevel;
  if (alertness > 70 && hr < 100) {
    alertLevel = "GREEN";
  } else if (alertness > 50 && hr < 120) {
    alertLevel = "YELLOW";
  } else if (alertness > 30) {
    alertLevel = "ORANGE";
  } else {
    alertLevel = "RED";
  }
  
  Serial.print(F("{\"type\":\"data_batch\",\"batch\":["));
  
  // Clinical Metrics with Zone Durations
  Serial.print(F("{\"type\":\"clinical_metrics\","));
  Serial.print(F("\"alertness\":"));
  Serial.print(alertness, 1);
  Serial.print(F(",\"cognitive_load\":"));
  Serial.print(cognitiveLoad, 1);
  Serial.print(F(",\"fatigue\":"));
  Serial.print(fatigueLevel, 1);
  Serial.print(F(",\"alert_level\":\""));
  Serial.print(alertLevel);
  Serial.print(F("\",\"zone_durations\":{"));
  Serial.print(F("\"green\":"));
  Serial.print(greenZoneDuration);
  Serial.print(F(",\"yellow\":"));
  Serial.print(yellowZoneDuration);
  Serial.print(F(",\"orange\":"));
  Serial.print(orangeZoneDuration);
  Serial.print(F(",\"red\":"));
  Serial.print(redZoneDuration);
  Serial.print(F("}},"));
  
  // Hydration with Zone Durations
  const char* hydrationZone;
  if (hydrationLevel > 85) {
    hydrationZone = "GREEN";
  } else if (hydrationLevel > 75) {
    hydrationZone = "YELLOW";
  } else if (hydrationLevel > 65) {
    hydrationZone = "ORANGE";
  } else {
    hydrationZone = "RED";
  }
  
  Serial.print(F("{\"type\":\"hydration\",\"value\":"));
  Serial.print(hydrationLevel, 1);
  Serial.print(F(",\"zone\":\""));
  Serial.print(hydrationZone);
  Serial.print(F("\",\"zone_durations\":{"));
  Serial.print(F("\"green\":"));
  Serial.print(hydrationGreenDuration);
  Serial.print(F(",\"yellow\":"));
  Serial.print(hydrationYellowDuration);
  Serial.print(F(",\"orange\":"));
  Serial.print(hydrationOrangeDuration);
  Serial.print(F(",\"red\":"));
  Serial.print(hydrationRedDuration);
  Serial.print(F("}},"));
  
  // PPG with Zone Durations
  const char* hrZone;
  if (hr < 100) {
    hrZone = "GREEN";
  } else if (hr < 120) {
    hrZone = "YELLOW";
  } else if (hr < 150) {
    hrZone = "ORANGE";
  } else {
    hrZone = "RED";
  }
  
  const char* spo2Zone;
  if (spo2 > 95) {
    spo2Zone = "GREEN";
  } else if (spo2 > 90) {
    spo2Zone = "YELLOW";
  } else {
    spo2Zone = "RED";
  }
  
  Serial.print(F("{\"type\":\"ppg\",\"heart_rate\":"));
  Serial.print(hr, 1);
  Serial.print(F(",\"spo2\":"));
  Serial.print(spo2, 1);
  Serial.print(F(",\"hr_zone\":\""));
  Serial.print(hrZone);
  Serial.print(F("\",\"spo2_zone\":\""));
  Serial.print(spo2Zone);
  Serial.print(F("\",\"hr_zone_durations\":{"));
  Serial.print(F("\"green\":"));
  Serial.print(hrGreenDuration);
  Serial.print(F(",\"yellow\":"));
  Serial.print(hrYellowDuration);
  Serial.print(F(",\"orange\":"));
  Serial.print(hrOrangeDuration);
  Serial.print(F(",\"red\":"));
  Serial.print(hrRedDuration);
  Serial.print(F("},\"spo2_zone_durations\":{"));
  Serial.print(F("\"green\":"));
  Serial.print(spo2GreenDuration);
  Serial.print(F(",\"yellow\":"));
  Serial.print(spo2YellowDuration);
  Serial.print(F(",\"red\":"));
  Serial.print(spo2RedDuration);
  Serial.print(F("}},"));
  
  // Core Temperature with Zone
  const char* tempZone;
  if (coreTemp < 37.5) {
    tempZone = "GREEN";
  } else if (coreTemp < 38.5) {
    tempZone = "YELLOW";
  } else if (coreTemp < 39.5) {
    tempZone = "ORANGE";
  } else {
    tempZone = "RED";
  }
  
  Serial.print(F("{\"type\":\"core_temp\",\"temperature_c\":"));
  Serial.print(coreTemp, 2);
  Serial.print(F(",\"zone\":\""));
  Serial.print(tempZone);
  Serial.print(F("\"},"));
  
  // Ambient Temperature
  Serial.print(F("{\"type\":\"ambient_temp\",\"temperature_c\":"));
  Serial.print(ambientTemp, 1);
  Serial.print(F("},"));
  
  // Impact Data with Zone Counts
  // Simulate occasional impact
  float currentImpact = 0.0;
  if (random(100) < 5) { // 5% chance of showing recent impact
    currentImpact = random(1, 100) / 10.0; // 0.1 to 10.0g
  }
  
  Serial.print(F("{\"type\":\"impact\",\"current_magnitude_g\":"));
  Serial.print(currentImpact, 1);
  Serial.print(F(",\"zone_counts\":{"));
  Serial.print(F("\"low\":"));
  Serial.print(impactCountLow);
  Serial.print(F(",\"medium\":"));
  Serial.print(impactCountMedium);
  Serial.print(F(",\"high\":"));
  Serial.print(impactCountHigh);
  Serial.print(F(",\"critical\":"));
  Serial.print(impactCountCritical);
  Serial.print(F("}}"));
  
  Serial.println(F("]}"));
}
