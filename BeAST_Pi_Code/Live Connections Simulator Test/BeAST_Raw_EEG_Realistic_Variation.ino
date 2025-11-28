/*
 * BeAST System Data Emulator - Raw EEG with Realistic Channel Variation
 * 
 * Each EEG channel now has:
 * - Unique baseline offset
 * - Different frequency dominance (some more alpha, some more theta, etc.)
 * - Channel-specific noise characteristics
 * - Realistic amplitude variations based on electrode location
 * 
 * Data Output Schedule:
 * - RAW EEG SAMPLES: Every 0.25 seconds (4 Hz)
 * - PROCESSED METRICS: Every 1 second (1 Hz)
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

enum OperatingMode {
  MODE_IDLE = 0,
  MODE_LIVE = 1,
  MODE_PLAYBACK = 2
};

OperatingMode currentMode = MODE_IDLE;

// Timing for dual-rate streaming
unsigned long lastRawUpdateTime = 0;
const unsigned long rawUpdateInterval = 250;  // 0.25 seconds

unsigned long lastFullUpdateTime = 0;
const unsigned long fullUpdateInterval = 1000; // 1 second

unsigned long sessionStartTime = 0;
float sessionMinutes = 0.0;

// Raw EEG simulation parameters
const int RAW_SAMPLES_PER_PACKET = 62;
const float RAW_SAMPLE_RATE = 250.0;

// Physiological state variables
float fatigueLevel = 0.0;
float stressLevel = 0.3;
float cognitiveLoad = 0.4;
float thermalStress = 0.0;
bool tbiEvent = false;
float hydrationLevel = 0.85;

const float baselineHR = 72.0;
const float baselineTemp = 37.0;
const float baselineSpO2 = 98.0;

// Playback variables
int currentPlaybackSession = 0;
int playbackIndex = 0;
int playbackLength = 0;

// Per-channel characteristics for realistic variation
struct ChannelProfile {
  const char* name;
  float baselineOffset;      // DC offset in microvolts
  float deltaWeight;         // Relative strength of delta (0-2)
  float thetaWeight;         // Relative strength of theta
  float alphaWeight;         // Relative strength of alpha
  float betaWeight;          // Relative strength of beta
  float gammaWeight;         // Relative strength of gamma
  float noiseAmplitude;      // Amount of random noise
  float artifactProbability; // Probability of artifacts (0-1)
  float driftRate;           // Slow baseline drift
};

// Channel profiles based on anatomical location and typical EEG characteristics
ChannelProfile channelProfiles[8] = {
  // Left Ear Channels
  {"Fp1", -2.5, 1.2, 0.8, 1.5, 1.0, 0.7, 2.5, 0.008, 0.002},  // Frontal - more theta, eye blinks
  {"F7",   1.0, 0.9, 1.0, 1.3, 1.2, 0.9, 1.8, 0.005, 0.001},  // Frontal-temporal - balanced
  {"T3",   0.5, 0.8, 1.3, 1.0, 0.8, 0.6, 1.5, 0.003, 0.001},  // Temporal - more theta/alpha
  {"A1",  -1.5, 1.0, 0.9, 1.1, 0.9, 0.8, 1.2, 0.004, 0.001},  // Mastoid reference - quieter
  
  // Right Ear Channels (slightly different from left due to hemispheric asymmetry)
  {"Fp2", -2.0, 1.3, 0.7, 1.4, 1.1, 0.8, 2.8, 0.009, 0.002},  // Frontal - more artifacts
  {"F8",   1.5, 0.8, 0.9, 1.4, 1.3, 1.0, 2.0, 0.006, 0.001},  // Frontal-temporal - more beta
  {"T4",   0.8, 0.7, 1.4, 0.9, 0.9, 0.7, 1.6, 0.004, 0.001},  // Temporal - alpha dominant
  {"A2",  -1.8, 1.1, 0.8, 1.0, 0.8, 0.7, 1.3, 0.003, 0.001}   // Mastoid reference
};

// Dynamic per-channel state
float channelDrift[8] = {0, 0, 0, 0, 0, 0, 0, 0};
float channelPhase[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // Phase offset for waveforms
unsigned long rawSampleCounter = 0;

// ============================================================================
// STORED SESSION DATA (same as before)
// ============================================================================

struct SessionMetadata {
  int sessionId;
  char sessionType[16];
  char description[32];
  int durationMinutes;
  int dataPointCount;
};

const int NUM_STORED_SESSIONS = 4;
SessionMetadata storedSessions[NUM_STORED_SESSIONS] = {
  {1, "Training", "Low stress patrol", 30, 5},
  {2, "Mission", "High stress combat", 45, 6},
  {3, "Assessment", "Cognitive testing", 20, 4},
  {4, "Recovery", "Post-mission rest", 60, 6}
};

struct SessionDataPoint {
  float alertness;
  float cogLoad;
  float fatigue;
  float stress;
  float heartRate;
  float coreTemp;
  float missionReadiness;
  bool tbiFlag;
  char alertLevel[12];
};

SessionDataPoint session1Data[] = {
  {95.0, 30.0, 5.0, 25.0, 68.0, 36.8, 92.0, false, "normal"},
  {93.0, 32.0, 8.0, 27.0, 70.0, 36.9, 90.0, false, "normal"},
  {91.0, 35.0, 12.0, 28.0, 72.0, 37.0, 88.0, false, "normal"},
  {89.0, 38.0, 15.0, 30.0, 74.0, 37.1, 85.0, false, "normal"},
  {87.0, 40.0, 18.0, 32.0, 76.0, 37.2, 83.0, false, "normal"}
};

SessionDataPoint session2Data[] = {
  {85.0, 60.0, 20.0, 65.0, 95.0, 37.5, 75.0, false, "caution"},
  {82.0, 68.0, 25.0, 70.0, 105.0, 37.8, 70.0, false, "caution"},
  {78.0, 75.0, 30.0, 75.0, 115.0, 38.0, 65.0, false, "warning"},
  {45.0, 90.0, 55.0, 85.0, 135.0, 38.5, 35.0, true, "critical"},
  {50.0, 85.0, 52.0, 80.0, 130.0, 38.3, 40.0, false, "warning"},
  {55.0, 80.0, 48.0, 75.0, 120.0, 38.0, 50.0, false, "warning"}
};

SessionDataPoint session3Data[] = {
  {90.0, 75.0, 10.0, 40.0, 75.0, 37.0, 82.0, false, "caution"},
  {88.0, 80.0, 15.0, 45.0, 78.0, 37.1, 78.0, false, "caution"},
  {85.0, 85.0, 20.0, 50.0, 80.0, 37.2, 75.0, false, "caution"},
  {82.0, 88.0, 25.0, 52.0, 82.0, 37.3, 72.0, false, "warning"}
};

SessionDataPoint session4Data[] = {
  {65.0, 45.0, 60.0, 55.0, 85.0, 37.8, 55.0, false, "warning"},
  {70.0, 40.0, 52.0, 50.0, 80.0, 37.5, 62.0, false, "caution"},
  {75.0, 35.0, 45.0, 45.0, 76.0, 37.3, 70.0, false, "caution"},
  {80.0, 30.0, 38.0, 40.0, 72.0, 37.1, 78.0, false, "normal"},
  {85.0, 28.0, 30.0, 35.0, 70.0, 37.0, 83.0, false, "normal"},
  {88.0, 25.0, 25.0, 32.0, 68.0, 36.9, 87.0, false, "normal"}
};

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(A0));
  
  // Initialize random phase offsets for each channel
  for (int i = 0; i < 8; i++) {
    channelPhase[i] = random(0, 628) / 100.0;  // Random phase 0-2π
  }
  
  delay(2000);
  
  Serial.println("{\"status\":\"BeAST Full System Emulator Ready\",\"version\":\"3.2\",\"mode\":\"idle\",\"features\":[\"raw_eeg_realistic_variation\",\"dual_rate_streaming\"]}");
  Serial.println("{\"info\":\"Commands: L=Live, S=Stop, P[n]=Playback, A=Sessions, R=Reset, ?=Status\"}");
  Serial.println("{\"info\":\"Raw EEG: 4 Hz with realistic per-channel variation\"}");
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  if (Serial.available() > 0) {
    handleCommand();
  }
  
  unsigned long currentTime = millis();
  
  if (currentMode == MODE_LIVE || currentMode == MODE_PLAYBACK) {
    sessionMinutes = (currentTime - sessionStartTime) / 60000.0;
    
    // Raw EEG updates at 4 Hz for both live and playback
    if (currentTime - lastRawUpdateTime >= rawUpdateInterval) {
      lastRawUpdateTime = currentTime;
      sendRawEEGPacket();
    }
    
    // Full metrics updates
    if (currentMode == MODE_LIVE) {
      // Live mode: update every 1 second
      if (currentTime - lastFullUpdateTime >= fullUpdateInterval) {
        lastFullUpdateTime = currentTime;
        updatePhysiologicalStates();
        sendFullDataPacket();
      }
    } else if (currentMode == MODE_PLAYBACK) {
      // Playback mode: calculate interval based on session duration
      // Distribute data points evenly across session duration
      int sessionDuration = storedSessions[currentPlaybackSession - 1].durationMinutes;
      unsigned long playbackInterval = (sessionDuration * 60000UL) / playbackLength;
      
      // Minimum 5 seconds between data points for visibility
      playbackInterval = max(playbackInterval, 5000UL);
      
      if (currentTime - lastFullUpdateTime >= playbackInterval) {
        lastFullUpdateTime = currentTime;
        sendPlaybackDataPacket();
      }
    }
  }
}

// ============================================================================
// COMMAND HANDLING (same as before)
// ============================================================================

void handleCommand() {
  char cmd = Serial.read();
  
  delay(10);
  String args = "";
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c != '\n' && c != '\r') {
      args += c;
    }
  }
  
  switch (cmd) {
    case 'L':
    case 'l':
      startLiveMode();
      break;
    case 'S':
    case 's':
      stopStreaming();
      break;
    case 'P':
    case 'p':
      if (args.length() > 0) {
        startPlaybackMode(args.toInt());
      } else {
        Serial.println("{\"error\":\"Playback requires session ID (e.g., P1)\"}");
      }
      break;
    case 'A':
    case 'a':
      listSessions();
      break;
    case 'R':
    case 'r':
      resetArduino();
      break;
    case '?':
      sendStatus();
      break;
    default:
      Serial.print("{\"error\":\"Unknown command: '");
      Serial.print(cmd);
      Serial.println("'\"}");
      break;
  }
}

void startLiveMode() {
  currentMode = MODE_LIVE;
  sessionStartTime = millis();
  lastRawUpdateTime = millis();
  lastFullUpdateTime = millis();
  rawSampleCounter = 0;
  
  fatigueLevel = 0.0;
  stressLevel = 0.3;
  cognitiveLoad = 0.4;
  thermalStress = 0.0;
  tbiEvent = false;
  hydrationLevel = 0.85;
  
  // Reset channel drifts
  for (int i = 0; i < 8; i++) {
    channelDrift[i] = 0.0;
  }
  
  Serial.println("{\"status\":\"Live streaming started\",\"mode\":\"live\",\"raw_eeg_rate\":\"4 Hz\",\"metrics_rate\":\"1 Hz\",\"channel_variation\":\"enabled\"}");
}

void stopStreaming() {
  OperatingMode previousMode = currentMode;
  currentMode = MODE_IDLE;
  
  Serial.print("{\"status\":\"Streaming stopped\",\"previous_mode\":\"");
  Serial.print(previousMode == MODE_LIVE ? "live" : "playback");
  Serial.println("\",\"mode\":\"idle\"}");
}

void startPlaybackMode(int sessionId) {
  if (sessionId < 1 || sessionId > NUM_STORED_SESSIONS) {
    Serial.print("{\"error\":\"Invalid session ID: ");
    Serial.print(sessionId);
    Serial.println("\"}");
    return;
  }
  
  currentMode = MODE_PLAYBACK;
  currentPlaybackSession = sessionId;
  playbackIndex = 0;
  sessionStartTime = millis();
  lastRawUpdateTime = millis();
  lastFullUpdateTime = millis();
  
  switch (sessionId) {
    case 1: playbackLength = sizeof(session1Data) / sizeof(SessionDataPoint); break;
    case 2: playbackLength = sizeof(session2Data) / sizeof(SessionDataPoint); break;
    case 3: playbackLength = sizeof(session3Data) / sizeof(SessionDataPoint); break;
    case 4: playbackLength = sizeof(session4Data) / sizeof(SessionDataPoint); break;
  }
  
  Serial.print("{\"status\":\"Playback started\",\"mode\":\"playback\",\"session_id\":");
  Serial.print(sessionId);
  Serial.print(",\"description\":\"");
  Serial.print(storedSessions[sessionId-1].description);
  Serial.print("\",\"duration_minutes\":");
  Serial.print(storedSessions[sessionId-1].durationMinutes);
  Serial.print(",\"data_points\":");
  Serial.print(playbackLength);
  Serial.print(",\"interval_seconds\":");
  Serial.print((storedSessions[sessionId-1].durationMinutes * 60) / playbackLength);
  Serial.println("}");
}

void listSessions() {
  Serial.println("{\"status\":\"Available sessions\",\"sessions\":[");
  
  for (int i = 0; i < NUM_STORED_SESSIONS; i++) {
    Serial.print("  {\"id\":");
    Serial.print(storedSessions[i].sessionId);
    Serial.print(",\"type\":\"");
    Serial.print(storedSessions[i].sessionType);
    Serial.print("\",\"description\":\"");
    Serial.print(storedSessions[i].description);
    Serial.print("\",\"duration_min\":");
    Serial.print(storedSessions[i].durationMinutes);
    Serial.print("}");
    
    if (i < NUM_STORED_SESSIONS - 1) {
      Serial.println(",");
    } else {
      Serial.println();
    }
  }
  
  Serial.println("]}");
}

void resetArduino() {
  currentMode = MODE_IDLE;
  sessionStartTime = 0;
  playbackIndex = 0;
  rawSampleCounter = 0;
  
  fatigueLevel = 0.0;
  stressLevel = 0.3;
  cognitiveLoad = 0.4;
  thermalStress = 0.0;
  tbiEvent = false;
  hydrationLevel = 0.85;
  
  for (int i = 0; i < 8; i++) {
    channelDrift[i] = 0.0;
  }
  
  Serial.println("{\"status\":\"Arduino reset complete\",\"mode\":\"idle\"}");
}

void sendStatus() {
  Serial.print("{\"status\":\"Status report\",\"mode\":\"");
  
  switch (currentMode) {
    case MODE_IDLE:
      Serial.print("idle");
      break;
    case MODE_LIVE:
      Serial.print("live");
      break;
    case MODE_PLAYBACK:
      Serial.print("playback");
      Serial.print("\",\"playback_session\":");
      Serial.print(currentPlaybackSession);
      Serial.print(",\"playback_progress\":");
      Serial.print(playbackIndex);
      Serial.print("/");
      Serial.print(playbackLength);
      break;
  }
  
  Serial.print("\",\"session_time_min\":");
  Serial.print(sessionMinutes, 2);
  Serial.print(",\"uptime_ms\":");
  Serial.print(millis());
  Serial.print(",\"raw_samples_sent\":");
  Serial.print(rawSampleCounter);
  Serial.println("}");
}

// ============================================================================
// RAW EEG DATA GENERATION WITH REALISTIC CHANNEL VARIATION
// ============================================================================

void sendRawEEGPacket() {
  Serial.print("{\"type\":\"raw_eeg\",\"timestamp\":");
  Serial.print(millis());
  Serial.print(",\"sample_rate\":");
  Serial.print(RAW_SAMPLE_RATE, 1);
  Serial.print(",\"samples_per_packet\":");
  Serial.print(RAW_SAMPLES_PER_PACKET);
  Serial.print(",\"channels\":{");
  
  // Calculate base rhythms from current state
  float deltaPower = 10.0 + fatigueLevel * 30.0;
  float thetaPower = 15.0 + fatigueLevel * 25.0 + cognitiveLoad * 10.0;
  float alphaPower = 25.0 - fatigueLevel * 15.0 - cognitiveLoad * 10.0;
  float betaPower = 20.0 + cognitiveLoad * 30.0 + stressLevel * 15.0;
  float gammaPower = 8.0 + cognitiveLoad * 12.0;
  
  if (tbiEvent) {
    thetaPower *= 1.5;
    deltaPower *= 1.3;
    gammaPower *= 0.6;
  }
  
  // Generate samples for each channel with unique characteristics
  for (int ch = 0; ch < 8; ch++) {
    Serial.print("\"");
    Serial.print(channelProfiles[ch].name);
    Serial.print("\":[");
    
    // Update slow drift for this channel
    channelDrift[ch] += channelProfiles[ch].driftRate * (random(-100, 100) / 100.0);
    channelDrift[ch] = constrain(channelDrift[ch], -5.0, 5.0);
    
    for (int i = 0; i < RAW_SAMPLES_PER_PACKET; i++) {
      float sample = generateRawEEGSample(
        ch, 
        i, 
        deltaPower, 
        thetaPower, 
        alphaPower, 
        betaPower, 
        gammaPower
      );
      
      Serial.print(sample, 2);
      
      if (i < RAW_SAMPLES_PER_PACKET - 1) {
        Serial.print(",");
      }
      
      rawSampleCounter++;
    }
    
    Serial.print("]");
    
    if (ch < 7) {
      Serial.print(",");
    }
  }
  
  Serial.println("}}");
}

float generateRawEEGSample(int channel, int sampleIndex, float delta, float theta, float alpha, float beta, float gamma) {
  ChannelProfile* profile = &channelProfiles[channel];
  
  // Time in seconds for this sample (with channel-specific phase offset)
  float t = (rawSampleCounter + sampleIndex) / RAW_SAMPLE_RATE + profile->name[0] * 0.01; // Unique per channel
  
  // Generate composite waveform with channel-specific weights
  // Delta: 0.5-4 Hz (slow waves)
  float deltaFreq = 2.0 + random(-50, 50) / 100.0;
  float deltaWave = delta * profile->deltaWeight * sin(2 * PI * deltaFreq * t + channelPhase[channel]);
  
  // Theta: 4-8 Hz (drowsiness, memory)
  float thetaFreq = 6.0 + random(-100, 100) / 100.0;
  float thetaWave = theta * profile->thetaWeight * sin(2 * PI * thetaFreq * t + channelPhase[channel] * 1.3);
  
  // Alpha: 8-13 Hz (relaxed alertness - most prominent)
  float alphaFreq = 10.0 + random(-150, 150) / 100.0;
  float alphaWave = alpha * profile->alphaWeight * sin(2 * PI * alphaFreq * t + channelPhase[channel] * 0.7);
  
  // Beta: 13-30 Hz (active thinking)
  float betaFreq = 20.0 + random(-300, 300) / 100.0;
  float betaWave = beta * profile->betaWeight * sin(2 * PI * betaFreq * t + channelPhase[channel] * 1.7);
  
  // Gamma: 30+ Hz (attention, binding)
  float gammaFreq = 40.0 + random(-500, 500) / 100.0;
  float gammaWave = gamma * profile->gammaWeight * 0.5 * sin(2 * PI * gammaFreq * t + channelPhase[channel] * 2.1);
  
  // Channel-specific noise
  float noise = (random(-100, 100) / 100.0) * profile->noiseAmplitude;
  
  // Occasional artifacts (eye blinks for frontal channels, muscle for temporal)
  float artifact = 0;
  if (random(10000) < profile->artifactProbability * 10000) {
    if (channel == 0 || channel == 4) {
      // Eye blink artifact (frontal channels) - sharp positive spike
      artifact = random(30, 80);
    } else {
      // Muscle artifact (temporal channels) - brief high frequency
      artifact = random(-20, 20);
    }
  }
  
  // Composite signal with all components
  float sample = profile->baselineOffset + 
                 channelDrift[channel] +
                 deltaWave + 
                 thetaWave + 
                 alphaWave + 
                 betaWave + 
                 gammaWave + 
                 noise + 
                 artifact;
  
  // Add occasional 60Hz power line interference (more on some channels)
  if (channel == 0 || channel == 1 || channel == 4) {
    sample += 0.5 * sin(2 * PI * 60.0 * t);
  }
  
  return sample;
}

// ============================================================================
// LIVE MODE PHYSIOLOGICAL STATE UPDATES
// ============================================================================

void updatePhysiologicalStates() {
  fatigueLevel = min(1.0, fatigueLevel + 0.0001 + random(-10, 10) / 10000.0);
  stressLevel = constrain(stressLevel + random(-50, 50) / 1000.0, 0.1, 0.9);
  cognitiveLoad = constrain(cognitiveLoad + random(-100, 100) / 1000.0, 0.2, 0.95);
  thermalStress = min(0.8, thermalStress + 0.00005 + random(-5, 10) / 100000.0);
  hydrationLevel = max(0.6, hydrationLevel - 0.00002);
  
  if (random(100) < 1 && !tbiEvent) {
    tbiEvent = true;
  }
  
  if (tbiEvent && random(30) < 1) {
    tbiEvent = false;
  }
}

// ============================================================================
// FULL DATA PACKET - LIVE MODE (1 Hz)
// Note: Keeping this abbreviated - use the same functions from previous version
// ============================================================================

void sendFullDataPacket() {
  Serial.print("{\"type\":\"full_metrics\",");
  Serial.print("\"mode\":\"live\",");
  Serial.print("\"timestamp\":");
  Serial.print(millis());
  Serial.print(",\"session_minutes\":");
  Serial.print(sessionMinutes, 2);
  Serial.print(",");
  
  sendCompleteEEGData();
  Serial.print(",");
  sendCompletePhysiologicalData();
  Serial.print(",");
  sendCompleteIMUData();
  Serial.print(",");
  sendCompleteEnvironmentalData();
  Serial.print(",");
  sendCompleteCalculatedMetrics();
  
  Serial.println("}");
}

// [Include all the sendComplete...() functions from previous version]
// For brevity, I'll include abbreviated versions here

void sendCompleteEEGData() {
  Serial.print("\"eeg\":{");
  
  float deltaPower = 10.0 + fatigueLevel * 30.0 + random(-20, 20) / 10.0;
  float thetaPower = 15.0 + fatigueLevel * 25.0 + cognitiveLoad * 10.0 + random(-15, 15) / 10.0;
  float alphaPower = 25.0 - fatigueLevel * 15.0 - cognitiveLoad * 10.0 + random(-20, 20) / 10.0;
  float betaPower = 20.0 + cognitiveLoad * 30.0 + stressLevel * 15.0 + random(-15, 15) / 10.0;
  float gammaPower = 8.0 + cognitiveLoad * 12.0 + random(-10, 10) / 10.0;
  
  if (tbiEvent) {
    thetaPower *= 1.5;
    gammaPower *= 0.6;
    deltaPower *= 1.3;
  }
  
  Serial.print("\"left_ear\":{");
  Serial.print("\"Fp1\":{");
  printBandPowers(deltaPower, thetaPower, alphaPower, betaPower, gammaPower, 0.92);
  Serial.print("},\"F7\":{");
  printBandPowers(deltaPower * 0.95, thetaPower * 0.98, alphaPower * 1.02, betaPower * 0.97, gammaPower * 1.01, 0.89);
  Serial.print("},\"T3\":{");
  printBandPowers(deltaPower * 1.03, thetaPower * 1.02, alphaPower * 0.96, betaPower * 1.04, gammaPower * 0.98, 0.91);
  Serial.print("},\"A1\":{");
  printBandPowers(deltaPower * 0.98, thetaPower * 0.97, alphaPower * 1.01, betaPower * 0.99, gammaPower * 1.02, 0.87);
  Serial.print("}},");
  
  Serial.print("\"right_ear\":{");
  Serial.print("\"Fp2\":{");
  printBandPowers(deltaPower * 1.02, thetaPower * 1.01, alphaPower * 0.99, betaPower * 1.01, gammaPower * 0.99, 0.93);
  Serial.print("},\"F8\":{");
  printBandPowers(deltaPower * 0.97, thetaPower * 0.99, alphaPower * 1.03, betaPower * 0.98, gammaPower * 1.03, 0.90);
  Serial.print("},\"T4\":{");
  printBandPowers(deltaPower * 1.01, thetaPower * 1.03, alphaPower * 0.97, betaPower * 1.02, gammaPower * 0.97, 0.88);
  Serial.print("},\"A2\":{");
  printBandPowers(deltaPower * 0.99, thetaPower * 0.98, alphaPower * 1.02, betaPower * 1.00, gammaPower * 1.01, 0.86);
  Serial.print("}},");
  
  float alphaTheta = alphaPower / (thetaPower + 0.01);
  float betaAlpha = betaPower / (alphaPower + 0.01);
  float thetaBeta = thetaPower / (betaPower + 0.01);
  float engagementIndex = betaPower / (thetaPower + alphaPower + 0.01);
  
  Serial.print("\"metrics\":{");
  Serial.print("\"alpha_theta_ratio\":");
  Serial.print(alphaTheta, 2);
  Serial.print(",\"beta_alpha_ratio\":");
  Serial.print(betaAlpha, 2);
  Serial.print(",\"theta_beta_ratio\":");
  Serial.print(thetaBeta, 2);
  Serial.print(",\"engagement_index\":");
  Serial.print(engagementIndex, 2);
  Serial.print("}}");
}

void printBandPowers(float delta, float theta, float alpha, float beta, float gamma, float quality) {
  delta += random(-10, 10) / 10.0;
  theta += random(-10, 10) / 10.0;
  alpha += random(-15, 15) / 10.0;
  beta += random(-10, 10) / 10.0;
  gamma += random(-8, 8) / 10.0;
  
  Serial.print("\"delta\":");
  Serial.print(max(0.0f, delta), 1);
  Serial.print(",\"theta\":");
  Serial.print(max(0.0f, theta), 1);
  Serial.print(",\"alpha\":");
  Serial.print(max(0.0f, alpha), 1);
  Serial.print(",\"beta\":");
  Serial.print(max(0.0f, beta), 1);
  Serial.print(",\"gamma\":");
  Serial.print(max(0.0f, gamma), 1);
  Serial.print(",\"quality\":");
  Serial.print(quality, 2);
}

// Include all other sendComplete functions from previous version
// (sendCompletePhysiologicalData, sendCompleteIMUData, sendCompleteEnvironmentalData, sendCompleteCalculatedMetrics)
// For complete code, copy from previous .ino file

void sendCompletePhysiologicalData() {
  Serial.print("\"physiological\":{");
  
  // PPG metrics
  float heartRate = baselineHR + stressLevel * 30.0 + fatigueLevel * 15.0 + 
                    thermalStress * 25.0 + random(-30, 30) / 10.0;
  heartRate = constrain(heartRate, 50, 180);
  
  float spo2 = baselineSpO2 - fatigueLevel * 3.0 - thermalStress * 2.0 + random(-10, 10) / 10.0;
  spo2 = constrain(spo2, 85, 100);
  
  // HRV metrics
  float rmssd = 50.0 - stressLevel * 30.0 - fatigueLevel * 20.0 + random(-50, 50) / 10.0;
  rmssd = max(10.0, rmssd);
  
  float sdnn = 60.0 - stressLevel * 25.0 - fatigueLevel * 15.0 + random(-40, 40) / 10.0;
  sdnn = max(15.0, sdnn);
  
  float pnn50 = 25.0 - stressLevel * 15.0 - fatigueLevel * 10.0 + random(-30, 30) / 10.0;
  pnn50 = max(0.0, pnn50);
  
  // LF/HF ratio (autonomic balance)
  float lfHfRatio = 1.2 + stressLevel * 2.0 + random(-20, 20) / 100.0;
  lfHfRatio = constrain(lfHfRatio, 0.5, 4.0);
  
  // Temperature
  float coreTemp = baselineTemp + thermalStress * 3.0 + cognitiveLoad * 0.3 + random(-5, 5) / 100.0;
  coreTemp = constrain(coreTemp, 36.0, 40.5);
  
  // Bioimpedance
  float phaseAngle = 5.5 + hydrationLevel * 2.0 - (1.0 - hydrationLevel) * 3.0 + random(-10, 10) / 100.0;
  phaseAngle = constrain(phaseAngle, 3.0, 7.0);
  
  float resistance = 500.0 - hydrationLevel * 100.0 + random(-50, 50) / 10.0;
  float reactance = 60.0 + hydrationLevel * 20.0 + random(-30, 30) / 10.0;
  
  // Output all physiological metrics
  Serial.print("\"heart_rate\":");
  Serial.print(heartRate, 1);
  Serial.print(",\"spo2\":");
  Serial.print(spo2, 1);
  Serial.print(",\"hrv_rmssd\":");
  Serial.print(rmssd, 1);
  Serial.print(",\"hrv_sdnn\":");
  Serial.print(sdnn, 1);
  Serial.print(",\"hrv_pnn50\":");
  Serial.print(pnn50, 1);
  Serial.print(",\"lf_hf_ratio\":");
  Serial.print(lfHfRatio, 2);
  Serial.print(",\"core_temp_c\":");
  Serial.print(coreTemp, 2);
  Serial.print(",\"core_temp_f\":");
  Serial.print(coreTemp * 9.0/5.0 + 32.0, 1);
  Serial.print(",\"bioimpedance_phase_angle\":");
  Serial.print(phaseAngle, 2);
  Serial.print(",\"bioimpedance_resistance\":");
  Serial.print(resistance, 1);
  Serial.print(",\"bioimpedance_reactance\":");
  Serial.print(reactance, 1);
  Serial.print(",\"hydration_level\":");
  Serial.print(hydrationLevel, 2);
  Serial.print("}");
}

void sendCompleteIMUData() {
  Serial.print("\"imu\":{");
  
  float accelX, accelY, accelZ;
  float gyroX, gyroY, gyroZ;
  float magX, magY, magZ;
  float impactMagnitude, overpressure;
  
  if (tbiEvent) {
    // High impact event
    accelX = random(-800, 800) / 100.0;
    accelY = random(-800, 800) / 100.0;
    accelZ = random(-800, 800) / 100.0;
    gyroX = random(-300, 300) / 10.0;
    gyroY = random(-300, 300) / 10.0;
    gyroZ = random(-300, 300) / 10.0;
    impactMagnitude = random(50, 150) / 10.0;
    overpressure = random(40, 80) / 10.0; // 4-8 psi
  } else {
    // Normal movement
    accelX = random(-100, 100) / 100.0;
    accelY = random(-100, 100) / 100.0;
    accelZ = 1.0 + random(-20, 20) / 100.0;
    gyroX = random(-50, 50) / 100.0;
    gyroY = random(-50, 50) / 100.0;
    gyroZ = random(-50, 50) / 100.0;
    impactMagnitude = random(0, 20) / 10.0;
    overpressure = random(0, 5) / 10.0;
  }
  
  // Magnetometer (heading)
  magX = 25.0 + random(-50, 50) / 10.0;
  magY = -15.0 + random(-40, 40) / 10.0;
  magZ = 40.0 + random(-60, 60) / 10.0;
  
  Serial.print("\"accel_x\":");
  Serial.print(accelX, 2);
  Serial.print(",\"accel_y\":");
  Serial.print(accelY, 2);
  Serial.print(",\"accel_z\":");
  Serial.print(accelZ, 2);
  Serial.print(",\"gyro_x\":");
  Serial.print(gyroX, 2);
  Serial.print(",\"gyro_y\":");
  Serial.print(gyroY, 2);
  Serial.print(",\"gyro_z\":");
  Serial.print(gyroZ, 2);
  Serial.print(",\"mag_x\":");
  Serial.print(magX, 1);
  Serial.print(",\"mag_y\":");
  Serial.print(magY, 1);
  Serial.print(",\"mag_z\":");
  Serial.print(magZ, 1);
  Serial.print(",\"impact_magnitude\":");
  Serial.print(impactMagnitude, 1);
  Serial.print(",\"overpressure_psi\":");
  Serial.print(overpressure, 2);
  Serial.print(",\"blast_detected\":");
  Serial.print(tbiEvent ? "true" : "false");
  Serial.print("}");
}

void sendCompleteEnvironmentalData() {
  Serial.print("\"environmental\":{");
  
  float ambientTemp = 25.0 + thermalStress * 20.0 + random(-20, 20) / 10.0;
  float humidity = 50.0 + thermalStress * 30.0 + random(-100, 100) / 10.0;
  humidity = constrain(humidity, 20, 90);
  
  float heatIndex = ambientTemp + 0.5 * (humidity - 50.0) / 10.0;
  float soundLevel = 60.0 + cognitiveLoad * 20.0 + random(-50, 50) / 10.0;
  
  Serial.print("\"ambient_temp_c\":");
  Serial.print(ambientTemp, 1);
  Serial.print(",\"ambient_temp_f\":");
  Serial.print(ambientTemp * 9.0/5.0 + 32.0, 1);
  Serial.print(",\"humidity_percent\":");
  Serial.print(humidity, 1);
  Serial.print(",\"heat_index_c\":");
  Serial.print(heatIndex, 1);
  Serial.print(",\"heat_index_f\":");
  Serial.print(heatIndex * 9.0/5.0 + 32.0, 1);
  Serial.print(",\"sound_level_db\":");
  Serial.print(soundLevel, 1);
  Serial.print(",\"cbrn_detected\":false");
  Serial.print(",\"cbrn_agent\":\"none\"");
  Serial.print(",\"contamination_level\":0.0");
  Serial.print("}");
}

void sendCompleteCalculatedMetrics() {
  Serial.print("\"metrics\":{");
  
  // Alertness Score (0-100)
  float alertness = 100.0 - fatigueLevel * 60.0 - thermalStress * 20.0;
  alertness = constrain(alertness, 20, 100);
  
  // Cognitive Load (0-100)
  float cogLoad = cognitiveLoad * 100.0;
  
  // Fatigue Level (0-100)
  float fatigue = fatigueLevel * 100.0;
  
  // Physical Load
  float physicalLoad = (stressLevel * 40.0 + thermalStress * 60.0);
  physicalLoad = constrain(physicalLoad, 0, 100);
  
  // Stress Level (0-100)
  float stress = stressLevel * 100.0;
  
  // TBI Risk Score
  float tbiRisk = tbiEvent ? random(70, 95) : random(0, 15);
  
  // Heat Stress Risk
  float heatRisk = thermalStress * 100.0;
  
  // Mission Readiness (composite)
  float missionReadiness = (alertness * 0.4 + (100 - fatigue) * 0.3 + 
                            (100 - stress) * 0.2 + (100 - heatRisk) * 0.1);
  missionReadiness = constrain(missionReadiness, 0, 100);
  
  // Determine alert level
  String alertLevel;
  if (tbiEvent || heatRisk > 80 || alertness < 40 || missionReadiness < 50) {
    alertLevel = "critical";
  } else if (fatigue > 70 || stress > 75 || heatRisk > 60 || alertness < 60) {
    alertLevel = "warning";
  } else if (fatigue > 50 || stress > 60 || cogLoad > 80) {
    alertLevel = "caution";
  } else {
    alertLevel = "normal";
  }
  
  Serial.print("\"alertness_score\":");
  Serial.print(alertness, 1);
  Serial.print(",\"cognitive_load\":");
  Serial.print(cogLoad, 1);
  Serial.print(",\"fatigue_level\":");
  Serial.print(fatigue, 1);
  Serial.print(",\"physical_load\":");
  Serial.print(physicalLoad, 1);
  Serial.print(",\"stress_level\":");
  Serial.print(stress, 1);
  Serial.print(",\"tbi_risk_score\":");
  Serial.print(tbiRisk, 1);
  Serial.print(",\"heat_stress_risk\":");
  Serial.print(heatRisk, 1);
  Serial.print(",\"mission_readiness\":");
  Serial.print(missionReadiness, 1);
  Serial.print(",\"alert_level\":\"");
  Serial.print(alertLevel);
  Serial.print("\"}");
}

void sendPlaybackDataPacket() {
  if (playbackIndex >= playbackLength) {
    Serial.println("{\"status\":\"Playback complete\",\"session_id\":" + String(currentPlaybackSession) + "}");
    currentMode = MODE_IDLE;
    return;
  }
  
  SessionDataPoint* dataPoint = nullptr;
  
  switch (currentPlaybackSession) {
    case 1: dataPoint = &session1Data[playbackIndex]; break;
    case 2: dataPoint = &session2Data[playbackIndex]; break;
    case 3: dataPoint = &session3Data[playbackIndex]; break;
    case 4: dataPoint = &session4Data[playbackIndex]; break;
  }
  
  if (dataPoint == nullptr) {
    Serial.println("{\"error\":\"Playback data not found\"}");
    currentMode = MODE_IDLE;
    return;
  }
  
  // Set state variables based on session data point
  // This allows us to generate realistic sensor data from stored summary
  fatigueLevel = dataPoint->fatigue / 100.0;
  stressLevel = dataPoint->stress / 100.0;
  cognitiveLoad = dataPoint->cogLoad / 100.0;
  thermalStress = (dataPoint->coreTemp - 36.5) / 4.0;  // Derive from temp
  tbiEvent = dataPoint->tbiFlag;
  hydrationLevel = dataPoint->missionReadiness / 100.0;  // Approximate from readiness
  
  // Send FULL data packet just like live mode
  Serial.print("{\"type\":\"full_metrics\",");
  Serial.print("\"mode\":\"playback\",");
  Serial.print("\"session_id\":");
  Serial.print(currentPlaybackSession);
  Serial.print(",\"playback_index\":");
  Serial.print(playbackIndex);
  Serial.print(",\"timestamp\":");
  Serial.print(millis());
  Serial.print(",\"session_minutes\":");
  // Calculate realistic session minutes based on data point position
  int sessionDuration = storedSessions[currentPlaybackSession - 1].durationMinutes;
  float simulatedMinutes = (float)(playbackIndex * sessionDuration) / (float)playbackLength;
  Serial.print(simulatedMinutes, 2);
  Serial.print(",");
  
  // Complete EEG processed data (generated from state variables)
  sendCompleteEEGData();
  Serial.print(",");
  
  // Full physiological suite
  sendPlaybackPhysiologicalData(dataPoint);
  Serial.print(",");
  
  // Complete IMU data (generated from state variables)
  sendCompleteIMUData();
  Serial.print(",");
  
  // Environmental sensors (generated from state variables)
  sendCompleteEnvironmentalData();
  Serial.print(",");
  
  // All calculated metrics (use stored values)
  sendPlaybackCalculatedMetrics(dataPoint);
  
  Serial.println("}");
  
  playbackIndex++;
}

void sendPlaybackPhysiologicalData(SessionDataPoint* dp) {
  Serial.print("\"physiological\":{");
  
  // Use stored heart rate and temperature
  float heartRate = dp->heartRate;
  
  // Derive other metrics from stored values
  float spo2 = 98.0 - dp->fatigue * 0.08 + random(-5, 5) / 10.0;
  spo2 = constrain(spo2, 85, 100);
  
  float rmssd = 50.0 - dp->stress * 0.35 + random(-20, 20) / 10.0;
  rmssd = max(10.0, rmssd);
  
  float sdnn = 60.0 - dp->stress * 0.3 + random(-15, 15) / 10.0;
  sdnn = max(15.0, sdnn);
  
  float pnn50 = 25.0 - dp->stress * 0.25 + random(-10, 10) / 10.0;
  pnn50 = max(0.0, pnn50);
  
  float lfHfRatio = 1.2 + dp->stress * 0.02 + random(-10, 10) / 100.0;
  lfHfRatio = constrain(lfHfRatio, 0.5, 4.0);
  
  float coreTemp = dp->coreTemp;
  
  // Bioimpedance from mission readiness
  float phaseAngle = 3.0 + (dp->missionReadiness / 100.0) * 3.5 + random(-5, 5) / 100.0;
  phaseAngle = constrain(phaseAngle, 3.0, 7.0);
  
  float resistance = 550.0 - (dp->missionReadiness / 100.0) * 100.0 + random(-20, 20) / 10.0;
  float reactance = 55.0 + (dp->missionReadiness / 100.0) * 25.0 + random(-10, 10) / 10.0;
  float hydration = (phaseAngle - 3.0) / 4.0;
  
  Serial.print("\"heart_rate\":");
  Serial.print(heartRate, 1);
  Serial.print(",\"spo2\":");
  Serial.print(spo2, 1);
  Serial.print(",\"hrv_rmssd\":");
  Serial.print(rmssd, 1);
  Serial.print(",\"hrv_sdnn\":");
  Serial.print(sdnn, 1);
  Serial.print(",\"hrv_pnn50\":");
  Serial.print(pnn50, 1);
  Serial.print(",\"lf_hf_ratio\":");
  Serial.print(lfHfRatio, 2);
  Serial.print(",\"core_temp_c\":");
  Serial.print(coreTemp, 2);
  Serial.print(",\"core_temp_f\":");
  Serial.print(coreTemp * 9.0/5.0 + 32.0, 1);
  Serial.print(",\"bioimpedance_phase_angle\":");
  Serial.print(phaseAngle, 2);
  Serial.print(",\"bioimpedance_resistance\":");
  Serial.print(resistance, 1);
  Serial.print(",\"bioimpedance_reactance\":");
  Serial.print(reactance, 1);
  Serial.print(",\"hydration_level\":");
  Serial.print(hydration, 2);
  Serial.print("}");
}

void sendPlaybackCalculatedMetrics(SessionDataPoint* dp) {
  Serial.print("\"metrics\":{");
  
  // Use stored values directly
  float alertness = dp->alertness;
  float cogLoad = dp->cogLoad;
  float fatigue = dp->fatigue;
  float stress = dp->stress;
  float missionReadiness = dp->missionReadiness;
  
  // Calculate derived metrics
  float physicalLoad = (stress * 0.4 + fatigue * 0.6);
  float tbiRisk = dp->tbiFlag ? 85.0 : 8.0;
  float heatRisk = (dp->coreTemp - 36.5) * 20.0;
  heatRisk = constrain(heatRisk, 0.0, 100.0);
  
  Serial.print("\"alertness_score\":");
  Serial.print(alertness, 1);
  Serial.print(",\"cognitive_load\":");
  Serial.print(cogLoad, 1);
  Serial.print(",\"fatigue_level\":");
  Serial.print(fatigue, 1);
  Serial.print(",\"physical_load\":");
  Serial.print(physicalLoad, 1);
  Serial.print(",\"stress_level\":");
  Serial.print(stress, 1);
  Serial.print(",\"tbi_risk_score\":");
  Serial.print(tbiRisk, 1);
  Serial.print(",\"heat_stress_risk\":");
  Serial.print(heatRisk, 1);
  Serial.print(",\"mission_readiness\":");
  Serial.print(missionReadiness, 1);
  Serial.print(",\"alert_level\":\"");
  Serial.print(dp->alertLevel);
  Serial.print("\"}");
}

