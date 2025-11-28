-- Schemas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS agg;

-- EEG raw time series
CREATE TABLE IF NOT EXISTS raw.earpiece_data (
    id         BIGSERIAL PRIMARY KEY,
    ts         timestamptz NOT NULL,
    device_id  uuid,
    session_id uuid,
    side       text,
    mode       text,
    data       real[]
);
CREATE INDEX IF NOT EXISTS earpiece_ts_idx ON raw.earpiece_data (ts);

-- EEG bands
CREATE TABLE IF NOT EXISTS raw.eeg_band (
    id         BIGSERIAL PRIMARY KEY,
    ts         timestamptz NOT NULL,
    device_id  uuid,
    session_id uuid,
    alpha      real,
    beta       real,
    gamma      real,
    delta      real,
    theta      real
);
CREATE INDEX IF NOT EXISTS eeg_band_ts_idx ON raw.eeg_band (ts);

-- Cognitive load / alertness / fatigue
CREATE TABLE IF NOT EXISTS raw.cognitive_metrics (
    id             BIGSERIAL PRIMARY KEY,
    ts             timestamptz NOT NULL,
    device_id      uuid,
    session_id     uuid,
    cognitive_load real,
    alertness      real,
    fatigue        real
);
CREATE INDEX IF NOT EXISTS cognitive_ts_idx ON raw.cognitive_metrics (ts);

-- Heart rate + SPO2
CREATE TABLE IF NOT EXISTS raw.ppg_reading (
    id         BIGSERIAL PRIMARY KEY,
    ts         timestamptz NOT NULL,
    device_id  uuid,
    session_id uuid,
    hr_bpm     real,
    spo2_pct   real,
    raw_data   real[]
);
CREATE INDEX IF NOT EXISTS ppg_ts_idx ON raw.ppg_reading (ts);

-- Hydration
CREATE TABLE IF NOT EXISTS raw.hydration_reading (
    id            BIGSERIAL PRIMARY KEY,
    ts            timestamptz NOT NULL,
    device_id     uuid,
    session_id    uuid,
    hydration_pct real
);
CREATE INDEX IF NOT EXISTS hydration_ts_idx ON raw.hydration_reading (ts);

-- Core temp
CREATE TABLE IF NOT EXISTS raw.temp_reading (
    id          BIGSERIAL PRIMARY KEY,
    ts          timestamptz NOT NULL,
    device_id   uuid,
    session_id  uuid,
    core_temp_c real
);
CREATE INDEX IF NOT EXISTS temp_ts_idx ON raw.temp_reading (ts);

-- IMU / impacts
CREATE TABLE IF NOT EXISTS raw.bioz_reading (
    id         BIGSERIAL PRIMARY KEY,
    ts         timestamptz NOT NULL,
    device_id  uuid,
    session_id uuid,
    accel_x    real,
    accel_y    real,
    accel_z    real,
    gyro_x     real,
    gyro_y     real,
    gyro_z     real
);
CREATE INDEX IF NOT EXISTS imu_ts_idx ON raw.bioz_reading (ts);

-- Environment
CREATE TABLE IF NOT EXISTS raw.environment (
    id           BIGSERIAL PRIMARY KEY,
    ts           timestamptz NOT NULL,
    device_id    uuid,
    ambient_temp real,
    humidity_pct real,
    heat_index_c real
);
CREATE INDEX IF NOT EXISTS environment_ts_idx ON raw.environment (ts);

-- Zone durations (aggregates)
CREATE TABLE IF NOT EXISTS agg.zone_durations (
    id          BIGSERIAL PRIMARY KEY,
    device_id   uuid NOT NULL,
    session_id  uuid NOT NULL,
    metric      text NOT NULL,     -- 'hr', 'spo2', 'cognitive', 'fatigue', 'hydration', etc.
    zone        text NOT NULL,     -- e.g. 'rest', 'moderate', 'high'
    duration_s  integer NOT NULL,  -- seconds
    computed_at timestamptz DEFAULT now()
);
