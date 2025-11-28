-- ==========================================================
-- BeAST Project – 8 Hour Synthetic Physiological Data
-- Generates 8 hours of 1-minute sampled data (480 samples)
-- ==========================================================

-- Choose a static device + session UUID for synthetic data
-- You can change these if desired.
DO $$
DECLARE
    device UUID := '11111111-2222-3333-4444-555555555555';
    session UUID := 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
    ts TIMESTAMPTZ := now() - INTERVAL '8 hours';   -- start 8 hours ago
    i INT := 0;
BEGIN
    RAISE NOTICE 'Generating 8 hours of synthetic BeAST data...';

    WHILE i < 480 LOOP   -- 8 hours * 60 minutes = 480 rows

        -- ===========================
        -- EEG Bands (Alpha/Beta/etc)
        -- ===========================
        INSERT INTO raw.eeg_band (
            ts, device_id, session_id,
            alpha, beta, gamma, delta, theta
        ) VALUES (
            ts, device, session,
            5 + random()*3,
            7 + random()*4,
            3 + random()*2,
            4 + random()*3,
            6 + random()*3
        );

        -- ===========================
        -- Cognitive Metrics
        -- ===========================
        INSERT INTO raw.cognitive_metrics (
            ts, device_id, session_id,
            cognitive_load, alertness, fatigue
        ) VALUES (
            ts, device, session,
            30 + random()*50,     -- cognitive load 30–80
            40 + random()*50,     -- alertness 40–90
            10 + random()*60      -- fatigue 10–70
        );

        -- ===========================
        -- Heart Rate + SPO2
        -- ===========================
        INSERT INTO raw.ppg_reading (
            ts, device_id, session_id,
            hr_bpm, spo2_pct, raw_data
        ) VALUES (
            ts, device, session,
            60 + random()*40,         -- 60–100 bpm
            92 + random()*6,          -- 92–98% SpO2
            ARRAY[
                (50 + random()*50)::real,
                (50 + random()*50)::real,
                (50 + random()*50)::real
            ]
        );

        -- ===========================
        -- Hydration
        -- ===========================
        INSERT INTO raw.hydration_reading (
            ts, device_id, session_id, hydration_pct
        ) VALUES (
            ts, device, session,
            40 + random()*20          -- 40–60%
        );

        -- ===========================
        -- Core Body Temperature
        -- ===========================
        INSERT INTO raw.temp_reading (
            ts, device_id, session_id, core_temp_c
        ) VALUES (
            ts, device, session,
            36.2 + random()*1.2       -- 36.2–37.4 °C
        );

        -- ===========================
        -- IMU (Accelerometer & Gyro)
        -- ===========================
        INSERT INTO raw.bioz_reading (
            ts, device_id, session_id,
            accel_x, accel_y, accel_z,
            gyro_x, gyro_y, gyro_z
        ) VALUES (
            ts, device, session,
            (-1 + random()*2),  -- accel x (±1g)
            (-1 + random()*2),
            (9 + random()*0.5), -- z tends to include gravity
            (-0.5 + random()*1),
            (-0.5 + random()*1),
            (-0.5 + random()*1)
        );

        -- ===========================
        -- Environment (Temp/Humidity)
        -- ===========================
        INSERT INTO raw.environment (
            ts, device_id,
            ambient_temp, humidity_pct, heat_index_c
        ) VALUES (
            ts, device,
            20 + random()*10,      -- ambient temp 20–30 °C
            40 + random()*40,      -- humidity 40–80%
            20 + random()*12       -- heat index 20–32 °C
        );

        -- Step forward by 1 minute
        ts := ts + INTERVAL '1 minute';
        i := i + 1;

    END LOOP;

END $$;
