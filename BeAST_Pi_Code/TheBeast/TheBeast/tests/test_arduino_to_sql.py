import pytest
import sys
import os
import json
from pathlib import Path

# Add the directory containing the script to sys.path
# The path contains spaces, so we need to be careful
project_root = Path(__file__).parent.parent.parent.parent
script_dir = project_root / "Live Connections Simulator Test"
sys.path.append(str(script_dir))

# Import the module
try:
    import beast_arduino_to_sql as bats
except ImportError:
    # Fallback if the path manipulation fails or file not found
    pytest.skip("Could not import beast_arduino_to_sql", allow_module_level=True)

def test_pg_literal():
    """Test PostgreSQL literal escaping"""
    assert bats.pg_literal(None) == "NULL"
    assert bats.pg_literal(123) == "123"
    assert bats.pg_literal(12.34) == "12.34"
    assert bats.pg_literal("hello") == "'hello'"
    assert bats.pg_literal("O'Connor") == "'O''Connor'"
    assert bats.pg_literal([1, 2]) == "ARRAY[1, 2]"
    
    # Test dict (JSON)
    d = {"key": "value"}
    literal = bats.pg_literal(d)
    assert literal.startswith("'")
    assert literal.endswith("'::jsonb")
    assert '"key": "value"' in literal

def test_build_insert_eeg_raw():
    """Test building INSERT statement for raw EEG"""
    packet = {
        "type": "eeg_raw",
        "ts": "2023-01-01T12:00:00",
        "channel": 1,
        "samples": [10, 20, 30]
    }
    session_id = "sess-123"
    device_id = "dev-456"
    
    sql = bats.build_insert(packet, session_id, device_id)
    
    assert sql is not None
    assert "INSERT INTO raw.eeg_raw" in sql
    assert "'sess-123'" in sql
    assert "'dev-456'" in sql
    assert "ARRAY[10, 20, 30]" in sql

def test_build_insert_clinical_metrics():
    """Test building INSERT statement for clinical metrics"""
    packet = {
        "type": "clinical_metrics",
        "ts": "2023-01-01T12:00:00",
        "alertness": 85.5,
        "cognitive_load": 40.2,
        "fatigue": 10.0,
        "zone_durations": {"green": 100, "red": 0}
    }
    session_id = "sess-123"
    device_id = "dev-456"
    
    sql = bats.build_insert(packet, session_id, device_id)
    
    assert sql is not None
    assert "INSERT INTO raw.clinical_metrics" in sql
    assert "85.5" in sql
    assert "40.2" in sql
    assert "10.0" in sql
    # Check JSON field
    assert "green" in sql

def test_build_insert_unknown_type():
    """Test that unknown packet types return None"""
    packet = {
        "type": "unknown_packet_type",
        "data": 123
    }
    sql = bats.build_insert(packet, "sess", "dev")
    assert sql is None
