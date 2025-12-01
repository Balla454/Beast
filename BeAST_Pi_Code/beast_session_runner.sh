#!/bin/bash
# Simple runner for the BeAST Arduino → SQL session recorder

set -e  # Exit on error

# Use the current logged-in user
CURRENT_USER=$(whoami)
PROJECT_DIR="/home/$CURRENT_USER/Beast/BeAST_Pi_Code"
VENV_DIR="$PROJECT_DIR/venv"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
WORKING_DIR="$PROJECT_DIR/Live Connections Simulator Test"

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory not found at $PROJECT_DIR"
    exit 1
fi

# Navigate to the working directory
if [ ! -d "$WORKING_DIR" ]; then
    echo "Error: Working directory not found at $WORKING_DIR"
    exit 1
fi
cd "$WORKING_DIR"

# Function to create virtual environment
create_venv() {
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    
    # Activate the new environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements if they exist
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "Installing dependencies from requirements.txt..."
        pip install -r "$REQUIREMENTS_FILE"
    else
        echo "Warning: No requirements.txt found at $REQUIREMENTS_FILE"
        echo "Installing minimal dependencies..."
        pip install pyserial mysql-connector-python
    fi
}

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    create_venv
else
    # Activate existing virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Check if virtual environment is still valid (has pip)
    if ! command -v pip >/dev/null 2>&1; then
        echo "Virtual environment appears corrupted. Recreating..."
        rm -rf "$VENV_DIR"
        create_venv
    fi
fi

echo "Virtual environment is active. Python path: $(which python)"
echo "Running beast_arduino_to_sql_updated.py..."

# Run the Python script
python beast_arduino_to_sql_updated.py
