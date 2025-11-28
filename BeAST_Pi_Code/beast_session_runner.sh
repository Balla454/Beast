#!/bin/bash
# Simple runner for the BeAST Arduino → SQL session recorder

cd "/home/pibeast/Downloads/Live Connections Simulator Test"
source /home/pibeast/Downloads/TheBeast/venv/bin/activate
python beast_arduino_to_sql_updated.py
