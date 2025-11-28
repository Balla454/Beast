"""
BeAST Live Data Stream Service - FIXED for Raw EEG
Properly handles dual-rate streaming: raw_eeg (4 Hz) and full_metrics (1 Hz)
"""

import serial
import json
import asyncio
import websockets
from datetime import datetime, timezone
from typing import Dict, Set, Optional
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

SERIAL_PORT = '/dev/cu.usbmodem213401'  # Change to your Arduino port
BAUD_RATE = 115200

WEBSOCKET_HOST = 'localhost'
WEBSOCKET_PORT = 8765

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VALID_COMMANDS = ['L', 'S', 'P1', 'P2', 'P3', 'P4', 'A', 'R', '?']

# ============================================================================
# WEBSOCKET SERVER
# ============================================================================

class WebSocketServer:
    """WebSocket server for real-time dashboard updates"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.command_queue = asyncio.Queue()
        logger.info(f"WebSocket server configured for {host}:{port}")
    
    async def register(self, websocket):
        """Register new client"""
        self.clients.add(websocket)
        logger.info(f"Dashboard connected. Total clients: {len(self.clients)}")
        
        await websocket.send(json.dumps({
            'type': 'connected',
            'message': 'Connected to BeAST live data stream',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }))
    
    async def unregister(self, websocket):
        """Unregister client"""
        self.clients.remove(websocket)
        logger.info(f"Dashboard disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if self.clients:
            message_json = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_json) for client in self.clients],
                return_exceptions=True
            )
    
    async def handler(self, websocket, path):
        """Handle WebSocket connections"""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    try:
                        cmd_data = json.loads(message)
                        logger.info(f"📨 Received from dashboard: {cmd_data}")
                        
                        if 'command' in cmd_data:
                            command = cmd_data['command']
                        else:
                            logger.warning(f"No 'command' field in message: {cmd_data}")
                            continue
                    except json.JSONDecodeError:
                        command = message.strip()
                        logger.info(f"📨 Received raw command: {command}")
                    
                    if command in VALID_COMMANDS:
                        await self.command_queue.put(command)
                        logger.info(f"📤 Queued command for Arduino: {command}")
                        
                        await websocket.send(json.dumps({
                            'type': 'command_ack',
                            'command': command,
                            'status': 'queued'
                        }))
                    else:
                        logger.warning(f"Invalid command: {command}")
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': f'Unknown command: {command}. Valid: {", ".join(VALID_COMMANDS)}'
                        }))
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        finally:
            await self.unregister(websocket)
    
    async def start(self):
        """Start WebSocket server"""
        async with websockets.serve(self.handler, self.host, self.port):
            logger.info(f"✅ WebSocket server started on ws://{self.host}:{self.port}")
            logger.info("Waiting for dashboard connections...")
            await asyncio.Future()

# ============================================================================
# SERIAL HANDLER
# ============================================================================

class SerialHandler:
    """Handles Arduino serial communication"""
    
    def __init__(self, port: str, baud_rate: int):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn: Optional[serial.Serial] = None
        self.ws_server: Optional[WebSocketServer] = None
        self.stats = {
            'raw_eeg_packets': 0,
            'full_metric_packets': 0,
            'commands_sent': 0,
            'errors': 0
        }
    
    def connect(self) -> bool:
        """Connect to Arduino"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            logger.info(f"✅ Connected to Arduino on {self.port}")
            
            import time
            time.sleep(2)
            
            startup_lines = []
            for _ in range(5):
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        startup_lines.append(line)
                        logger.info(f"Arduino: {line}")
                time.sleep(0.2)
            
            logger.info("📡 Ready to receive commands from dashboard")
            logger.info(f"Valid commands: {', '.join(VALID_COMMANDS)}")
            
            return True
        except serial.SerialException as e:
            logger.error(f"❌ Failed to connect to Arduino: {e}")
            logger.error(f"Make sure Arduino is connected to {self.port}")
            return False
    
    def send_command(self, command: str):
        """Send command to Arduino"""
        if self.serial_conn:
            try:
                self.serial_conn.write(command.encode('utf-8'))
                self.serial_conn.flush()
                self.stats['commands_sent'] += 1
                logger.info(f"📤 Sent to Arduino: '{command}'")
            except Exception as e:
                logger.error(f"Error sending command: {e}")
    
    def read_line(self) -> Optional[str]:
        """Read line from serial port"""
        if self.serial_conn and self.serial_conn.in_waiting > 0:
            try:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                return line
            except Exception as e:
                logger.error(f"Error reading serial: {e}")
                self.stats['errors'] += 1
        return None
    
    async def run(self, ws_server: WebSocketServer):
        """Main processing loop"""
        self.ws_server = ws_server
        
        if not self.connect():
            logger.error("Cannot start - Arduino connection failed")
            return
        
        logger.info("="*60)
        logger.info("BeAST Live Data Stream - Ready")
        logger.info("="*60)
        logger.info("Expecting:")
        logger.info("  - Raw EEG: 4 Hz (every 0.25s)")
        logger.info("  - Full Metrics: 1 Hz (every 1.0s)")
        logger.info(f"Commands: {', '.join(VALID_COMMANDS)}")
        logger.info("="*60)
        
        try:
            while True:
                # Check for commands from dashboard
                try:
                    command = ws_server.command_queue.get_nowait()
                    logger.info(f"🎮 Processing command from dashboard: {command}")
                    self.send_command(command)
                    
                    await ws_server.broadcast({
                        'type': 'command_sent',
                        'command': command,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                except asyncio.QueueEmpty:
                    pass
                
                # Read serial data
                line = self.read_line()
                
                if line:
                    try:
                        # Parse JSON from Arduino
                        data = json.loads(line)
                        
                        # CRITICAL: Forward data DIRECTLY based on 'type' field
                        # Don't wrap in another layer!
                        
                        if 'type' in data:
                            packet_type = data['type']
                            
                            if packet_type == 'raw_eeg':
                                # Raw EEG packet - forward as-is
                                self.stats['raw_eeg_packets'] += 1
                                await ws_server.broadcast(data)
                                
                                if self.stats['raw_eeg_packets'] % 20 == 0:  # Every 5 seconds
                                    logger.info(f"📊 Raw EEG: {self.stats['raw_eeg_packets']} packets")
                            
                            elif packet_type == 'full_metrics':
                                # Full metrics packet - forward as-is
                                self.stats['full_metric_packets'] += 1
                                await ws_server.broadcast(data)
                                
                                logger.info(
                                    f"📊 Metrics: {self.stats['full_metric_packets']} full, "
                                    f"{self.stats['raw_eeg_packets']} raw EEG, "
                                    f"{len(ws_server.clients)} clients"
                                )
                            
                            else:
                                # Other typed messages (status, sessions, etc.)
                                await ws_server.broadcast(data)
                                logger.info(f"📨 Arduino message type: {packet_type}")
                        
                        elif 'status' in data:
                            # Status messages
                            logger.info(f"🤖 Arduino: {data.get('status', 'unknown status')}")
                            await ws_server.broadcast(data)
                        
                        elif 'sessions' in data:
                            # Session list
                            logger.info(f"📋 Session list received")
                            await ws_server.broadcast(data)
                        
                        else:
                            # Unknown JSON structure - forward anyway
                            logger.debug(f"Unknown JSON structure: {list(data.keys())}")
                            await ws_server.broadcast(data)
                    
                    except json.JSONDecodeError:
                        # Non-JSON messages (status/info lines)
                        logger.debug(f"Arduino: {line}")
                
                await asyncio.sleep(0.001)  # Very small delay
        
        except KeyboardInterrupt:
            logger.info("\n⚠️  Interrupted by user")
        finally:
            if self.serial_conn:
                self.serial_conn.close()
                logger.info("Serial connection closed")
            
            logger.info("="*60)
            logger.info(f"Final Stats:")
            logger.info(f"  Raw EEG Packets: {self.stats['raw_eeg_packets']}")
            logger.info(f"  Full Metric Packets: {self.stats['full_metric_packets']}")
            logger.info(f"  Commands Sent: {self.stats['commands_sent']}")
            logger.info(f"  Errors: {self.stats['errors']}")
            logger.info("="*60)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

async def main():
    """Main application entry point"""
    
    print("\n" + "="*60)
    print("BeAST Full System Live Stream with Raw EEG")
    print("Raw EEG: 4 Hz | Full Metrics: 1 Hz")
    print("="*60 + "\n")
    
    ws_server = WebSocketServer(WEBSOCKET_HOST, WEBSOCKET_PORT)
    serial_handler = SerialHandler(SERIAL_PORT, BAUD_RATE)
    
    ws_task = asyncio.create_task(ws_server.start())
    await asyncio.sleep(1)
    
    serial_task = asyncio.create_task(serial_handler.run(ws_server))
    
    await asyncio.gather(ws_task, serial_task, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutdown requested")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
