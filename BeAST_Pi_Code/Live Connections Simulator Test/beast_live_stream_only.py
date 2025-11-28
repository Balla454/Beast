"""
BeAST Live Data Stream Service (No Database)
Simplified version that only reads Arduino serial data and broadcasts to dashboard via WebSocket

Features:
- Serial port reading from Arduino
- JSON validation
- WebSocket server for real-time dashboard updates
- NO database storage (live streaming only)
- Command interface for Arduino mode control
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

# Serial Configuration
SERIAL_PORT = '/dev/ttyACM0'  # Change to your Arduino port
# SERIAL_PORT = '/dev/ttyACM0'  # Linux
# SERIAL_PORT = '/dev/cu.usbmodem14101'  # Mac
BAUD_RATE = 115200

# WebSocket Configuration
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# WEBSOCKET SERVER
# ============================================================================

class WebSocketServer:
    """WebSocket server for real-time dashboard updates"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.command_queue = asyncio.Queue()  # Queue for commands to Arduino
        logger.info(f"WebSocket server configured for {host}:{port}")
    
    async def register(self, websocket):
        """Register new client"""
        self.clients.add(websocket)
        logger.info(f"Dashboard connected. Total clients: {len(self.clients)}")
        
        # Send welcome message
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
            # Send to all clients, ignore errors
            await asyncio.gather(
                *[client.send(message_json) for client in self.clients],
                return_exceptions=True
            )
    
    async def handler(self, websocket, path):
        """Handle WebSocket connections"""
        await self.register(websocket)
        try:
            async for message in websocket:
                # Handle commands from dashboard
                try:
                    cmd = json.loads(message)
                    logger.info(f"📨 Received from dashboard: {cmd}")
                    
                    # Check if it's a command for Arduino
                    if 'command' in cmd:
                        command = cmd['command']
                        if command in ['L1', 'L2', 'L3', 'S']:
                            # Put command in queue for processing by main loop
                            await self.command_queue.put(command)
                            logger.info(f"📤 Queued command for Arduino: {command}")
                            
                            # Send acknowledgment back to client
                            await websocket.send(json.dumps({
                                'type': 'command_ack',
                                'command': command,
                                'status': 'queued'
                            }))
                        else:
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': f'Unknown command: {command}. Valid: L1, L2, L3, S'
                            }))
                    else:
                        # Echo back for other messages
                        await websocket.send(json.dumps({
                            'type': 'message_received',
                            'data': cmd
                        }))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        finally:
            await self.unregister(websocket)
    
    async def start(self):
        """Start WebSocket server"""
        async with websockets.serve(self.handler, self.host, self.port):
            logger.info(f"✅ WebSocket server started on ws://{self.host}:{self.port}")
            logger.info("Waiting for dashboard connections...")
            await asyncio.Future()  # Run forever

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
            'packets_received': 0,
            'packets_broadcast': 0,
            'errors': 0
        }
    
    def connect(self) -> bool:
        """Connect to Arduino"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            logger.info(f"✅ Connected to Arduino on {self.port}")
            
            # Wait for Arduino to be ready
            import time
            time.sleep(4)
            
            # Clear any startup messages
            while self.serial_conn.in_waiting > 0:
                self.serial_conn.readline()
            
            logger.info("📡 Waiting for dashboard to send mode command (L1/L2/L3)")
            
            return True
        except serial.SerialException as e:
            logger.error(f"❌ Failed to connect to Arduino: {e}")
            logger.error(f"Make sure Arduino is connected to {self.port}")
            return False
    
    def send_command(self, command: str):
        """Send command to Arduino"""
        if self.serial_conn:
            self.serial_conn.write(command.encode('utf-8'))
            logger.info(f"📤 Sent command to Arduino: {command}")
    
    def send_command(self, command: str):
        """Send command to Arduino"""
        if self.serial_conn:
            try:
                self.serial_conn.write(f"{command}\n".encode('utf-8'))
                self.serial_conn.flush()
                logger.info(f"📤 Sent command to Arduino: {command}")
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
        logger.info("Waiting for Arduino data...")
        logger.info("Dashboard can send commands: L1, L2, L3, S")
        logger.info("="*60)
        
        try:
            while True:
                # Check for commands from dashboard
                try:
                    command = ws_server.command_queue.get_nowait()
                    logger.info(f"🎮 Processing command from dashboard: {command}")
                    self.send_command(command)
                    
                    # Broadcast command confirmation
                    await ws_server.broadcast({
                        'type': 'command_sent',
                        'command': command,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                except asyncio.QueueEmpty:
                    pass  # No commands waiting
                
                # Read serial data
                line = self.read_line()
                
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Handle batch data
                        if 'batch' in data:
                            batch = data['batch']
                            self.stats['packets_received'] += len(batch)
                            
                            # Create timestamp for this batch
                            timestamp = datetime.now(timezone.utc).isoformat()
                            
                            # Broadcast to dashboard
                            await ws_server.broadcast({
                                'type': 'data_batch',
                                'batch': batch,
                                'timestamp': timestamp,
                                'stats': self.stats.copy()
                            })
                            
                            self.stats['packets_broadcast'] += len(batch)
                            
                            # Log summary (not every packet)
                            if self.stats['packets_received'] % 40 == 0:  # Every 5 seconds
                                logger.info(
                                    f"📊 Stats: {self.stats['packets_received']} packets received, "
                                    f"{self.stats['packets_broadcast']} broadcast, "
                                    f"{len(ws_server.clients)} clients"
                                )
                        
                        # Handle status messages
                        elif 'status' in data:
                            logger.info(f"🤖 Arduino status: {data['status']}")
                            await ws_server.broadcast({
                                'type': 'status',
                                'data': data,
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            })
                        
                        # Handle session list
                        elif 'sessions' in data:
                            logger.info(f"📋 Session list received")
                            await ws_server.broadcast({
                                'type': 'sessions',
                                'data': data,
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            })
                    
                    except json.JSONDecodeError:
                        # Non-JSON status messages
                        logger.debug(f"Status: {line}")
                
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
        
        except KeyboardInterrupt:
            logger.info("\n⚠️  Interrupted by user")
        finally:
            if self.serial_conn:
                self.serial_conn.close()
                logger.info("Serial connection closed")
            
            logger.info("="*60)
            logger.info(f"Final Stats:")
            logger.info(f"  Packets Received: {self.stats['packets_received']}")
            logger.info(f"  Packets Broadcast: {self.stats['packets_broadcast']}")
            logger.info(f"  Errors: {self.stats['errors']}")
            logger.info("="*60)

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

async def command_interface(serial_handler: SerialHandler):
    """
    Simple command interface to send commands to Arduino
    Run this in a separate terminal if needed
    """
    logger.info("\n" + "="*60)
    logger.info("Command Interface (optional)")
    logger.info("="*60)
    logger.info("You can send commands to Arduino:")
    logger.info("  L - Start live streaming")
    logger.info("  S - Stop streaming")
    logger.info("  P[n] - Playback session n (e.g., P1, P2)")
    logger.info("  A - List sessions")
    logger.info("  R - Reset")
    logger.info("  ? - Status")
    logger.info("="*60 + "\n")
    
    # This is just informational - actual commands should be sent
    # via Arduino Serial Monitor or separate script
    await asyncio.sleep(0.1)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

async def main():
    """Main application entry point"""
    
    print("\n" + "="*60)
    print("BeAST Live Data Stream Service")
    print("Live streaming only - No database storage")
    print("="*60 + "\n")
    
    # Initialize components
    ws_server = WebSocketServer(WEBSOCKET_HOST, WEBSOCKET_PORT)
    serial_handler = SerialHandler(SERIAL_PORT, BAUD_RATE)
    
    # Start WebSocket server in background
    ws_task = asyncio.create_task(ws_server.start())
    
    # Wait for WebSocket to initialize
    await asyncio.sleep(1)
    
    # Show command info
    await command_interface(serial_handler)
    
    # Start serial data processing
    serial_task = asyncio.create_task(serial_handler.run(ws_server))
    
    # Wait for tasks
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
