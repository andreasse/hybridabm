#!/usr/bin/env python3
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/api/v1/runs/ws/exp3_20250622_104141_f96789d6"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            # Send chunk request
            request = {
                "action": "requestFrames",
                "start": 1,
                "end": 5,
                "requestId": "test-1-5"
            }
            await websocket.send(json.dumps(request))
            print(f"Sent chunk request: {request}")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)
                print(f"Received response type: {data.get('type')}")
                if data.get('type') == 'chunk_response':
                    print(f"Frames received: {len(data.get('frames', []))}")
                    print(f"Request ID: {data.get('requestId')}")
                else:
                    print(f"Unexpected response: {data}")
            except asyncio.TimeoutError:
                print("TIMEOUT: No response received within 10 seconds")
                
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())