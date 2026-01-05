"""Test connection to local Showdown server"""

import asyncio
import websockets

async def test_connection():
    url = "ws://127.0.0.1:8000/showdown/websocket"

    print(f"Connecting to {url}...")

    try:
        websocket = await websockets.connect(url)
        print("[CONNECTED]")

        print("\nWaiting for initial message...")
        initial = await websocket.recv()

        print(f"\nReceived {len(initial)} bytes")
        print("="*60)
        print("Parsing lines (looking for challstr):")
        print("="*60)

        challstr_found = False
        for i, line in enumerate(initial.split('\n')):
            # Only print lines with challstr or first 10 lines
            if 'challstr' in line.lower() or i < 10:
                # Use repr to avoid unicode issues
                print(f"Line {i}: {repr(line[:100])}")
            if 'challstr' in line.lower():
                print(f"  ^ CHALLSTR LINE FOUND!")
                challstr_found = True
                # Try to extract it
                if line.startswith('|challstr|'):
                    challstr = line.split('|challstr|')[1]
                    print(f"  Extracted challstr: {challstr}")

        if not challstr_found:
            print("\nNO CHALLSTR FOUND!")
            print("Showing all lines:")
            for i, line in enumerate(initial.split('\n')):
                print(f"Line {i}: {repr(line[:80])}")

        await websocket.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())
