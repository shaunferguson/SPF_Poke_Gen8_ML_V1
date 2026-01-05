"""Debug script to test challenging process step by step"""

import asyncio
import websockets

async def debug_challenge():
    """Test the challenge process"""

    # Connect two clients
    print("Connecting clients...")
    ws1 = await websockets.connect("ws://127.0.0.1:8000/showdown/websocket")
    ws2 = await websockets.connect("ws://127.0.0.1:8000/showdown/websocket")

    # Receive initial messages
    msg1 = await ws1.recv()
    msg2 = await ws2.recv()

    # Extract usernames
    username1 = None
    username2 = None

    for line in msg1.split('\n'):
        if line.startswith('|updateuser|'):
            username1 = line.split('|')[2].strip()

    for line in msg2.split('\n'):
        if line.startswith('|updateuser|'):
            username2 = line.split('|')[2].strip()

    print(f"Client 1: {username1}")
    print(f"Client 2: {username2}")

    # Get challstr for proper login
    challstr1 = None
    challstr2 = None

    for line in msg1.split('\n'):
        if line.startswith('|challstr|'):
            challstr1 = line.split('|challstr|')[1]

    for line in msg2.split('\n'):
        if line.startswith('|challstr|'):
            challstr2 = line.split('|challstr|')[1]

    # Try to set custom usernames using /trn
    print(f"\nTrying to set usernames with /trn...")
    print(f"Challstr1: {challstr1[:50]}...")
    print(f"Challstr2: {challstr2[:50]}...")

    await ws1.send(f"|/trn Bot1,0,{challstr1}")
    await asyncio.sleep(0.5)

    await ws2.send(f"|/trn Bot2,0,{challstr2}")
    await asyncio.sleep(0.5)

    # Check responses
    try:
        while True:
            msg = await asyncio.wait_for(ws1.recv(), timeout=0.5)
            print(f"Bot1: {msg[:200]}")
    except:
        pass

    try:
        while True:
            msg = await asyncio.wait_for(ws2.recv(), timeout=0.5)
            print(f"Bot2: {msg[:200]}")
    except:
        pass

    # Try setting team with /utm
    print("\nTrying to set team...")
    simple_team = "Garchomp||ChoiceScarf|RoughSkin|Earthquake,DragonClaw,RockSlide,FireFang|Jolly|,252,,,4,252|||||100"

    await ws1.send(f"|/utm {simple_team}")
    await asyncio.sleep(0.5)

    # Try challenge
    print(f"\nTrying to challenge {username2}...")
    await ws1.send(f"|/challenge {username2}, gen8ou")
    await asyncio.sleep(1)

    # Check responses
    print("\nChecking responses...")
    try:
        while True:
            msg = await asyncio.wait_for(ws1.recv(), timeout=0.5)
            print(f"Bot1: {msg[:300]}")
    except:
        pass

    try:
        while True:
            msg = await asyncio.wait_for(ws2.recv(), timeout=0.5)
            print(f"Bot2: {msg[:300]}")
    except:
        pass

    await ws1.close()
    await ws2.close()

if __name__ == "__main__":
    asyncio.run(debug_challenge())
