"""
Fixed Pokemon Showdown WebSocket Client for local self-play
"""

import asyncio
import websockets
import random
from typing import Optional, Callable, Dict, Any

def pack_team(team_text: str) -> str:
    """Pack team into single-line format the server expects"""
    packed_pokemon = []
    for block in [b.strip() for b in team_text.split('\n\n') if b.strip()]:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue

        header = lines[0]
        moves = []
        item = ''
        species = header.split(' @ ')[0].strip()
        if ' @ ' in header:
            item = header.split(' @ ')[1].strip()

        # Grab moves
        for line in lines[1:]:
            if line.startswith('- '):
                moves.append(line[2:].strip())

        moves_str = ','.join(moves[:4])
        packed = f"]{species}|{item}|||{moves_str}||"
        packed_pokemon.append(packed)

    return '|'.join(packed_pokemon)


class ShowdownClient:
    def __init__(self, server_url: str = "ws://127.0.0.1:8000/showdown/websocket"):
        self.server_url = server_url
        self.websocket = None
        self.username = "Guest"
        self.battle_callback: Optional[Callable] = None

    async def connect(self):
        print(f"[CONNECT] Trying {self.server_url}...")
        self.websocket = await websockets.connect(self.server_url)
        print("[SUCCESS] Connected!")

    async def login(self):
        print("[LOGIN] Waiting for auto-login...")
        initial = await self.websocket.recv()
        for line in initial.split('\n'):
            if line.startswith('|updateuser|'):
                parts = line.split('|')
                self.username = parts[2].strip()
                print(f"[SUCCESS] Logged in as {self.username}")
                return
        print("[INFO] No auto-login found (remote server?)")

    async def send_command(self, cmd: str):
        print(f"[SEND] {cmd}")
        await self.websocket.send('|' + cmd)

    async def make_move(self, battle_id: str, choice: str):
        print(f"[MOVE] {battle_id}: {choice}")
        await self.websocket.send(f"{battle_id}|/choose {choice}")

    async def listen(self, callback: Callable):
        self.battle_callback = callback
        print("[LISTEN] Started")
        while True:
            msg = await self.websocket.recv()
            if msg.startswith('>'):
                room = msg.split('\n')[0][1:]
                content = '\n'.join(msg.split('\n')[1:])
                if room.startswith('battle-') and self.battle_callback:
                    # Simplified state - expand with your full parser later
                    state = {'room': room, 'raw': content}
                    if 'request' in content or 'winner' in content:
                        await self.battle_callback(room, state)
            else:
                print(f"Server: {msg[:200]}")