"""
Pokemon Showdown WebSocket Client
Handles connection and communication with Pokemon Showdown server
"""

import asyncio
import websockets
import requests
import json
import re
from typing import Optional, Callable, Dict, Any


class ShowdownClient:
    """Client for interacting with Pokemon Showdown server"""

    def __init__(self, username: str, password: Optional[str] = None,
                 server_url: str = "ws://sim3.psim.us:8000/showdown/websocket"):
        """
        Initialize the Showdown client

        Args:
            username: Username for the bot
            password: Password (optional, for registered accounts)
            server_url: WebSocket URL of the Showdown server
        """
        self.username = username
        self.password = password
        self.server_url = server_url
        self.websocket = None
        self.logged_in = False
        self.battle_callback: Optional[Callable] = None
        self.current_battles: Dict[str, Dict[str, Any]] = {}

    async def connect(self):
        """Establish WebSocket connection to the server"""
        self.websocket = await websockets.connect(self.server_url)
        print(f"Connected to {self.server_url}")

    async def login(self):
        """Login to the server"""
        # Wait for initial server message
        initial = await self.websocket.recv()

        # Check if already logged in (local server auto-login)
        if '|updateuser|' in initial:
            # Local server has already logged us in
            self.logged_in = True
            # Extract username from updateuser message
            for line in initial.split('\n'):
                if line.startswith('|updateuser|'):
                    parts = line.split('|')
                    if len(parts) > 2:
                        actual_username = parts[2].strip()
                        # Update our username to what the server assigned
                        self.username = actual_username
                        print(f"Logged in as {actual_username} (local server auto-login)")
                        return

        # Extract challenge string for remote servers
        challstr = None
        for line in initial.split('\n'):
            if line.startswith('|challstr|'):
                challstr = line.split('|challstr|')[1]
                break

        if not challstr:
            # If no challstr and no updateuser, something is wrong
            if not self.logged_in:
                print(f"\n{'='*60}")
                print("WARNING: Unexpected server response")
                print('='*60)
                print(f"Server response (first 500 chars):\n{initial[:500]}")
                print('='*60)
                # Try to continue anyway - local server might work differently
                self.logged_in = True
                print(f"Attempting to continue as {self.username}...")
                return

        # Login process for remote servers
        if self.password:
            # Login with registered account
            login_url = "https://play.pokemonshowdown.com/api/login"
            data = {
                'name': self.username,
                'pass': self.password,
                'challstr': challstr
            }
            response = requests.post(login_url, data=data)
            assertion = json.loads(response.text[1:])['assertion']

            await self.websocket.send(f"|/trn {self.username},0,{assertion}")
        else:
            # Login as guest
            await self.websocket.send(f"|/trn {self.username},0,{challstr}")

        self.logged_in = True
        print(f"Logged in as {self.username}")

    async def challenge_user(self, opponent: str, team: str, battle_format: str = "gen8nationaldexdraft"):
        """
        Challenge another user to a battle

        Args:
            opponent: Username to challenge
            team: Team in packed format
            battle_format: Battle format (default: gen8nationaldexdraft for Gen 8 mechanics)
        """
        await self.websocket.send(f"|/utm {team}")
        await self.websocket.send(f"|/challenge {opponent}, {battle_format}")

    async def accept_challenge(self, opponent: str, team: str):
        """Accept a challenge from another user"""
        await self.websocket.send(f"|/utm {team}")
        await self.websocket.send(f"|/accept {opponent}")

    async def search_battle(self, team: str, battle_format: str = "gen8nationaldexdraft"):
        """
        Search for a random battle

        Args:
            team: Team in packed format
            battle_format: Battle format
        """
        await self.websocket.send(f"|/utm {team}")
        await self.websocket.send(f"|/search {battle_format}")

    async def cancel_search(self):
        """Cancel battle search"""
        await self.websocket.send("|/cancelsearch")

    async def make_move(self, battle_id: str, choice: str):
        """
        Make a move in battle

        Args:
            battle_id: ID of the battle room
            choice: Choice string (e.g., "move 1", "switch 2")
        """
        await self.websocket.send(f"{battle_id}|/choose {choice}")

    async def forfeit(self, battle_id: str):
        """Forfeit a battle"""
        await self.websocket.send(f"{battle_id}|/forfeit")

    def parse_battle_message(self, room_id: str, message: str) -> Dict[str, Any]:
        """
        Parse battle state from server message

        Args:
            room_id: Battle room ID
            message: Raw message from server

        Returns:
            Parsed battle state dictionary
        """
        if room_id not in self.current_battles:
            self.current_battles[room_id] = {
                'active': True,
                'player_id': None,
                'team': {'active': None, 'pokemon': []},
                'opponent': {'active': None, 'pokemon': []},
                'field': {},
                'request': None,
                'winner': None
            }

        battle_state = self.current_battles[room_id]

        for line in message.split('\n'):
            if not line.startswith('|'):
                continue

            parts = line.split('|')[1:]
            if not parts:
                continue

            msg_type = parts[0]

            # Determine player ID
            if msg_type == 'player' and len(parts) >= 3:
                if parts[2] == self.username:
                    battle_state['player_id'] = parts[1]

            # Track pokemon
            elif msg_type == 'poke' and len(parts) >= 3:
                player = parts[1]
                species = parts[2].split(',')[0]

                if battle_state['player_id'] and player == battle_state['player_id']:
                    battle_state['team']['pokemon'].append({
                        'species': species,
                        'condition': '100/100',
                        'active': False,
                        'stats': {},
                        'moves': []
                    })
                else:
                    battle_state['opponent']['pokemon'].append({
                        'species': species,
                        'condition': '100/100',
                        'active': False,
                        'stats': {},
                        'moves': []
                    })

            # Switch pokemon
            elif msg_type in ['switch', 'drag'] and len(parts) >= 3:
                player_pokemon = parts[1]
                species = parts[2].split(',')[0]
                condition = parts[3] if len(parts) > 3 else '100/100'

                is_player = battle_state['player_id'] and player_pokemon.startswith(battle_state['player_id'])

                if is_player:
                    battle_state['team']['active'] = {
                        'species': species,
                        'condition': condition
                    }
                else:
                    battle_state['opponent']['active'] = {
                        'species': species,
                        'condition': condition
                    }

            # Update HP
            elif msg_type == '-damage' and len(parts) >= 3:
                player_pokemon = parts[1]
                condition = parts[2]

                is_player = battle_state['player_id'] and player_pokemon.startswith(battle_state['player_id'])

                if is_player and battle_state['team']['active']:
                    battle_state['team']['active']['condition'] = condition
                elif battle_state['opponent']['active']:
                    battle_state['opponent']['active']['condition'] = condition

            # Track moves
            elif msg_type == 'move' and len(parts) >= 3:
                player_pokemon = parts[1]
                move = parts[2]

                is_player = battle_state['player_id'] and player_pokemon.startswith(battle_state['player_id'])

                if not is_player and battle_state['opponent']['active']:
                    if 'moves' not in battle_state['opponent']['active']:
                        battle_state['opponent']['active']['moves'] = []
                    if move not in battle_state['opponent']['active']['moves']:
                        battle_state['opponent']['active']['moves'].append(move)

            # Request (decision prompt)
            elif msg_type == 'request' and len(parts) >= 2:
                request_data = parts[1]
                if request_data:
                    try:
                        battle_state['request'] = json.loads(request_data)
                    except json.JSONDecodeError:
                        pass

            # Win condition
            elif msg_type == 'win' and len(parts) >= 2:
                battle_state['winner'] = parts[1]
                battle_state['active'] = False

        return battle_state

    async def listen(self, battle_callback: Optional[Callable] = None):
        """
        Listen for messages from server

        Args:
            battle_callback: Async callback function called with (battle_id, state) when action needed
        """
        self.battle_callback = battle_callback

        try:
            while True:
                message = await self.websocket.recv()

                # Determine room
                room_id = None
                if message.startswith('>'):
                    room_id = message.split('\n')[0][1:]
                    message = '\n'.join(message.split('\n')[1:])

                # Parse battle messages
                if room_id and room_id.startswith('battle-'):
                    battle_state = self.parse_battle_message(room_id, message)

                    # If we have a request and a callback, invoke it
                    if battle_state.get('request') and self.battle_callback:
                        await self.battle_callback(room_id, battle_state)

                # Debug: print non-battle messages
                elif not room_id or not room_id.startswith('battle-'):
                    print(f"Server: {message[:200]}")

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")

    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            print("Disconnected from server")


def pack_team(team_text: str) -> str:
    """
    Convert Showdown team format to packed format

    Args:
        team_text: Team in Showdown export format

    Returns:
        Packed team string
    """
    # This is a simplified version - full implementation would need complete parsing
    # For now, we'll use the team as-is and rely on the server's team validation
    # In production, you'd want a proper team parser
    return team_text


async def main():
    """Example usage"""
    client = ShowdownClient(username="TestBot123")

    await client.connect()
    await client.login()

    # Example: search for a battle
    team = "Sample team here"  # Would be packed team format
    # await client.search_battle(team)

    # Listen for battles
    async def handle_battle(battle_id, state):
        print(f"Battle {battle_id} - Request: {state.get('request')}")
        # Make a random move for testing
        if state.get('request'):
            await client.make_move(battle_id, "move 1")

    await client.listen(battle_callback=handle_battle)


if __name__ == "__main__":
    asyncio.run(main())
