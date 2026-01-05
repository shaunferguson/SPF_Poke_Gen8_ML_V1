# Quick Start Guide

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify team loading:**
```bash
python team_loader.py
```

You should see your 2019 BDSP Battle Tower teams loaded successfully.

## Understanding the Components

### 1. Team Loader (`team_loader.py`)
Manages your 2019 BDSP Battle Tower teams. Already working!

```python
from team_loader import TeamLoader

loader = TeamLoader()
print(f"Loaded {loader.get_team_count()} teams")

# Get a random team
team = loader.get_random_team()
print(f"Team: {team['filename']}")

# Get team info
info = loader.parse_team_info(team)
for pokemon in info['pokemon']:
    print(f"{pokemon['species']} @ {pokemon['item']}")
```

### 2. State Encoder (`state_encoder.py`)
Converts battle states into neural network inputs.

```python
from state_encoder import BattleStateEncoder

encoder = BattleStateEncoder()
print(f"State vector size: {encoder.state_size}")

# Test encoding
state_tensor = encoder.encode_state(battle_state)
action_mask = encoder.get_action_mask(battle_state)
```

### 3. DQN Model (`dqn_model.py`)
Neural network for learning battle strategy.

```python
from dqn_model import DuelingDQN

model = DuelingDQN(state_size=encoder.state_size, action_size=9)
q_values = model(state_tensor)
```

### 4. Battle Agent (`battle_agent.py`)
Coordinates everything to make decisions and learn.

```python
from battle_agent import BattleAgent

agent = BattleAgent()
action = agent.select_action(battle_state)
```

## Next Steps: Getting Battles Working

The framework is complete, but you need a battle engine. Here are your options:

### Option A: Pokemon Showdown Server (Recommended)

**Pros:**
- Official battle engine
- Accurate mechanics
- Can battle against other players/bots

**Steps:**
1. Set up a local Pokemon Showdown server
2. Create a custom format for Gen 8 BDSP (no Dynamax, first 493 Pokemon)
3. Configure the server to accept battles
4. Run training with online mode

**Resources:**
- Showdown server: https://github.com/smogon/pokemon-showdown
- Custom format guide: https://github.com/smogon/pokemon-showdown/blob/master/config/CUSTOM-FORMATS.md

### Option B: poke-env Library

**Pros:**
- Python-based
- Handles Showdown integration
- Good for reinforcement learning

**Steps:**
```bash
pip install poke-env
```

Then adapt the code to use poke-env's environment interface. You'll need to modify:
- `battle_agent.py` to work with poke-env's battle format
- `state_encoder.py` to parse poke-env's state representation
- `train.py` to use poke-env's player classes

**Resources:**
- poke-env docs: https://poke-env.readthedocs.io/
- Examples: https://github.com/hsahovic/poke-env/tree/master/examples

### Option C: Custom Battle Engine

**Pros:**
- Full control
- No server needed

**Cons:**
- Very complex to implement
- Need to code all battle mechanics

This requires implementing:
- Damage calculation formulas
- Status effects
- Abilities
- Type effectiveness
- Weather/terrain
- Stat changes
- Move effects
- And much more...

## Recommended Path

**For fastest results, use poke-env:**

1. Install poke-env:
```bash
pip install poke-env
```

2. Study the poke-env examples, especially the RL ones

3. Create an adapter between your DQN agent and poke-env's player interface

4. Modify `train.py` to use poke-env battles instead of raw WebSocket

**Rough adaptation outline:**

```python
from poke_env.player import RandomPlayer, Player
from poke_env.environment import Battle

class DQNPlayer(Player):
    def __init__(self, battle_agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.battle_agent = battle_agent

    def choose_move(self, battle: Battle):
        # Convert battle to your state format
        battle_state = self._convert_battle_to_state(battle)

        # Get action from agent
        action = self.battle_agent.select_action(battle_state)

        # Convert action to poke-env move
        return self._convert_action_to_move(action, battle)

    def _convert_battle_to_state(self, battle: Battle):
        # Map poke-env Battle object to your state format
        pass

    def _convert_action_to_move(self, action: str, battle: Battle):
        # Map your action string to poke-env's move format
        pass
```

## Testing Without Battles

While setting up the battle engine, you can test components:

### Test Model Training Logic
```python
from battle_agent import BattleAgent

agent = BattleAgent()

# Create dummy state
dummy_state = {
    'team': {'active': {'species': 'Garchomp', 'condition': '100/100'}, 'pokemon': []},
    'opponent': {'active': {'species': 'Charizard', 'condition': '100/100'}, 'pokemon': []},
    'field': {},
    'request': {'active': [{'moves': [
        {'move': 'Earthquake', 'pp': 10, 'maxpp': 10}
    ]}]}
}

# Test action selection
action = agent.select_action(dummy_state)
print(f"Selected: {action}")

# Simulate learning
for i in range(100):
    # Dummy experience
    agent.store_transition(dummy_state, reward=1.0, done=False)
    loss = agent.train_step()
    if loss:
        print(f"Step {i}: Loss = {loss:.4f}")
```

### Analyze Your Teams
```bash
python evaluate.py --mode analyze
```

This shows Pokemon/item usage statistics from your 2019 teams.

## Workflow Summary

1. **Now:** Components are built and team loading works
2. **Next:** Set up battle engine (poke-env recommended)
3. **Then:** Run training battles
4. **Finally:** Evaluate team fitness

## Key Files You'll Edit

When adding battle engine:
- `train.py` - Modify to use your chosen battle system
- `battle_agent.py` - May need adapter for battle format
- `state_encoder.py` - Ensure it parses your battle state correctly

## Training Tips

Once battles work:

1. **Start small:** Train for 100 battles first to verify everything works
2. **Monitor epsilon:** Should decrease from 1.0 to ~0.01 over training
3. **Check rewards:** Should generally increase over time
4. **Save often:** Checkpoints save every 100 battles
5. **Use GPU:** Training is much faster with CUDA

## GPU Setup

If you have an NVIDIA GPU:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

The agent will automatically use GPU if available.

## Questions?

Common issues:
- **"No teams loaded"** - Check `bdsp_BT_teams` directory path
- **"State size mismatch"** - State encoder needs full Pokemon data
- **"No legal actions"** - Action masking may need adjustment
- **"Battles not starting"** - Need to implement/connect battle engine

Happy training!
