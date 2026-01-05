<<<<<<< HEAD
# Pokemon Battle AI - DQN Agent

A Deep Q-Network (DQN) based AI agent for playing Pokemon battles, specifically designed to emulate Gen 8 mechanics (BDSP - Brilliant Diamond/Shining Pearl) with the first 493 Pokemon.

## Project Structure

```
Showdown_ML/
├── showdown_client.py      # WebSocket client for Pokemon Showdown server
├── state_encoder.py         # Encodes battle state into neural network inputs
├── dqn_model.py            # DQN model architecture and training components
├── battle_agent.py         # Battle agent that coordinates model and decisions
├── team_loader.py          # Utility to load BDSP Battle Tower teams
├── train.py                # Main training script
├── requirements.txt        # Python dependencies
├── bdsp_BT_teams/         # Directory with Battle Tower team files
├── checkpoints/           # Saved model checkpoints (created during training)
└── logs/                  # Training logs (created during training)
```

## Features

- **DQN Architecture**: Implements Deep Q-Network with Double DQN and optional Dueling DQN
- **Prioritized Experience Replay**: Improved learning from important experiences
- **State Encoding**: Comprehensive encoding of battle state including:
  - Pokemon HP, types, status conditions, stat stages
  - Team composition and health
  - Field conditions (weather, hazards)
  - Available moves and their properties
- **Action Masking**: Only selects legal actions
- **Modular Design**: Separate modules for different concerns
- **BDSP Teams**: Uses authentic Battle Tower teams from BDSP

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Agent

Currently, the project supports two training modes:

#### Self-Play Mode (Framework Ready)
```bash
python train.py --mode selfplay --battles 1000 --teams-dir bdsp_BT_teams
```

Note: Self-play requires implementing a battle simulator or using a library like `poke-env`.

#### Online Mode (Requires Showdown Server)
```bash
python train.py --mode online --battles 1000 --username YourBotName --teams-dir bdsp_BT_teams
```

### Loading from Checkpoint
```bash
python train.py --checkpoint checkpoints/agent_latest.pt --battles 1000
```

### Testing Individual Components

Test the team loader:
```bash
python team_loader.py
```

Test the state encoder:
```bash
python state_encoder.py
```

Test the DQN model:
```bash
python dqn_model.py
```

Test the battle agent:
```bash
python battle_agent.py
```

## How It Works

### 1. State Encoding
The `BattleStateEncoder` converts the battle state into a fixed-size tensor:
- Active Pokemon information (HP, types, status, stat stages)
- Team information (HP and faint status of all Pokemon)
- Field conditions (weather, terrain, hazards)
- Available moves (PP, base power, type)

### 2. DQN Model
The agent uses a neural network to estimate Q-values for each possible action:
- **Standard DQN**: Simple feedforward network
- **Dueling DQN**: Separates state value and action advantages for better learning

### 3. Action Selection
- **Training**: Epsilon-greedy policy (explores random actions initially, gradually exploits learned policy)
- **Evaluation**: Greedy policy (always selects best action)
- **Action Masking**: Only legal actions are considered

### 4. Experience Replay
Stores past experiences and samples random batches for training:
- **Standard Replay**: Uniform random sampling
- **Prioritized Replay**: Samples important experiences more frequently

### 5. Reward System
The agent receives rewards based on:
- Dealing damage to opponent (+)
- Taking damage (-)
- Knocking out opponent's Pokemon (++)
- Losing own Pokemon (--)
- Winning battle (+++)
- Losing battle (---)

## Next Steps

To make this fully functional, you'll need to implement or integrate a battle engine. Options include:

### Option 1: Local Showdown Server
1. Set up a local Pokemon Showdown server
2. Create a custom format for Gen 8 without Dynamax, limited to first 493 Pokemon
3. Run battles between your agent and:
   - Random player
   - Another agent instance (self-play)
   - Rule-based AI

### Option 2: Use poke-env
The `poke-env` library provides a Python interface to Pokemon Showdown and includes battle simulation:

```bash
pip install poke-env
```

This would require adapting the current code to use poke-env's environment interface.

### Option 3: Implement Battle Engine
Create a custom battle simulator that:
- Parses Pokemon data (stats, moves, abilities)
- Implements battle mechanics (damage calculation, status effects, etc.)
- Enforces Gen 8 rules without Dynamax
- Updates battle state each turn

## Battle Format Specifications

- **Generation**: Gen 8 mechanics (BDSP)
- **Pokemon Pool**: First 493 Pokemon only (National Dex #1-493)
- **No Dynamax/Gigantamax**
- **Teams**: 4-6 Pokemon
- **Format**: Singles or Doubles (configurable)

## Team Fitness Evaluation

Once the basic agent is working, you can:

1. Load multiple teams from `bdsp_BT_teams/`
2. Have agents battle with different teams
3. Track win rates for each team
4. Rank teams by relative fitness
5. Use evolutionary algorithms to optimize teams

Example workflow:
```python
from team_loader import TeamLoader
from battle_agent import BattleAgent

loader = TeamLoader()
agent = BattleAgent()
agent.load("checkpoints/agent_latest.pt")

# Evaluate each team
team_fitness = {}
for team in loader.get_all_teams():
    # Run N battles with this team
    # Track win rate
    # Store in team_fitness
    pass
```

## Model Improvements

Future enhancements:
- [ ] Add Pokemon species embeddings
- [ ] Implement move type effectiveness calculations
- [ ] Add recurrent layers (LSTM/GRU) for sequential decision making
- [ ] Implement Monte Carlo Tree Search (MCTS) for move planning
- [ ] Add opponent modeling
- [ ] Implement curriculum learning (start with simple battles)
- [ ] Multi-agent training with population diversity

## Known Limitations

1. **No Battle Engine**: Currently requires external battle simulation
2. **Simplified State Encoding**: Some battle mechanics not fully represented
3. **Placeholder Data**: Pokemon types, move data, etc. need full implementation
4. **No Team Validation**: Doesn't check if teams are legal for the format

## Contributing

To extend this project:
1. Implement missing Pokemon data (types, stats, moves, abilities)
2. Add battle engine integration
3. Improve reward function
4. Add more sophisticated state features
5. Implement team building algorithms

## References

- Pokemon Showdown: https://pokemonshowdown.com/
- DQN Paper: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- Double DQN: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
- Dueling DQN: "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
=======
# SPF_Poke_Gen8_ML_V1
ML pokemon agent
>>>>>>> f4d759e980bcb24630e7ba6975296e42bbdfa1b1
