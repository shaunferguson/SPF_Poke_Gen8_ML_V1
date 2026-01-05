# Local Pokemon Showdown Server Guide

## Overview

You now have a local Pokemon Showdown server configured for BDSP Battle AI training!

**Location:** `showdown-server/` directory
**Server URL:** `ws://127.0.0.1:8000/showdown/websocket`
**Web Interface:** `http://127.0.0.1:8000`

## Quick Start

### 1. Start the Server

**Windows:**
```bash
start_server.bat
```

**Linux/Mac:**
```bash
chmod +x start_server.sh
./start_server.sh
```

**Manual:**
```bash
cd showdown-server
node pokemon-showdown start
```

### 2. Verify Server is Running

Open your browser and go to:
```
http://127.0.0.1:8000
```

You should see the Pokemon Showdown interface.

### 3. Train Your Agent

In a **new terminal** (keep the server running):

```bash
python train_local.py --battles 100
```

## Custom BDSP Formats

Three custom formats are configured:

### 1. [Gen 8] BDSP Singles
- Singles battles
- First 493 Pokemon only (Gen 1-4)
- No Dynamax
- Standard competitive rules

### 2. [Gen 8] BDSP Doubles
- Doubles battles
- First 493 Pokemon only
- No Dynamax
- Standard doubles rules

### 3. [Gen 8] BDSP Battle Tower
- Doubles format
- **Exactly 4 Pokemon teams** (matches your team files!)
- First 493 Pokemon only
- No Dynamax
- Perfect for your Battle Tower teams

## Training Options

### Basic Training
```bash
python train_local.py --battles 1000
```

### Resume from Checkpoint
```bash
python train_local.py --battles 1000 --checkpoint checkpoints/agent_latest.pt
```

### Custom Bot Names
```bash
python train_local.py --battles 100 --bot-name MyBot --opponent-name RandomBot
```

### Specify Device (GPU)
```bash
python train_local.py --battles 1000 --device cuda
```

## Server Configuration

The server is configured in `showdown-server/config/config.js`:

- **Port:** 8000
- **Address:** 127.0.0.1 (localhost only)
- **Processes:** 0 (runs in main process, simpler for training)
- **Guest accounts:** Enabled (bots don't need registration)
- **Ladders:** Disabled (training only)
- **Chat logging:** Disabled

## Custom Format Configuration

The BDSP formats are defined in `showdown-server/config/formats.js`:

Key features:
- Bans all Pokemon with National Dex > 493
- Disables Dynamax
- Team validation for correct Pokemon pool
- Battle Tower format enforces exactly 4 Pokemon

## Troubleshooting

### Server won't start

**Error:** `Port 8000 already in use`
- Another instance is running
- Kill it: `taskkill /F /IM node.exe` (Windows)
- Or change port in `config.js`

**Error:** `Cannot find module`
- Re-run: `cd showdown-server && npm install`

### Training won't connect

**Issue:** Bot can't connect to server
- Make sure server is running first
- Check server URL: `ws://127.0.0.1:8000/showdown/websocket`
- Try browser connection first: `http://127.0.0.1:8000`

**Issue:** Battles not starting
- Make sure both bots are connected
- Check server output for errors
- Try manual battle through web interface

### Format not available

**Issue:** BDSP formats not showing
- Make sure `formats.js` is in `showdown-server/config/`
- Restart the server
- Check server logs for format loading errors

## Architecture

```
Training Setup:
┌─────────────────────────────────────────────────┐
│  Your Python Training Script (train_local.py)  │
│                                                 │
│  ┌──────────────┐      ┌──────────────┐       │
│  │ TrainingBot  │      │ OpponentBot  │       │
│  │  (DQN Agent) │      │   (Random)   │       │
│  └──────┬───────┘      └──────┬───────┘       │
│         │                      │                │
│         └──────────┬───────────┘                │
│                    │                            │
└────────────────────┼────────────────────────────┘
                     │ WebSocket
                     │
┌────────────────────▼────────────────────────────┐
│     Local Pokemon Showdown Server               │
│                                                 │
│  - Runs battles                                 │
│  - Enforces BDSP rules                          │
│  - Validates teams                              │
│  - Tracks game state                            │
└─────────────────────────────────────────────────┘
```

## Workflow

1. **Start server** in one terminal
2. **Start training** in another terminal
3. **Training bot** challenges **Opponent bot**
4. **Server** runs the battle
5. **Bots** make moves
6. **Agent** learns from results
7. Repeat!

## Advanced Usage

### Manual Testing

You can test the server manually:

1. Start the server
2. Open browser: `http://127.0.0.1:8000`
3. Log in as guest
4. Go to teambuilder
5. Import one of your teams from `bdsp_BT_teams/`
6. Challenge another user or use `/challenge username, format`

### Multiple Training Sessions

You can run multiple training sessions simultaneously:

**Terminal 1:**
```bash
python train_local.py --bot-name Bot1 --opponent-name Opponent1 --battles 500
```

**Terminal 2:**
```bash
python train_local.py --bot-name Bot2 --opponent-name Opponent2 --battles 500
```

### Custom Opponent AI

Edit `train_local.py` to implement different opponent strategies:

```python
async def opponent_battle_callback(self, battle_id: str, battle_state: dict):
    # Implement your custom strategy here
    # Options:
    # - Random (current)
    # - Greedy (always best immediate move)
    # - Loaded agent (agent vs agent)
    # - Rule-based (type advantages, etc.)
```

## Performance Tips

### For Faster Training

1. **Use GPU:** `--device cuda`
2. **Reduce logging:** Comment out print statements in `train_local.py`
3. **Batch battles:** Run multiple bots simultaneously
4. **Increase server capacity:** In `config.js`, increase `maxbattlesperprocess`

### For Better Learning

1. **Vary opponents:** Implement different AI strategies
2. **Curriculum learning:** Start with simple opponents, increase difficulty
3. **Team diversity:** Ensure both bots use different teams each battle
4. **Reward tuning:** Adjust reward function in `battle_agent.py`

## Monitoring Training

### Real-time Stats

Training prints:
- Battle number
- Win/Loss record
- Win rate
- Training loss
- Epsilon (exploration rate)
- Average reward

### Log Files

Saved to `logs/`:
- `training_log_<timestamp>.json` - Full stats every 50 battles
- `training_history.csv` - CSV for easy plotting

### Checkpoints

Saved to `checkpoints/`:
- `agent_battle_<N>_<timestamp>.pt` - Every 50 battles
- `agent_latest.pt` - Always the latest

## Next Steps

Once training works:

1. **Train for 1000+ battles** to see real learning
2. **Evaluate teams** using `evaluate.py`
3. **Compare agents** from different checkpoints
4. **Implement better opponents** (rule-based, or load another agent)
5. **Tune hyperparameters** (learning rate, epsilon decay, reward weights)

## Server Commands

While server is running, you can use admin commands.

**Restart server:**
- Ctrl+C to stop
- Run start script again

**Clean battles:**
```bash
cd showdown-server
rm -rf logs/*
```

## Useful URLs

When server is running:

- **Main page:** http://127.0.0.1:8000
- **Teambuilder:** http://127.0.0.1:8000/teambuilder
- **Replays:** http://127.0.0.1:8000/replays (if enabled)
- **Ladder:** http://127.0.0.1:8000/ladder (disabled by default)

## Summary

You're all set! You have:

- ✅ Local Pokemon Showdown server running
- ✅ Custom BDSP formats (Gen 1-4 only, no Dynamax)
- ✅ Battle Tower format (4 Pokemon teams)
- ✅ Training script ready to use
- ✅ 2019 teams loaded and ready

Just run:
1. `start_server.bat` (in one terminal)
2. `python train_local.py --battles 100` (in another terminal)

Happy training! 🚀
