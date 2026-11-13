# Matched simulation replays

Open `replay.html` in a browser for synchronized play/pause, speed control, and frame stepping.
Individual GIFs and decision CSVs are beside it. All assets in the HTML are embedded.

- Source: `dqn_long_training_20261112_170403_004536`; frozen policies trained for 1,000 episodes.
- Traffic seed: **100000**; weather: **clear**.
- Frames: one per decision, every 0.5 s. Shorter runs hold their final frame.
- Green car: ego agent. Road view: simulator truth. Map: sensor evidence, not ground truth.
- Actions label the interval ending at that frame. Reset begins with no measurements.
- GIFs hold the final outcome for two seconds before looping.
- This scenario is illustrative, not a representative estimate of policy performance.

Regenerate another scenario from the project root:

```bash
.venv/bin/python -m experiments.replay_agents --seed 100011 --weather clear
```
