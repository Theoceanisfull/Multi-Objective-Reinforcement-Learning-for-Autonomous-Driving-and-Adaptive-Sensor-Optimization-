# Learning algorithms

Start with [the DQN notebook](../experiments/sensor_dqn_experiment.ipynb) for a
step-by-step comparison of Expected SARSA, DQN, and Dyna-DQN. The notebook uses
the same 300 m / 16 s task as the short-route SARSA experiment.

## Files

- `sensor_agent.py`: the original linear Expected SARSA and shared observation encoder.
- `dqn_agent.py`: the small Q network, experience buffer, and Dyna prediction model.
- `../experiments/dqn_helpers.py`: the training loop for either neural agent.

## Read the DQN code in this order

1. **`DQNSettings`** holds the learning choices in one place. `episodes` controls
   practice trips; `learning_rate` controls neural weight updates; `gamma` controls
   the importance of future rewards. Epsilon controls random exploration.
2. **`make_network`** creates an input layer, one hidden layer with ReLU, and an
   output layer. Each output estimates the return for one driving/sensing choice.
3. **`ReplayBuffer`** remembers real steps. Sampling random batches mixes older
   and newer experience. Copying inputs prevents later changes to an observation
   from changing saved experience.
4. **`DQNAgent.__init__`** creates an online network to train and a target network
   to supply future-value estimates. Both start with the same weights. Adam
   adjusts the online weights; the target changes only at copy intervals.
5. **`values`, `select`, and `decode`** estimate action values, choose an action
   using epsilon-greedy exploration, and translate its index into the environment's
   drive command and sensor mask. They use the same action ordering as SARSA.
6. **`td_targets`** calculates `reward + gamma * best_next_Q` using the target
   network. An actual terminal flag removes the future term. Unlike Expected
   SARSA, the next value is a maximum rather than an average under exploration.
7. **`learn_batch`** compares the selected action values with the targets using
   Smooth L1 loss. `backward()` calculates gradients, clipping limits their size,
   and `optimizer.step()` adjusts the online weights. Target estimates have no
   gradient because they are the reference for this update.
8. **`observe`** saves each real step. Once the buffer is large enough, it samples
   a batch and learns once. This ties learning to real environment interaction.
9. **`save` and `load`** store and restore the learned policy for evaluation.
   Training state is not saved, so loading is not an exact way to resume a run.

## Read the Dyna addition

`DynaDQNAgent` inherits the DQN behavior and adds a learned prediction model.
This is a Dyna-style offshoot, not just extra replay updates.

- **`model_predictions`** inputs features and a one-hot action. It predicts the
  feature change, reward, and terminal logit. A sigmoid converts that logit to
  an ending probability.
- **`learn_model`** fits those predictions to real experience. Only this method
  updates the prediction model.
- **`imagine_batch`** starts from stored real feature/action pairs and predicts
  one step. Predicted features are clipped to the encoder's approximate bounds,
  and the constant bias stays at 1. Future Q is weighted by the predicted
  probability of continuing: `reward_hat + gamma * (1-p_end) * best_next_Q`.
- **`observe`** first performs the normal real DQN update, then fits the model.
  After the model warm-up threshold, it performs `planning_steps` extra Q updates
  using predicted outcomes. Imagined data never enters the real replay buffer.

The default is two extra updates, each with a batch of 32 imagined transitions.
The target-copy interval counts both real and imagined Q updates. Planning costs
extra computation and can amplify prediction mistakes. One-step predictions avoid
long chains of model errors, but do not make the predictions accurate.

## The training loop

`train_dqn` resets the environment for each episode, encodes observations, chooses
an action, takes a real step, and calls `observe`. It returns an agent and a pandas
training history. Loss columns are missing until their updates begin. `q_loss`
tracks real replay updates; `planning_loss` tracks imagined updates; `model_loss`
combines the three prediction errors. These losses have different meanings.

Finite-route deadlines stop bootstrapping, just as in the existing SARSA loop.
The original continuing-task time cutoff still bootstraps. Use
`experiments.agent_helpers.evaluate_agents` for held-out evaluation without learning.

## Run it

Install the optional dependency from the project root:

```bash
.venv/bin/python -m pip install -r requirements-dqn.txt
```

```python
from algorithms.dqn_agent import DQNSettings
from environments.sensor_highway import ExperimentConfig
from experiments.dqn_helpers import train_dqn

config = ExperimentConfig()  # Original continuing-driving task.
settings = DQNSettings(episodes=60)
dqn, history = train_dqn(config, settings, seed=42)
dyna, dyna_history = train_dqn(config, settings, seed=42, dyna=True)
```

For the matched finite-route comparison and model saving, use the notebook.
A short run checks the pipeline, not convergence. The shared encoder loses scene
detail, and the prediction model averages uncertain outcomes without recurrent
memory. Compare crashes, route completion, power, and interaction counts across
multiple training seeds before drawing conclusions.

DQN reference: [official PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
