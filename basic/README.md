# HighwayEnv and tabular Q-learning

Open `highway_q_learning_tutorial.ipynb` and run its cells in order. The notebook
contains the environment, state encoder, Q-learning implementation, evaluation,
animation, and suggested experiments. No RL framework is needed.

A project-local `.venv` has been prepared. From the **project root**:

```bash
source .venv/bin/activate
python -m jupyterlab basic/highway_q_learning_tutorial.ipynb
```

Select **Python (MORL4AVSO)** if Jupyter asks for a kernel.
In VS Code, select the project's `.venv/bin/python` interpreter for the notebook.

To recreate the setup with Python 3.14 (the version used here):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r basic/requirements.txt
python -m ipykernel install --user --name morl4avso --display-name "Python (MORL4AVSO)"
python -m jupyterlab basic/highway_q_learning_tutorial.ipynb
```

The default run trains for 200 episodes and evaluates on 30 separate traffic
seeds. Set `TRAIN_EPISODES = 20` and `EVAL_EPISODES = 5` for a quick first pass;
short runs only demonstrate the workflow. Runtime depends on your computer.
Training does not render; the preview and animation use inline RGB images.

Run **Restart Kernel and Run All Cells** after changing the environment, bins,
or training settings. The save cell writes a Q-table, metadata, and CSV metrics
to `runs/tutorial/highway_q_learning/` under the repository root. Existing
files with those names are replaced when that cell is rerun.

This is the driving-only learning stage of the proposal. Sensor simulation,
energy rewards, and the proposed nonterminating collision design are discussed
as next steps in the notebook; they are not part of this baseline.
