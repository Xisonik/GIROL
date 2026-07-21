![Isaac Lab](docs/source/_static/pipeline_main.jpg)


## Getting Started
The installation process fully complies with the official Isaac Lab documentation, with the exception that you need to clone the current pipeline, and not from the official Isaac Lab repository.

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)

The launch is carried out in accordance with the official documentation. The name of the environment - Isaac-Aloha-Direct-v0
The SAC algorithm of the skrl library is used here.
Install requirements:
```
pip install -r requirements.txt
```
or
```
conda env create -f environment.yml
```

download assets:
```
./download_girol_assets.sh
```
Train navigation:
```
./isaaclab.sh -p scripts/algos/run_experiments.py
```

### Configure checkpoints

Use `scripts/algos/set_checkpoint_paths.py` to automatically update the checkpoint paths in an experiment configuration. The script fills:

* `paths.agent_checkpoint`;
* `paths.state_preprocessor_checkpoint`;
* `paths.aux_checkpoint`.

It reads `run.task_name` and `run.name` from the selected configuration, so only the configuration name and training run folder are required.

Run from the repository root:

```bash
python scripts/algos/set_checkpoint_paths.py \
  cur_dqn/ddqn_discrete.json \
  07.21_16-26-19_cur_dqn
```

This resolves the checkpoint directory as:

```text
logs/skrl/Aloha_nav_hab_wr/07.21_16-26-19_cur_dqn/ddqn_discrete
```

To select checkpoints from a specific training step:

```bash
python scripts/algos/set_checkpoint_paths.py \
  cur_dqn/ddqn_discrete.json \
  07.21_16-26-19_cur_dqn \
  1000
```

To override the complete path relative to `logs/skrl`:

```bash
python scripts/algos/set_checkpoint_paths.py \
  cur_dqn/ddqn_discrete.json \
  --p Aloha_nav_hab_wr/07.21_16-26-19_cur_dqn/ddqn_discrete
```

To reset all checkpoint paths to `null`:

```bash
python scripts/algos/set_checkpoint_paths.py \
  cur_dqn/ddqn_discrete.json
```

When no training step is specified, the newest checkpoint files are selected automatically. If the agent, state preprocessor, or auxiliary checkpoint cannot be found, the script exits with an error and does not modify the configuration.


Imitation learning:
To generate paths via dijkstra algo (Check, that there no all_paths.json in data):
```
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/aloha/path_generator.py 
```