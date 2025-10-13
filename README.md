# Reinforcement Learning for Deliberation in ROS 2

This repository contains materials for the [ROSCon 2025](https://roscon.ros.org/2025/) workshop on ROS 2 Deliberation Technologies.

> [!NOTE]
> This was moved here from <https://github.com/ros-wg-delib/roscon25-workshop>.

## Setup

This repo uses Pixi and RoboStack along with ROS 2 Kilted.

First, install dependencies on your system (assuming you are using Linux).

<!--- new-env: ubuntu:latest --->
<!--
```bash
apt update
apt install -y git curl build-essential
```
-->

<!--- skip-next --->
```bash
sudo apt install build-essential curl
```

Then, install Pixi.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

<!--
This is necessary to make pixi work ...
```bash
echo 'export PATH=\"/root/.pixi/bin:$PATH\"' >> /root/.bashrc
```
-->

Clone the repo including submodules.

```bash
git clone --recursive https://github.com/ros-wg-delib/rl_deliberation.git
```

Build the environment.

<!--- workdir: /rl_deliberation --->
```bash
pixi run build
```

To verify your installation, the following should launch a window of PyRoboSim.

<!--- skip-next --->
```bash
pixi run start_world --env GreenhousePlain
```

To explore the setup, you can also drop into a shell in the Pixi environment.

<!--- skip-next --->
```bash
pixi shell
```

## Explore the environment

There are different environments available. For example, to run the Greenhouse environment:

<!--- skip-next --->
```bash
pixi run start_world --env GreenhousePlain
```

All the following commands assume that the environment is running.
You can also run the environment in headless mode for training.

<!--- skip-next --->
```bash
pixi run start_world --env GreenhousePlain --headless
```

But first, we can explore the environment with a random agent.

## Evaluating with a random agent

Assuming the environment is running, execute the evaluation script in another terminal:

<!--- skip-next --->
```bash
pixi run eval --model pyrobosim_ros_gym/policies/GreenhousePlain_DQN_random.pt --num-episodes 1
```
<!--- workdir: /rl_deliberation --->
<!--
```bash
pixi run start_world --env GreenhousePlain --headless & pid=$!; pixi run eval --model pyrobosim_ros_gym/policies/GreenhousePlain_DQN_random.pt --num-episodes 1; kill $pid
```
-->

In your terminal, you will see multiple sections in the following format:

```plaintext
..........
obs=array([1.        , 0.99194384, 0.        , 2.7288349, 0.        , 3.3768525, 1.0], dtype=float32)
action=array(0)
Maximum steps (10) exceeded. Truncated episode.
reward=0.0
custom_metrics={'watered_plant_fraction': 0.0, 'battery_level': 100.0}
terminated=False
truncated=False
..........
```

This is one step of the environment and the agent's interaction with it.

- `obs` is the observation from the environment. It is an array with information about the 3 closest plant objects, with a class label (0 or 1), the distance to each object. It also has the robot's battery level and whether its current location is watered at the end.
- `action` is the action taken by the agent. In this simple example, it can choose between 0 = move on and 1 = water plant.
- `reward` is the reward received after taking the action, which is `0.0` in this case, because the agent did not water any plant.
- `custom_metrics` provides additional information about the episode:
  - `watered_plant_fraction` indicates the fraction of plants (between 0 and 1) watered thus far in the episode.
  - `battery_level` indicates the current battery level of the robot. (This will not decrease for this environment type, but it will later.)
- `terminated` indicates whether the episode reached a terminal state (e.g., the task was completed or failed).
- `truncated` indicates whether the episode ended due to a time limit.

In the PyRoboSim window, you should also see the robot moving around at every step.

At the end of the episode, and after all episodes are completed, you will see some more statistics printed in the terminal.

```plaintext
..........
<<< Episode 1 finished.
Total reward: 0.0
Mean watered_plant_fraction: 0.0
Mean battery_level: 100.0
====================
Summary:
Reward over 1 episodes:
 Mean: 0.0
 Min: 0.0
 Max: 0.0
Custom metric 'watered_plant_fraction' over 1 episodes:
 Mean: 0.0
 Min: 0.0
 Max: 0.0
Custom metric 'battery_level' over 1 episodes:
 Mean: 100.0
 Min: 100.0
 Max: 100.0
```

## Training a model

While the environment is running (in headless mode if you prefer), you can train a model.

### Choose model type

For example PPO

<!--- skip-next --->
```bash
pixi run train --env GreenhousePlain --config greenhouse_env_config.yaml --model-type PPO --log
```

Or DQN.
Note that this needs the `--discrete-actions` flag.

<!--- skip-next --->
```bash
pixi run train --env GreenhousePlain --config greenhouse_env_config.yaml --model-type DQN --discrete-actions --log
```

Note that at the end of training, the model name and path will be printed in the terminal:

```plaintext
New best mean reward!
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━ 100/100  [ 0:00:35 < 0:00:00 , 2 it/s ]

Saved model to GreenhousePlain_PPO_<timestamp>.pt
```

Remember this path, as you will need it later.

### You may find tensorboard useful

<!--- skip-next --->
```bash
pixi run tensorboard
```

It should contain one entry named after your recent training run (e.g. `GreenhousePlain_PPO_<timestamp>`).

### See your freshly trained policy in action

<!--- skip-next --->
```bash
pixi run eval --model GreenhousePlain_PPO_<timestamp>.pt
```
