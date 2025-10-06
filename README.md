# Reinforcement Learning for Deliberation in ROS 2

This repository contains materials for the [ROSCon 2025](https://roscon.ros.org/2025/) workshop on ROS 2 Deliberation Technologies.

> [!NOTE]
> This was moved here from https://github.com/ros-wg-delib/roscon25-workshop.

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
pixi run start_world --env Banana
```

To explore the setup, you can also drop into a shell in the Pixi environment.

<!--- skip-next --->
```bash
pixi shell
```

## Explore the environment

There are different environments available. For example, to run the Banana environment:

<!--- skip-next --->
```bash
pixi run start_world --env Banana
```

All the following commands assume that the environment is running. You can also run the environment in headless mode for training.

```bash
pixi run start_world --env Banana --headless
```

But first, we can explore the environment with a random agent.

## Evaluating with a random agent

Assuming the environment is running, execute the evaluation script in another terminal:

```bash
pixi run eval --model pyrobosim_ros_gym/policies/BananaPick_DQN_random.pt --num-episodes 1
```

In your terminal, you will see multiple sections in the following format:

```plaintext
..........
obs=array([ 1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  1.,
        1.,  1., -1., -1., -1.,  1., -1.,  1., -1., -1.,  1.],
      dtype=float32)
action=array(8)
reward=-2.0
terminated=False
truncated=False
..........
```

This is one step of the environment and the agent's interaction with it.

- `obs` is the observation from the environment, which is a vector of 24 floats representing the state of the agent and its surroundings.
- `action` is the action taken by the agent, which is `8` in this case, corresponding to placing an object.
- `reward` is the reward received after taking the action, which is `-2.0` here, a penalty because the robot did not hold any object.
- `terminated` indicates whether the episode reached a terminal state (e.g., the task was completed or failed).
- `truncated` indicates whether the episode ended due to a time limit.

## Training a model

### Start Environment


### Choose model type

For example PPO

<!--- skip-next --->
```bash
pixi run train --env BananaPick --config banana_env_config.yaml --model-type PPO --log
```

Or DQN.
Note that this needs the `--discrete-actions` flag.

<!--- skip-next --->
```bash
pixi run train --env BananaPick --config banana_env_config.yaml --model-type DQN --discrete-actions --log
```

### You may find tensorboard useful

<!--- skip-next --->
```bash
pixi run tensorboard
```

### See your freshly trained policy in action

<!--- skip-next --->
```bash
pixi run eval --model <path_to_model.pt>
```
