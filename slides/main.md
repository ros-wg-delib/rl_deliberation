---
title:

- Reinforcement Learning for Deliberation in ROS 2
author:
- Christian Henkel
- Sebastian Castro
theme:
- Bergen
date:
- ROSCon 2025 / October 27, 2025
logo:
- media/ros-wg-delib.png
aspectratio: 169
fontsize: 9pt
colorlinks: true
header-includes:
  - \usepackage{listings}
  - \usepackage{xcolor}
  - \lstset{
      basicstyle=\ttfamily\small,
      backgroundcolor=\color{gray!10},
      keywordstyle=\color{blue},
      stringstyle=\color{orange},
      commentstyle=\color{gray},
      showstringspaces=false
    }
---

# Agenda

<!-- Build with `pandoc -t beamer main.md -o main.pdf --listings` -->
<!-- https://pandoc.org/MANUAL.html#variables-for-beamer-slides -->

| __Time__       | __Topic__                                               |
|----------------|---------------------------------------------------------|
| 13:00 - 13:30  | Introduction / Software Setup                           |
| 13:30 - 14:00  | (Very) Quick intro to Reinforcement Learning            |
| 14:00 - 15:00  | Training and evaluating RL agents                       |
| 15:00 - 15:30  | [Coffee break / leave a longer training running]        |
| 15:30 - 16:15  | Evaluating trained agents and running in ROS nodes      |
| 16:15 - 17:00  | Discussion: ROS 2, RL, and Deliberation                 |

# Software Setup

1. Clone the repository

    ```bash
    git clone --recursive \
      https://github.com/ros-wg-delib/rl_deliberation.git
    ```

2. Install Pixi:

    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

    (or \href{https://pixi.sh/latest/installation}{\texttt{https://pixi.sh/latest/installation}} – recommend autocompletion!)

3. Build the project:

    ```bash
    pixi run build
    ```

4. Run an example:

    ```bash
    pixi run start_world --env GreenhousePlain
    ```

# Introduction

::: columns

:::: column
__What is Reinforcement Learning (RL)?__

---

Basic model:

- Given an __agent__ and an __environment__.
- Subject to the __state__ of the environment,
- the agent takes an __action__.
- the environment responds with a new __state__ and a __reward__.

::::

:::: column
![Agent Environment Interaction](media/agent-env.drawio.png)

See also [Sutton and Barto, Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf)
::::

:::

# Introduction: Notation

::: columns

:::: column

- Discrete time steps $t = 0, 1, 2, \dots$
- The environment is in a __state $S_t$__
- Agent performs an __action $A_t$__
- Environment responds with a new state $S_{t+1}$ and a reward $R_{t+1}$
- Based on that $S_t$ the agent selects the __next action__ $A_{t+1}$

::::

:::: column
![Agent Environment Interaction](media/agent-env-sym.drawio.png)
::::

:::

# RL Software in this Workshop

::: columns

:::: column

![Gymnasium \tiny](media/gymnasium.png)

\footnotesize Represents environments for RL <https://gymnasium.farama.org/>

::::

:::: column

![Stable Baselines 3 (SB3)](media/sb3.png)

\footnotesize RL algorithm implementations in PyTorch <https://github.com/DLR-RM/stable-baselines3>

::::

:::

# Exercise 1: You are the agent

::: columns

:::: column

Start by exploring the environment.

```bash
pixi run start_world --env \
  GreenhousePlain
```

![Greenhouse environment](media/greenhouse.png){height=100px}

You are a robot that has to water plants in a greenhouse.

::::

:::: column

Then, in another terminal, run:

```bash
pixi run eval --model manual \
  --env GreenhousePlain
```

```plain
Enter action from [0, 1]:
```

On this prompt, you can choose:

- __0__: Move forward without watering, or
- __1__: Water the plant and move on.

But be __careful__: If you water the evil plant _(red)_, you will be eaten.

![Evil Plant \tiny flickr/Tippitiwichet](media/venus_flytrap_src_wikimedia_commons_Tippitiwichet.jpg){width=80px}

::::

:::

# Introduction: Environment = MDP

## MDP

We assume the environment to be a __Markov Decision Process (MDP)__.
An MDP is defined as $< \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}>$.

- $s \in \mathcal{S}$ states and $a \in \mathcal{A}$ actions as above.
- $\mathcal{P}$ State Transition Probability: $P(s'|s, a)$.
  - For an action $a$ taken in state $s$, what is the probability of reaching state $s'$?
- $\mathcal{R}$ Reward Function: $R(s, a)$.
  - We will use this to motivate the agent to learn desired behavior.

Implicit to the above is the __Markov Property__:

The future state $S_{t+1}$ depends only on the current state $S_t$
and action $A_t$, not on the sequence of events that preceded it.

# Introduction: Agent = Policy

## Policy

The agent's behavior is defined by a __policy__ $\pi$.
A policy is a mapping from states to actions: $\pi: \mathcal{S} \rightarrow \mathcal{A}$.

## Reminder

We are trying to optimize the __cumulative reward__ (or __return__) over time:

$$
G_t = R_0 + R_1 + R_2 + \dots
$$

In practice, we use a __discount factor__ $\gamma \in [0, 1]$ to prioritize immediate rewards:

$$
G_t = R_0 + \gamma R_1 + \gamma^2 R_2 + \dots
$$
$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k}
$$

# Introduction: Learning

How do we learn a good policy?

## Bellman Equation

This is probably the __most fundamental equation in RL__.
It assigns a value to each state $s$ under a policy $\pi$:

$$v_{\pi}(s) = \mathbb{E}_{\pi} [G_t | S_t = s]$$
$$ = \mathbb{E}_{\pi} [R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s]$$
$$ = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')]$$

Here, $v_{\pi}(s)$ is known as the __state value function__.

# Introduction: Temporal Differencing

The Bellman equation gives rise to __temporal differencing (TD)__ for training a policy.

$$v_{\pi}(S_t) \leftarrow (1 - \alpha) v_{\pi}(S_t) + \alpha (R_{t+1} + \gamma v_{\pi}(S_{t+1}))$$

where

- $v_{\pi}(S_t)$ is the expected value of state $S_t$
- $R_{t+1} + \gamma v_{\pi}(S_{t+1})$ is the actual reward obtained at $S_t$ plus the expected value of the next state $S_{t+1}$
- $R_{t+1} + \gamma v_{\pi}(S_{t+1}) - v_{\pi}(S_t)$ is the __TD error__.
- $\alpha$ is the __learning rate__.

\small (a variant using the __state-action value function__ $Q_{\pi}(s, a)$ is known as __Q-learning__.)

# Classic RL: Tabular Methods

RL began with known MDPs + discrete states/actions, so $v_{\pi}(s)$ or $q_{\pi}(s,a)$ are __tables__.

![Grid world \tiny ([Silver, 2025](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-3-planning-by-dynamic-programming-.pdf))](media/grid-world.png){width=180px}

Can use __dynamic programming__ to iterate through the entire environment and converge on an optimal policy.

::: columns

:::: column

![Value iteration \tiny ([Silver, 2025](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-3-planning-by-dynamic-programming-.pdf))](media/value-iteration.png){width=80px}

::::

:::: column

![Policy iteration \tiny ([Silver, 2025](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-3-planning-by-dynamic-programming-.pdf))](media/policy-iteration.png){width=80px}

::::

:::

# Model-Free Reinforcement Learning

If the state-action space is too large, need to perform __rollouts__ to gain experience.

Key: Balancing __exploitation__ and __exploration__!

![Model-free RL methods \tiny ([Silver, 2025](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-4-model-free-prediction-.pdf))](media/model-free-rl.png){width=420px}


# Deep Reinforcement Learning

When the observation space is too large (or worse, continuous), tabular methods no longer work.

Need a different function approximator -- *...why not a neural network?*

![Deep Q-Network \tiny ([Mnih et al., 2015](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf))](media/dqn.png){width=300px}

__Off-policy__: Can train on old experiences from a *replay buffer*.

# Actor-Critic / Policy Gradient Methods

DQN only works for discrete actions, so what about continuous actions?

::: columns

:::: column

- __Critic__ approximates value function. Trained via TD learning.

- __Actor__ outputs actions (i.e., the policy). Trained via __policy gradient__, backpropagated from critic loss.

Initial methods were __on-policy__ -- can only train on the latest version of the policy with current experiences.
Example: Proximal Policy Optimization (PPO) ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)).

Other approaches train actor and critic at different time scales to allow off-policy.
Example: Soft Actor-Critic (SAC) ([Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)).

::::

:::: column

![Actor-Critic methods \tiny ([Sutton + Barto, 2020](http://incompleteideas.net/book/the-book-2nd.html))](media/actor-critic.png){width=180px}

::::

:::

# Reference: Reinforcement Learning Algorithms

## \footnotesize Deep Q Network (DQN)

Learns a Q-function $Q(s, a)$.
Introduced _experience replay_ (off-policy) and _target networks_.
[Mnih et al., 2013](https://arxiv.org/abs/1312.5602), [Mnih et al., 2015](https://www.nature.com/articles/nature14236), [SB3 docs](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)

## \footnotesize Advantage Actor-Critic (A2C)

Introduced the _advantage function_ $A(s, a) = Q(s, a) - V(s)$ to reduce variance.
[Mnih et al., 2016](https://arxiv.org/abs/1602.01783), [SB3 docs](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)

## \footnotesize Proximal Policy Optimization (PPO)

Optimize policy directly. Uses a _clipped surrogate objective_ to ensure stability.
[Schulman et al., 2017](https://arxiv.org/abs/1707.06347), [SB3 docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

## \footnotesize Soft Actor-Critic (SAC)

Off-policy algorithm encouraging exploration with _entropy_ term.
[Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290), [SB3 docs](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)

# Exercise 2: Run with a Random Agent

```bash
pixi run start_world --env GreenhousePlain

pixi run eval --realtime --model \
  pyrobosim_ros_gym/policies/GreenhousePlain_DQN_random.pt
```

![Greenhouse environment](media/greenhouse.png){height=150px}


# Exercise 3: Training Your First Agent

Start the world.

```bash
pixi run start_world --env GreenhousePlain
```

Kick off training.

```bash
pixi run train --config greenhouse_env_config.yaml \
  --env GreenhousePlain --algorithm DQN --discrete-actions \
  --realtime
```

The `--config` file points to `pyrobosim_ros_gym/config/greenhouse_env_config.yaml`, which lets you easily set up different algorithms and training parameters.

# Exercise 3: Training Your First Agent (For Real...)

... this is going to take a while.
Let's speed things up.

```bash
# Run simulation headless, i.e., without the GUI
pixi run start_world --env GreenhousePlain --headless

# No "realtime" flag, i.e., run actions as fast as possible
pixi run train --config greenhouse_env_config.yaml \
  --env GreenhousePlain --algorithm DQN --discrete-actions
```

__NOTE:__ Seeding the training run is important for reproducibility!

We are running with `--seed 42` by default, but you can change it.

# Visualizing Training Progress

Stable Baselines 3 has visualization support for [TensorBoard](https://www.tensorflow.org/tensorboard).

By adding the `--log` argument, a log file will be written to the `train_logs` folder.

```bash
pixi run train --config greenhouse_env_config.yaml \
  --env GreenhousePlain --algorithm DQN --discrete-actions --log
```

Open TensorBoard and follow the URL displayed (usually `http://localhost:6006/`).

```bash
pixi run tensorboard
```

![TensorBoard](media/tensorboard.png){width=200px}

# Evaluating Your Trained Agent

Once you have your trained model, you can evaluate it against the simulator.

```bash
pixi run eval --model <path_to_your_model>.pt --num-episodes 10
```

By default, this will run just like training (as quickly as possible).

You can add the `--realtime` flag to slow things down to "real-time" so you can visually inspect the results.

![Example evaluation results](media/eval-results.png){width=240px}

# Exercise 4: Train More Complicated Environment Variations

Training the `GreenhousePlain` environment is easy because the environment is *deterministic*; the plants are always in the same locations.

For harder environments, you may want to switch algorithms (e.g., `PPO` or `SAC`).

::: columns

:::: column

![`GreenhouseRandom` environment](media/greenhouse-random.png){width=120px}

Plants are now spawned in random locations -- but only one per table.

::::

::: column

![`GreenhouseBattery` environment](media/greenhouse-battery.png){width=120px}

Watering costs 49% battery -- must recharge after watering twice.

Charging is a new action (id `3`).

::::

:::

# Deploying a Trained Policy as a ROS Node

1. Start an environment of your choice.

```bash
pixi run start_world --env GreenhouseRandom
```

2. Start the node with an appropriate model.

```bash
pixi run policy_node --model <path_to_your_model>.pt
```

3. Open an interactive shell.

```bash
pixi shell
```

4. In the shell, send an action goal to run the policy to completion!

```bash
ros2 action send_goal /execute_policy rl_interfaces/ExecutePolicy {}
```

# Discussion 1: Scaling up Learning

- Parallel simulation
- Curriculum learning

# Discussion 2: RL Experimentation

- Running with multiple seeds and reporting intervals
- Hyperparameter tuning

# Discussion 3: Deploying policies to ROS

- Python: PyTorch
- C++: ONNX + ros2_control

# Discussion 4: RL for Deliberation

## Background

Much state of the art RL is for fast, low-level control policies (e.g., locomotion)

- Requires sim-to-real training because on-robot RL is hard and/or unsafe.
- Alternatives: fine-tune pretrained policies or train _residual_ policies.

## Deliberation

How does this change for deliberation applications?

- Facilitates on-robot RL: train high-level decision making, with a safety layer below.
- What kinds of high-level decisions can/should be learned?

# Resources

## RL theory

- Sutton + Barto Textbook: <http://incompleteideas.net/book/the-book-2nd.html>
- David Silver Lectures: <https://davidstarsilver.wordpress.com/teaching/>

## Deliberation

- ROS Deliberation Community Group: <https://github.com/ros-wg-delib>
- Workshop Repo: <https://github.com/ros-wg-delib/rl_deliberation>

![Happy RL journey!](media/twitter-post.png){height=100px}
