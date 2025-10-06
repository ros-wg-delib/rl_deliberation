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
- ros-wg-delib.png
aspectratio: 169
header-includes:
- \hypersetup{colorlinks=true}
---

# Introduction
<!-- Build with `pandoc -t beamer main.md -o main.pdf` -->
<!-- https://pandoc.org/MANUAL.html#variables-for-beamer-slides -->

## What is Reinforcement Learning (RL)?

::: columns

:::: column
Basic model:
Given an __agent__ and an __environment__.

- In discrete time steps $t = 0, 1, 2, ...$
- The environment is in a __state $s_t$__
- Agent performs an __action $a_t$__
- Environment responds with a new state $s_{t+1}$ and a reward $r_{t+1}$
- Based on that $s_t$ the agent selects the __next action__ $a_{t+1}$
- ... and so on.

The goal is to maximize the __cumulative reward__ over time.
$R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$, where $\gamma$ is a discount factor.
::::

:::: column
![Agent Environment Interaction](agent-env.drawio.png)
::::

:::

# Test your setup

```bash
pixi run pyrobosim_demo
```

# Basics: MDP

A Markov Decision Process (MDP) is defined as $< \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma >$.

- $\mathcal{S}$ States.
- $\mathcal{A}$ Possible Actions.
- ...

# Basics: Belman Equation

$V(s) = \max_a {   }$.

# Reference: Reinforcement Learning Algorithms

## Deep Q Network (DQN)

Learns a Q-function $Q(s, a)$.
Introduced *experience replay* (off-policy) and *target networks*.
[Minh et al., 2013](https://arxiv.org/abs/1312.5602), [Minh et al., 2015](https://www.nature.com/articles/nature14236), [SB3 docs](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)

## Advantage Actor-Critic (A2C)

Introduced the *advantage function* $A(s, a) = Q(s, a) - V(s)$ to reduce variance.
[Mnih et al., 2016](https://arxiv.org/abs/1602.01783), [SB3 docs](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)

## Proximal Policy Optimization (PPO)

Optimize policy directly. Uses a *clipped surrogate objective* to ensure stability.
[Schulman et al., 2017](https://arxiv.org/abs/1707.06347), [SB3 docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

## Soft Actor-Critic (SAC)

Off-policy algorithm encouraging exploration with *entropy* term.
[Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290), [SB3 docs](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
