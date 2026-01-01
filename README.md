[繁體中文](README.zh-TW.md)

# pysc2-tutorial

This repository implements a simple rule-based agent and a DQN agent.

# Dependencies

* Python 3.6
* Anaconda
* TensorFlow
* PySC2
* Baselines

# Getting Started

First, you must install StarCraft II and create an account (it's free). You can refer to the following slides for installation:
https://goo.gl/d5L4yD

Next, install the required packages. The following instructions are based on a macOS Sierra environment. When installing Anaconda, please press Enter and select "Yes" all the way through.

```
wget https://repo.continuum.io/archive/Anaconda3-5.0.0-MacOSX-x86_64.sh
bash Anaconda3-5.0.0-MacOSX-x86_64.sh
source .bash_profile
pip install tensorflow
pip install baselines
pip install pysc2
pip install absl-py
```

# How to run

* scripted agent
```
python -m pysc2.bin.agent --map Simple64 --agent scripted_agent.simple_agent_protoss.RuleBaseAgent --agent_race protoss
```

* dqn agent
```
python dqn_agent/train_mineral_shards.py
```

# Result

* scripted agent

Consistently defeats the computer at the easiest difficulty level.

* dqn agent

Gets stuck around 13-14 points and cannot improve further.

# Slide

Slides from a presentation at Taiwan R User Group / MLDM on 2017.10.02:
https://goo.gl/oeEFvr


# References

[deepmind/pysc2](https://github.com/deepmind/pysc2)
[openai/baselines](https://github.com/openai/baselines)
[Building a Basic PySC2 Agent](https://medium.com/@skjb/building-a-basic-pysc2-agent-b109cde1477c)
[chris-chris/pysc2-examples](https://github.com/chris-chris/pysc2-examples)
[xhujoy/pysc2-agents](https://github.com/xhujoy/pysc2-agents)
