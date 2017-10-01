# pysc2-tutorial

這個repo實作了簡單的rule-based agent以及dqn agent。

# Dependencies

* Python 3.6
* Anaconda
* TensorFlow
* PySC2
* Baselines

# Getting Started

以下以MacOSX Sierra 環境為準，安裝Anaconda時請一路Enter與Yes到底。

```
wget https://repo.continuum.io/archive/Anaconda3-5.0.0-MacOSX-x86_64.sh
bash Anaconda3-5.0.0-MacOSX-x86_64.sh
source .bash_profile
pip install tensorflow
pip install baselines
pip install pysc2
```

# How to run

* scripted agent
```
python -m pysc2.bin.agent --map Simple64 --agent scripted_agent.simple_agent_protoss.RuleBaseAgent --agent_race P --bot_race Z
```

* dqn agent
```
python dqn_agent/train_mineral_shards.py
```

# Result

* scripted agent

穩定打贏難度級別最簡單的電腦。

* dqn agent

卡在13~14分左右就上不去了。


# References

[deepmind/pysc2](https://github.com/deepmind/pysc2)  
[openai/baselines](https://github.com/openai/baselines)  
[Building a Basic PySC2 Agent](https://medium.com/@skjb/building-a-basic-pysc2-agent-b109cde1477c)  
[chris-chris/pysc2-examples](https://github.com/chris-chris/pysc2-examples)  
[xhujoy/pysc2-agents](https://github.com/xhujoy/pysc2-agents)  
