# pysc2-tutorial

這個repo實作了簡單的rule-based agent以及dqn agent。

# Dependencies

* Python 3.6
* Anaconda
* TensorFlow
* PySC2
* Baselines

# Getting Started

首先必須先安裝星海爭霸2並申請帳號(免費)，可參考以下slide進行安裝：  
https://goo.gl/d5L4yD

接下來安裝需要的套件，以下以MacOSX Sierra 環境為準，安裝Anaconda時請一路Enter與Yes到底。

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

# Slide

2017.10.02於Taiwan R User Group / MLDM 分享的投影片：  
https://goo.gl/oeEFvr


# References

[deepmind/pysc2](https://github.com/deepmind/pysc2)  
[openai/baselines](https://github.com/openai/baselines)  
[Building a Basic PySC2 Agent](https://medium.com/@skjb/building-a-basic-pysc2-agent-b109cde1477c)  
[chris-chris/pysc2-examples](https://github.com/chris-chris/pysc2-examples)  
[xhujoy/pysc2-agents](https://github.com/xhujoy/pysc2-agents)  
