# Multi-Agent Reinforcement Learning Capture the Flag
This repository contains the code used in my study on agent specialisation with heterogeneous agents, using reinforcement learning.

### The Important Stuff
The key files in this repository are:

| Filename | Description |
|----------|-------------|
| requirements.txt | Contains python packages required to replicate the development environment. |
| 0_the_split.py, ..., 8_arena.py | These files contain the logic for each of the experiments in my study. Running these files from the command line will execute the experiments and save results in a /runs subfolder. |
| gridworld_ctf.py | Contains the logic of the CTF environment. |
| ppo.py | Contains the implementation of PPO.
| league_training.py | Contains the logic used for self-play, and also has logic for league training. League training was not used in my study as simple self-play produced good results, and runs much faster. |
| metrics_logger | Generates metrics from the experiments. |
### Repository Contents

The entire contents of this repository are described below:

```bash
├── alt_exp/                        <- Folder containing alternative versions of experiments, not used in final report.
├── gifs/                           <- Folder animated gifs of the experiments.
├── img/                            <- Folder containing images used in presentation layer of the environment.
├── report/                         <- Folder containing images used in the final report.
├── 0_the_split.py                  <- Python file to run experiment 1.
├── 1_fence.py                      <- Python file to run experiment 2.
├── 2_jailbreak.py                  <- Python file to run experiment 3.
├── 3_one_way_out.py                <- Python file to run experiment 4.
├── 4_keyhole.py                    <- Python file to run experiment 5.
├── 5_skittles.py                   <- Python file to run experiment 6.
├── 6_the_wall.py                   <- Python file to run experiment 7.
├── 7_gridlocked.py                 <- Python file to run experiment 8.
├── 8_arena.py                      <- Python file to run experiment 9.
├── agent_network.py                <- Python file containing policy and value networks.
├── analysis.ipynb                  <- Jupyter notebook containing summaries of results for each experiment.
├── env_testing.ipynb               <- Jupyter notebook used for testing and debugging the CTF environment.
├── gridworld_ctf.py                <- Python file containing the logic of the CTF environment.
├── league_training.py              <- Python file containing the logic for self-play and league training.
├── learning_experiments.ipynb      <- Jupyter notebook containing logic for learning experiments, i.e. action masking.
├── metrics_logger.py               <- Python file containing logic for capturing experiment metrics.
├── ppo.py                          <- Python file containing the PPO implementation.
├── README.md                       <- README file.
├── requirements.txt                <- Python packages required to replicate the development environment.
├── scenarios.py                    <- Python file containing map configurations for each experiment.
├── utils.py                        <- Python file containing utility functions for duelling, producing tables, plots etc.
├── .gitignore                      <- Stuff to ignore.
```

