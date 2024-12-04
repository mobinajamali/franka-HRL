# franka-HRL
Solving multi-stage, long-horizon robotic tasks in the Franka Kitchen gym environment via Imitation and Reinforcement Learning.

- This two-phase approach consists of an imitation learning
stage that produces goal-conditioned hierarchical policies, and a reinforcement learning phase that finetunes these policies for long-horizon task performance.

- Actor-critic with human experiences having weighted training
- Franks Kitchen environment is a multi-step, sparsed reward environment whre 
- In this project SAC algorithm has been used 
- coordinated attempt of moving joints

- For complex tasks with minimum rewards, one of the ways to solve is to integrate human experiences into the mix. We will be integrating a game controller interface to work with the environment to pilot the robot to take actions such as opening the microwave for many times and build up a replay buffer of at least 30000 steps for the actor to train on. We then take the buffer and feed it in the standard actor-critic process
- Used weighted replay buffer (for a defined amount of times the robot uses most of the experiences from the human generated data and for the next ones modifying the amount of reliancy to those experiences)
- For multiple tasks: A form of hierachical RL where we have a meta agent with a static list of policies that is going to coordinate the different policies and choose each one when it comes to their turn to be loaded on the memory and be used on the environment to accomplish a certain task


<!-- ## 1. MADDPG (Multi Agent Deep Deterministic Policy Gradient):
- MADDPG extends the DDPG algorithm to multi-agent settings by using centralized learning and decentralized execution. Each agent has a centralized critic that evaluates all agents' actions, while having a decentralized actor for decision-making.
- Paper: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
  
### Simple-speaker-listener-v4
Demo| Results| 
:-----------------------:|:-----------------------:|
![](./MADDPG/thumbnails/video-ezgif.com-video-to-gif-converter.gif)| ![](./MADDPG/plots/maddpg.png)| 

### :bell: More algorithms and enhancements are coming soon!

## Installation
```bash
git@github.com:mobinajamali/franka-HRL.git
```
```shell
pip install -r requirements.txt
```



## Acknowledgement
- Credits go to [Robert Cowher](https://github.com/bobcowher) for his brilliant course!
- 
-->


