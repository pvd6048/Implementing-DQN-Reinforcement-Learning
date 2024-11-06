# Implementing-DQN-Reinforcement-Learning
Course Project DA241M


This course project presents the development of a reinforcement learning agent for playing the Cartpole and the Lunar Landing Game using Reinforcement Learning(RL). The agent is trained using Deep Q-Networks (DQN), which efficiently combines Neural Networks with Q-learning (A famous reinforcement learning strategies). Experimental results show that the agent learns to perform significantly well. It can also achieve scores comparable to the human-level expertise. We discuss model architecture, hyperparameter tuning, and the reward system that enables the agent’s learning process. Our results demonstrate the viability of Reinforcement Learning approaches in learning gaming strategies.

<img width="612" alt="image" src="https://github.com/user-attachments/assets/5843e6f8-3a99-458c-b9fa-51d19fa3f069">

<img width="612" alt="image" src="https://github.com/user-attachments/assets/8529fae1-7e40-49ef-bad6-bfb57378cf79">



CartPole

As the agent observes the current state of the environment and chooses an action, the environment transitions to a new state, and also returns a reward that indicates the consequences of the action. In this task, rewards are +1 for every incremental timestep and the environment terminates if the pole falls over too far or the cart moves more than 2.4 units away from center. This means better performing scenarios will run for longer duration, accumulating larger return.

The CartPole task is designed so that the inputs to the agent are 4 real values representing the environment state (position, velocity, etc.). We take these 4 inputs without any scaling and pass them through a small fully-connected network with 2 outputs, one for each action. The network is trained to predict the expected value for each action, given the input state. The action with the highest expected value is then chosen.
