# Implementing-DQN-Reinforcement-Learning
Course Project DA241M


This course project presents the development of a reinforcement learning agent for playing the Cartpole and the Lunar Landing Game using Reinforcement Learning(RL). The agent is trained using Deep Q-Networks (DQN), which efficiently combines Neural Networks with Q-learning (A famous reinforcement learning strategies). Experimental results show that the agent learns to perform significantly well. It can also achieve scores comparable to the human-level expertise. We discuss model architecture, hyperparameter tuning, and the reward system that enables the agent’s learning process. Our results demonstrate the viability of Reinforcement Learning approaches in learning gaming strategies.

<img width="612" alt="image" src="https://github.com/user-attachments/assets/5843e6f8-3a99-458c-b9fa-51d19fa3f069">

****Learning rate****:

The learning rate or step size determines to what extent newly acquired information overrides old information. A factor of 0 makes the agent learn nothing (exclusively exploiting prior knowledge), while a factor of 1 makes the agent consider only the most recent information (ignoring prior knowledge to explore possibilities).

****Discount Factor****
The discount factor, denoted as gamma, determines how much importance an agent places on future rewards. If gamma is set to 0, the agent becomes "myopic" or short-sighted, only valuing immediate rewards, rt​. As gamma approaches 1, the agent increasingly prioritizes long-term rewards, aiming to maximize future gains.

****Initial Conditions****
Since Q-learning is an iterative algorithm, it implicitly assumes an initial condition before the first update occurs. High initial values, known as "optimistic initial conditions," can promote exploration: regardless of the selected action, the update rule will reduce its value, increasing the probability of choosing other actions. The first reward can be used to reset the initial conditions. According to this approach, the first time an action is taken, the reward is used to set the Q value, enabling immediate learning when rewards are fixed and deterministic.
![image](https://github.com/user-attachments/assets/dcf7bc2a-6275-4e64-88ac-08b8ec3a9888)



# ******CartPole******

As the agent observes the current state of the environment and chooses an action, the environment transitions to a new state, and also returns a reward that indicates the consequences of the action. In this task, rewards are +1 for every incremental timestep and the environment terminates if the pole falls over too far or the cart moves more than 2.4 units away from center. This means better performing scenarios will run for longer duration, accumulating larger return.

The CartPole task is designed so that the inputs to the agent are 4 real values representing the environment state (position, velocity, etc.). We take these 4 inputs without any scaling and pass them through a small fully-connected network with 2 outputs, one for each action. The network is trained to predict the expected value for each action, given the input state. The action with the highest expected value is then chosen.
