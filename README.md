# Implementing-DQN-Reinforcement-Learning
#Course Project DA241M


This course project presents the development of a reinforcement learning agent for playing the Cartpole and the Lunar Landing Game using Reinforcement Learning(RL). The agent is trained using Deep Q-Networks (DQN), which efficiently combines Neural Networks with Q-learning (A famous reinforcement learning strategies). Experimental results show that the agent learns to perform significantly well. It can also achieve scores comparable to the human-level expertise. We discuss model architecture, hyperparameter tuning, and the reward system that enables the agent’s learning process. Our results demonstrate the viability of Reinforcement Learning approaches in learning gaming strategies.

# <img width="612" alt="image" src="https://github.com/user-attachments/assets/5843e6f8-3a99-458c-b9fa-51d19fa3f069">

****Learning rate****:

The learning rate or step size determines to what extent newly acquired information overrides old information. A factor of 0 makes the agent learn nothing (exclusively exploiting prior knowledge), while a factor of 1 makes the agent consider only the most recent information (ignoring prior knowledge to explore possibilities).

****Discount Factor****:

The discount factor, denoted as gamma, determines how much importance an agent places on future rewards. If gamma is set to 0, the agent becomes "myopic" or short-sighted, only valuing immediate rewards, rt​. As gamma approaches 1, the agent increasingly prioritizes long-term rewards, aiming to maximize future gains.

****Initial Conditions****:

Since Q-learning is an iterative algorithm, it implicitly assumes an initial condition before the first update occurs. High initial values, known as "optimistic initial conditions," can promote exploration: regardless of the selected action, the update rule will reduce its value, increasing the probability of choosing other actions. The first reward can be used to reset the initial conditions. According to this approach, the first time an action is taken, the reward is used to set the Q value, enabling immediate learning when rewards are fixed and deterministic.
![image](https://github.com/user-attachments/assets/dcf7bc2a-6275-4e64-88ac-08b8ec3a9888)



# ******CartPole******

As the agent observes the current state of the environment and chooses an action, the environment transitions to a new state, and also returns a reward that indicates the consequences of the action. In this task, rewards are +1 for every incremental timestep and the environment terminates if the pole falls over too far or the cart moves more than 2.4 units away from center. This means better performing scenarios will run for longer duration, accumulating larger return.

The CartPole task is designed so that the inputs to the agent are 4 real values representing the environment state (position, velocity, etc.). We take these 4 inputs without any scaling and pass them through a small fully-connected network with 2 outputs, one for each action. The network is trained to predict the expected value for each action, given the input state. The action with the highest expected value is then chosen.


# **Replay Memory**
We’ll be using experience replay memory for training our DQN. It stores the transitions that the agent observes, allowing us to reuse this data later. By sampling from it randomly, the transitions that build up a batch are decorrelated. It has been shown that this greatly stabilizes and improves the DQN training procedure.

For this, we’re going to need two classes:

Transition - a named tuple representing a single transition in our environment. It essentially maps (state, action) pairs to their (next_state, reward) result, with the state being the screen difference image as described later on.

ReplayMemory - a cyclic buffer of bounded size that holds the transitions observed recently. It also implements a .sample() method for selecting a random batch of transitions for training.


Our environment is deterministic, so all equations presented here are also formulated deterministically for the sake of simplicity. In the reinforcement learning literature, they would also contain expectations over stochastic transitions in the environment.

Our aim will be to train a policy that tries to maximize the discounted, cumulative reward 
R
t
0
=
∑
t
=
t
0
∞
γ
t
−
t
0
r
t
R 
t 
0
​
 
​
 =∑ 
t=t 
0
​
 
∞
​
 γ 
t−t 
0
​
 
 r 
t
​
 , where 
R
t
0
R 
t 
0
​
 
​
  is also known as the return. The discount, 
γ
γ, should be a constant between 
0
0 and 
1
1 that ensures the sum converges. A lower 
γ
γ makes rewards from the uncertain far future less important for our agent than the ones in the near future that it can be fairly confident about. It also encourages agents to collect reward closer in time than equivalent rewards that are temporally far away in the future.

The main idea behind Q-learning is that if we had a function 
Q
∗
:
S
t
a
t
e
×
A
c
t
i
o
n
→
R
Q 
∗
 :State×Action→R, that could tell us what our return would be, if we were to take an action in a given state, then we could easily construct a policy that maximizes our rewards:

π
∗
(
s
)
=
arg
⁡
max
⁡
a
 
Q
∗
(
s
,
a
)
π 
∗
 (s)=arg 
a
max
​
  Q 
∗
 (s,a)
However, we don’t know everything about the world, so we don’t have access to 
Q
∗
Q 
∗
 . But, since neural networks are universal function approximators, we can simply create one and train it to resemble 
Q
∗
Q 
∗
 .

For our training update rule, we’ll use a fact that every 
Q
Q function for some policy obeys the Bellman equation:

Q
π
(
s
,
a
)
=
r
+
γ
Q
π
(
s
′
,
π
(
s
′
)
)
Q 
π
 (s,a)=r+γQ 
π
 (s 
′
 ,π(s 
′
 ))
The difference between the two sides of the equality is known as the temporal difference error, 
δ
δ:

δ
=
Q
(
s
,
a
)
−
(
r
+
γ
max
⁡
a
′
Q
(
s
′
,
a
)
)
δ=Q(s,a)−(r+γ 
a
max
′
​
 Q(s 
′
 ,a))
To minimize this error, we will use the Huber loss. The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large - this makes it more robust to outliers when the estimates of 
Q
Q are very noisy. We calculate this over a batch of transitions, 
B
B, sampled from the replay memory:

L
=
1
∣
B
∣
∑
(
s
,
a
,
s
′
,
r
)
 
∈
 
B
L
(
δ
)
L= 
∣B∣
1
​
  
(s,a,s 
′
 ,r) ∈ B
∑
​
 L(δ)
where
L
(
δ
)
=
{
1
2
δ
2
for 
∣
δ
∣
≤
1
,
∣
δ
∣
−
1
2
otherwise.
whereL(δ)={ 
2
1
​
 δ 
2
 
∣δ∣− 
2
1
​
 
​
  
for ∣δ∣≤1,
otherwise.
​
 

