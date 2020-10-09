Code for Kumar et al. 2020 "Meta-Learning of Compositional Task Distributions in Humans and Machines"

Contents:
grid_grammar.py: Implements the generative grammar and provides functions to generate boards from compositional and null task distributions. 
grid_env.py: Reinforcement learning enviornment (OpenAI Gym) that implements task in the paper.
null_task_distribution.py: Code to train fully connected network to learn conditional distributions within the compositional boards and perform Gibbs sampling to obtain null task distribution.
train.py: Code to train the meta-reinforcement learning agent on the grid task described in the paper. 
held_out/: Directory containing example boards in each distribution
	all.npy: Compositional boards
	null.npy: Null boards

 
