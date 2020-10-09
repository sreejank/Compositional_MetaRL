import gym
from gym.utils import seeding
from PIL import Image as PILImage
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
from itertools import product 
from itertools import permutations 
import pickle
from grid_grammar import * 

	

class BattleshipEnv(gym.Env): 
	
	reward_range = (-float('inf'), float('inf'))
	metadata = {'render.modes': ['human', 'rgb_array'],'video.frames_per_second' : 3} 
	#hold_out>0: run env while excluding boards in heldout/<rules>.npy. 
	#hold_out<0: run env cycling through boards in heldout/<rules>.npy
	#hold_out=0: run env on whole distribution without holding out boards (not used in paper). 
	def __init__(self,rules='chain',n_board=7,hold_out=0):
		
		self.viewer = None
		self.seed()
		action_converter=[]
		for i in range(n_board):
			for j in range(n_board): 
				action_converter.append((i,j))
		self.action_converter=np.asarray(action_converter)
		self.n_board=n_board
		self.hold_out=hold_out
		self.rules=rules

		if hold_out==-1:
			self.heldout=np.load('held_out/'+rules+'.npy')
			self.maze_idx=0 
			self.maze=np.reshape(self.heldout[self.maze_idx],(7,7))
			if self.rules in ['all','chain','tree','loop']:
				start=np.load('held_out/'+self.rules+'_starts.npy')[self.maze_idx]
			else:
				hit_idx=np.where(self.maze==1)
				choice=np.random.choice(list(range(len(hit_idx[0]))),size=1)
				start=(hit_idx[0][choice],hit_idx[1][choice])

		else:
			if hold_out>0:
				heldout=np.load('held_out/'+rules+'.npy')
				self.heldout=set([tuple(x) for x in heldout]) 
			
			
			gen=generate_grid(self.rules,n=self.n_board)
			if len(gen)==2:
				grid,start=gen 
			else:
				grid=gen 
				hit_idx=np.where(grid==1)
				choice=np.random.choice(list(range(len(hit_idx[0]))),size=1)
				start=(hit_idx[0][choice],hit_idx[1][choice])

			if hold_out>0:
				while tuple(grid.flatten()) in self.heldout:
					gen=generate_grid(self.rules,n=self.n_board)
					if len(gen)==2:
						grid,start=gen 
					else:
						grid=gen 
						hit_idx=np.where(grid==1)
						choice=np.random.choice(list(range(len(hit_idx[0]))),size=1)
						start=(hit_idx[0][choice],hit_idx[1][choice])
			self.maze=grid

		self.board=np.ones(self.maze.shape)*-1
		self.current_position=start 
		self.board[self.current_position[0],self.current_position[1]]=1
		self.num_hits=0
		self.self_hits={}
		
		self.observation_space = Box(low=-1, high=1, shape=(n_board*n_board+n_board*n_board+1,), dtype=np.float)
		self.action_space = Discrete(np.prod(self.maze.shape))
		self.nA=n_board*n_board

		self.prev_reward=0
		self.prev_action=np.zeros((self.nA,))

		self.valid_actions=[1 for _ in range(self.nA)]
		
		
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		
	def step(self, action):
		prev_position=self.current_position
		self.current_position=self.action_converter[action]
		reward=0
		
		if self.board[self.current_position[0],self.current_position[1]]==-1:
			if self.maze[self.current_position[0],self.current_position[1]]==1:
				self.board[self.current_position[0],self.current_position[1]]=1
				self.num_hits+=1
				reward=1
			else:
				self.board[self.current_position[0],self.current_position[1]]=0
				reward=-1
		else:
			reward=-2
			if (self.current_position[0],self.current_position[1]) not in self.self_hits.keys(): 
				self.self_hits[(self.current_position[0],self.current_position[1])]=1
			else:
				self.self_hits[(self.current_position[0],self.current_position[1])]+=1
					
		if self._is_goal():
			reward=+10
			done = True
			if self.hold_out==-1:
				self.maze_idx+=1
		else:
			done = False

		p_action=self.prev_action
		p_reward=self.prev_reward
		self.prev_action=np.zeros((self.nA,))
		self.prev_action[action]=1
		self.prev_reward=reward 

		obs=self.board.flatten()

		obs_array=np.concatenate((obs,p_action,[p_reward]))
		
		
		return obs_array, reward, done, {}
	
	def _is_goal(self):
		return np.sum(self.board==1)==np.sum(self.maze==1)
	
	def get_image(self):
		img=np.empty((*self.board.shape, 3), dtype=np.uint8)
		for i in range(self.board.shape[0]):
			for j in range(self.board.shape[1]):
				if self.board[i,j]==-1:
					img[i,j,:]=255,255,255

				elif self.board[i,j]==1:
					img[i,j,:]=255,0,0
					if (i,j) in self.self_hits.keys():
						if (255-10*self.self_hits[(i,j)])<5:
							img[i,j,:]=0,0,0
						else:
							img[i,j,:]=(255-10*self.self_hits[(i,j)]),0,0
				else:
					img[i,j,:]=0,0,255
					if (i,j) in self.self_hits.keys():
						if (255-10*self.self_hits[(i,j)])<5:
							img[i,j,:]=0,0,0
						else:
							img[i,j,:]=0,0,(255-10*self.self_hits[(i,j)])
						
		return img
	
	def set_task(self,task):
		self.maze = task
		self.board=np.zeros(self.maze.shape)
		self.current_position=[np.random.choice(range(self.maze.shape[0])),np.random.choice(self.maze.shape[1])]

		self.num_hits=0
		self.self_hits={}
		return self.board.flatten()
	
	def reset(self):
		if self.hold_out==-1:
			self.maze=np.reshape(self.heldout[self.maze_idx%len(self.heldout)],(7,7))
			if self.rules in ['all','chain','tree','loop']:
				start=np.load('held_out/'+self.rules+'_starts.npy')[self.maze_idx%len(self.heldout)]
			else:
				hit_idx=np.where(self.maze==1)
				choice=np.random.choice(list(range(len(hit_idx[0]))),size=1)
				start=(hit_idx[0][choice],hit_idx[1][choice])
		else:
			gen=generate_grid(self.rules,n=self.n_board)
			if len(gen)==2:
				grid,start=gen 
			else:
				grid=gen 
				hit_idx=np.where(grid==1)
				choice=np.random.choice(list(range(len(hit_idx[0]))),size=1)
				start=(hit_idx[0][choice],hit_idx[1][choice])

			if self.hold_out>0:
				while tuple(grid.flatten()) in self.heldout:
					gen=generate_grid(self.rules,n=self.n_board)
					if len(gen)==2: 
						grid,start=gen 
					else:
						grid=gen 
						hit_idx=np.where(grid==1)
						choice=np.random.choice(list(range(len(hit_idx[0]))),size=1)
						start=(hit_idx[0][choice],hit_idx[1][choice])
			self.maze=grid

		self.board=np.ones(self.maze.shape)*-1
		self.current_position=start 
		self.board[self.current_position[0],self.current_position[1]]=1

		self.num_hits=0
		self.self_hits={}
		obs=self.board.flatten()

		obs_array=np.concatenate((obs,self.prev_action,[self.prev_reward]))
		self.valid_actions=[1 for _ in range(self.nA)] 
		return obs_array
	
	def render(self, mode='human', max_width=500): 
		img = self.get_image()
		img = np.asarray(img).astype(np.uint8)
		img_height, img_width = img.shape[:2]
		ratio = max_width/img_width
		img = PILImage.fromarray(img).resize([int(ratio*img_width), int(ratio*img_height)])
		img = np.asarray(img)
		if mode == 'rgb_array':
			return img
		elif mode == 'human':
			from gym.envs.classic_control.rendering import SimpleImageViewer
			if self.viewer is None:
				self.viewer = SimpleImageViewer()
			self.viewer.imshow(img)
			
			return self.viewer.isopen 
	def close(self):
		if self.viewer is not None:
			self.viewer.close()
			self.viewer = None
	
def register_grid_env(env_id,rules,n_board=7,hold_out=0,max_episode_steps=60):  
	gym.envs.register(id=env_id, entry_point=BattleshipEnv, max_episode_steps=max_episode_steps,kwargs={'rules':rules,'n_board':n_board,'hold_out':hold_out}) 