import torch
import torch.nn.functional as F
import numpy as np 
import os


class Net(torch.nn.Module):
	def __init__(self,S=7):
		super(Net, self).__init__()
		self.fc1 = torch.nn.Linear(S*S, S*S)
		self.fc2 = torch.nn.Linear(S*S, S*S)
		self.fc3 = torch.nn.Linear(S*S, S*S)
	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.sigmoid(x)

network=Net()
if os.path.isfile("task_generator.pt"):
	network.load_state_dict(torch.load("task_generator.pt")) 
network.eval()

#Sample single board from null task distribution. 
def gibbs_sample(S=7,numSweeps=20,network=network):
	M=np.random.choice([0,1],size=49,replace=True)
	idxs=np.arange(S*S)
	for sweep in range(numSweeps):
		np.random.shuffle(idxs)
		for i in idxs:
			M_eval=M.copy()
			M_eval[i]=-1
			masked=torch.from_numpy(M_eval).float()
			preds=network(masked).detach().numpy()
			if np.random.rand()<preds[i]:
				M[i]=1
			else:
				M[i]=0
	return M 

#Vectorized form of above function to produce many boards at the same time. 
def batch_gibbs(S=7,numSweeps=20,batch_size=500,network=network):
	M=torch.from_numpy(np.random.choice([0,1],size=S*S*batch_size,replace=True).reshape((batch_size,S*S))).float()
	idxs=np.arange(S*S)
	for sweep in range(numSweeps):
		np.random.shuffle(idxs)
		for i in idxs:
			M_eval=M.clone().detach()
			M_eval[:,i]=-1
			preds=network(M_eval)[:,i]
			r=torch.rand(batch_size)
			M[r<preds,i]=1
			M[r>=preds,i]=0
	return M.detach().numpy() 

#If called as main, trains the network. 
if __name__=='__main__': 
	from grid_env import *
	S=7
	net=Net(S=S)
	lr=0.0002
	configurations=np.asarray(list(set([tuple(generate_grid('all')[0].flatten()) for _ in range(100000)])))


	optimizer=torch.optim.Adam(net.parameters())
	criterion = torch.nn.BCELoss()

	true=configurations.copy() 
	for epoch in range(10000):
		np.random.shuffle(true)
		num_changes=1
		change_idxs=np.zeros(true.shape).astype('bool')
		for i in range(true.shape[0]):
			idxs=np.random.choice(np.arange(true.shape[1]),size=num_changes,replace=False)
			change_idxs[i,idxs]=True


		data=true.copy()
		data[change_idxs]=-1
		masked=torch.from_numpy(data).float()

		labels=torch.from_numpy(true).float()
		tensor_idxs=torch.from_numpy(change_idxs).bool() 

		preds=net(masked)

		loss=criterion(preds[tensor_idxs],labels[tensor_idxs])
		y_hat=(preds[tensor_idxs]>0.5).int().numpy()
		y=labels[tensor_idxs].int().numpy()


		print(epoch,loss,np.sum(y_hat==y)/y_hat.shape[0])
		net.zero_grad()
		loss.backward()
		optimizer.step()

	torch.save(net.state_dict(),"task_generator.pt")