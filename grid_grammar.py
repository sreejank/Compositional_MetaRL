import numpy as np 

from null_task_distribution import *
#1=terminal, 2=south, 3=east, 4=north, 5=west
def complement(s):
    if s=='2':
        return '4'
    if s=='4':
        return '2'
    if s=='3':
        return '5'
    if s=='5':
        return '3'
#Single chain production step. If check=true, returns true/false if further production is possible. Else returns next board.
def chain_production(grid,check=True,n=4):
    idxs=np.where(grid>1)
    possible_expansions_nodes=[]
    for i in range(len(idxs[0])):
        possible_expansions=[]
        curr=(idxs[0][i],idxs[1][i])
        val=str(grid[curr])
        if '2' in val and curr[0]+1<n and grid[curr[0]+1,curr[1]]==0:
            possible_expansions.append((curr[0]+1,curr[1],'2',curr))
        if '3' in val and curr[1]+1<n and grid[curr[0],curr[1]+1]==0:
            possible_expansions.append((curr[0],curr[1]+1,'3',curr))
        if '4' in val and curr[0]-1>=0 and grid[curr[0]-1,curr[1]]==0:
            possible_expansions.append((curr[0]-1,curr[1],'4',curr))
        if '5' in val and curr[1]-1>=0 and grid[curr[0],curr[1]-1]==0:
            possible_expansions.append((curr[0],curr[1]-1,'5',curr))
        possible_expansions_nodes.append(possible_expansions)
    if check:
        return len(possible_expansions_nodes)>0 and len(possible_expansions_nodes[0])>0
    else:
        expansions=possible_expansions_nodes[0]
        if len(expansions)>2:
            choice_idx2=np.random.choice(range(len(expansions)),size=2)
            expansion=[expansions[i] for i in choice_idx2]
            directions=[e[2] for e in expansion]
            while directions[0]!=complement(directions[1]):
                choice_idx2=np.random.choice(range(len(expansions)),size=2)
                expansion=[expansions[i] for i in choice_idx2]
                directions=[e[2] for e in expansion]
        else:
            choice_idx2=np.random.choice(range(len(expansions)),size=1)
            expansion=[expansions[i] for i in choice_idx2]

        for node in expansion:
            terminal=np.random.binomial(1,0.5)
            if terminal==1:
                grid[node[0],node[1]]=1
            else:
                grid[node[0],node[1]]=int(node[2])
            grid[node[3][0],node[3][1]]=1
    return grid


#1=terminal, 2=south, 3=east, 4=north, 5=west
#Single tree production step. If check=true, returns true/false if further production is possible. Else returns next board.
def tree_production(grid,check=True,n=4):
    idxs=np.where(grid>1)
    possible_expansions_nodes=[]
    for i in range(len(idxs[0])):
        possible_expansions=[]
        curr=(idxs[0][i],idxs[1][i])
        val=str(grid[curr])
        if '2' in val and curr[0]+1<n-1 and grid[curr[0]+1,curr[1]]==0 and sum(grid[curr[0]+1,max(curr[1]-1,0):min(curr[1]+1,n-1)+1])==0:
            possible_expansions.append((curr[0]+1,curr[1],'2',curr))
        if '3' in val and curr[1]+1<n-1 and grid[curr[0],curr[1]+1]==0 and sum(grid[max(curr[0]-1,0):min(curr[0]+1,n-1)+1,curr[1]+1])==0:
            possible_expansions.append((curr[0],curr[1]+1,'3',curr))
        if '4' in val and curr[0]-1>=0 and grid[curr[0]-1,curr[1]]==0 and sum(grid[curr[0]-1,max(curr[1]-1,0):min(curr[1]+1,n-1)+1])==0:
            possible_expansions.append((curr[0]-1,curr[1],'4',curr))
        if '5' in val and curr[1]-1>=0 and grid[curr[0],curr[1]-1]==0 and sum(grid[max(curr[0]-1,0):min(curr[0]+1,n-1)+1,curr[1]-1])==0:
            possible_expansions.append((curr[0],curr[1]-1,'5',curr))
        if len(possible_expansions)>=2:
            possible_expansions_nodes.append(possible_expansions)
    #print(possible_expansions_nodes)
    if check:
        return len(possible_expansions_nodes)>0 and len(possible_expansions_nodes[0])>0
    else:
        choice_idx=np.random.choice(range(len(possible_expansions_nodes)),size=1)[0]
        expansions=possible_expansions_nodes[choice_idx]
        if len(expansions)>2:
            choice_idx2=np.random.choice(range(len(expansions)),size=2)
            expansion=[expansions[i] for i in choice_idx2]
            directions=[e[2] for e in expansion]
            while directions[0]==complement(directions[1]) or directions[0]==directions[1]:
                choice_idx2=np.random.choice(range(len(expansions)),size=2)
                expansion=[expansions[i] for i in choice_idx2]
                directions=[e[2] for e in expansion]
            
        else:
            expansion=expansions
        node0=(expansion[0][0],expansion[0][1],expansion[0][2]+complement(expansion[1][2]),expansion[0][3])
        node1=(expansion[1][0],expansion[1][1],expansion[1][2]+complement(expansion[0][2]),expansion[0][3])
        #print(expansions,expansion,node0,node1)
        expansion=[node0,node1]
        for i,node in enumerate(expansion):
            if len(str(grid[node[3][0],node[3][0]]))>2:
                terminal=0
            else:
                terminal=np.random.binomial(1,0.5)
            if terminal:
                grid[node[0],node[1]]=1
            else:
                grid[node[0],node[1]]=int(node[2])
            grid[node[3][0],node[3][1]]=1
    return grid

#1=terminal, 2=south, 3=east, 4=north, 5=west
#Single tree production step. If check=true, returns true/false if further production is possible. Else returns next board.
def loop_production(grid,check=True,n=4):
    idxs=np.where(grid>1)
    possible_expansions_nodes=[]
    for i in range(len(idxs[0])):
        possible_expansions=[]
        curr=(idxs[0][i],idxs[1][i])
        val=str(grid[curr])
        if '2' in val and curr[0]+1<n-1 and grid[curr[0]+1,curr[1]]==0:
            possible_expansions.append((curr[0]+1,curr[1],'2',curr))
        if '3' in val and curr[1]+1<n-1 and grid[curr[0],curr[1]+1]==0:
            possible_expansions.append((curr[0],curr[1]+1,'3',curr))
        if '4' in val and curr[0]-1>=0 and grid[curr[0]-1,curr[1]]==0:
            possible_expansions.append((curr[0]-1,curr[1],'4',curr))
        if '5' in val and curr[1]-1>=0 and grid[curr[0],curr[1]-1]==0:
            possible_expansions.append((curr[0],curr[1]-1,'5',curr))
        if len(possible_expansions)>=2:
            possible_expansions_nodes.append(possible_expansions)
    #print(possible_expansions_nodes)
    if check:
        return len(possible_expansions_nodes)>0 and len(possible_expansions_nodes[0])>0
    else:
        choice_idx=np.random.choice(range(len(possible_expansions_nodes)),size=1)[0]
        expansions=possible_expansions_nodes[choice_idx]
        if len(expansions)>2:
            choice_idx2=np.random.choice(range(len(expansions)),size=2)
            expansion=[expansions[i] for i in choice_idx2]
            directions=[e[2] for e in expansion]
            while directions[0]==complement(directions[1]) or directions[0]==directions[1]:
                choice_idx2=np.random.choice(range(len(expansions)),size=2)
                expansion=[expansions[i] for i in choice_idx2]
                directions=[e[2] for e in expansion]
            
        else:
            expansion=expansions
        node0=(expansion[0][0],expansion[0][1],expansion[0][2]+expansion[1][2],expansion[0][3])
        node1=(expansion[1][0],expansion[1][1],expansion[1][2]+expansion[0][2],expansion[0][3])
        #print(expansions,expansion,node0,node1)
        expansion=[node0,node1]
        
        for i,node in enumerate(expansion):
            terminal=np.random.binomial(1,0.5)
            if terminal:
                grid[node[0],node[1]]=1
            else:
                grid[node[0],node[1]]=int(node[2])
            grid[node[3][0],node[3][1]]=1
        grid[expansion[0][0],expansion[1][1]]=int(expansion[0][2])
        grid[expansion[1][0],expansion[0][1]]=int(expansion[1][2])
    return grid 

#All: Compositional Task Distribution
#Null: Null Task Distribution  
def generate_grid(rules,n=7): 
    if rules=='all':
        production=np.random.choice([chain_production,tree_production,loop_production])
    elif rules=='chain':
        production=chain_production
    elif rules=='tree':
        production=tree_production
    elif rules=='loop':
        production=loop_production
    elif rules=='null':
        grid=gibbs_sample().reshape((7,7))
        while np.sum(grid)<3:
            grid=gibbs_sample().reshape((7,7))
        return grid

    grid=np.zeros((n,n)).astype('int')
    start=np.random.choice(list(range(2,n-1)),size=2)
    #start=[2,2]
    grid[start[0],start[1]]=2345 

    while np.sum(grid>1)>0:
        if production(grid,check=True,n=n):
            grid=production(grid,check=False,n=n)
        else:
            grid[grid>1]=1
    return grid,start 


