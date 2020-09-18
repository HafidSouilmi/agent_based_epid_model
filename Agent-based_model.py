#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np
import matplotlib.pyplot as plt 


# In[110]:


def updatePosition(homes):
    r=2*np.random.rand(homes.shape[0],homes.shape[1])-1
    x= homes+r
    return x


# In[111]:


def evolution(d,states,positions,homes,length,limit):
    fractions=np.zeros(2)
    positions=updatePosition(homes)
    
    for i in range(len(states)):
        if states[i]==1:
            for j in range(len(states)):
                if j==i: continue
                if np.linalg.norm(positions[i,:]-positions[j,:])<d:
                    states[j]=1
    
    #healed/death
    for i in range(len(states)):
        if states[i]==1:
            length[i]+=1
        if length[i]>limit:
            states[i]=2
            
    fractions[0]= np.mean([1 if states[i]==1 else 0 for i in range(len(states))])
    fractions[1]= np.mean([1 if states[i]==0 else 0 for i in range(len(states))])
    return fractions


# In[ ]:


N=50 #grid size
d,limit=0.5,30
epochs=200
initinfections=5

homes=np.array([np.array([k%N,k//N]) for k in range(N*N)])
positions= homes+2*np.random.rand(N*N,2)-1
states=np.zeros(N*N)
length=np.zeros(N*N)

ind=np.random.randint(0,N*N,size=initinfections)
states[ind]=np.zeros(initinfections)+1
length[ind]=np.zeros(initinfections)+1

fr=[]
for i in range(epochs):
    fr.append(evolution(d,states,positions,homes,length,limit))
    if(i%50==0): print('{}%'.format(i*100/epochs))


# In[114]:


fr=np.array(fr)
t=np.arange(0,epochs)
plt.plot(t,fr[:,0]) 
plt.plot(t,fr[:,1])
plt.show()

