#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# In[40]:


N = 50
epochs=200
n=N*N

i0,r0=5,0
s0=n-i0-r0
st= 25

beta,gamma = np.linspace(0.1,0.7,st), 1./10

t=np.linspace(0,epochs,epochs)
def derivative(y,t,n,b,g):
    s,i,r = y
    ds= -b*s*i/n
    di= b*s*i/n-g*i
    dr= g*i
    return ds,di,dr

y0= s0,i0,r0
y=np.zeros((epochs,3,st))
for i in range(beta.shape[0]):
    ret= odeint(derivative,y0,t, args=(n,beta[i],gamma))
    y[:,:,i]=ret


# In[43]:


S,I,R=y[:,:,15].T

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/n, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/n, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/n, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,1.1)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# In[50]:


def updatePosition(homes):
    r=2*np.random.rand(homes.shape[0],homes.shape[1])-1
    x= homes+r
    return x
def evolution(d,states,positions,homes,length,limit):
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
    fractions=np.zeros(2)        
    fractions[0]= np.mean([1 if states[i]==1 else 0 for i in range(len(states))])
    fractions[1]= np.mean([1 if states[i]==0 else 0 for i in range(len(states))])
    return fractions


# In[51]:


N=50 #grid size
nbrs=40
d,limit=np.linspace(0.1,0.7,nbrs),30
epochs=200
initinfections=5

homes=np.array([np.array([k%N,k//N]) for k in range(N*N)])
positions= homes+2*np.random.rand(N*N,2)-1
states=np.zeros((N*N,nbrs))
length=np.zeros((N*N,nbrs))

ind=np.random.randint(0,N*N,size=initinfections)
states[ind,:]=np.zeros((initinfections,nbrs))+1
length[ind,:]=np.zeros((initinfections,nbrs))+1


# In[ ]:


fr=np.zeros((epochs,2,nbrs))
for i in range(epochs):
    positions=updatePosition(homes)
    for j in range(nbrs):
        fr[i,:,j]=evolution(d[j],states[:,j],positions,homes,length[:,j],limit)

