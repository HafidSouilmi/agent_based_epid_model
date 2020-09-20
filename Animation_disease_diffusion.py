import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def updatePosition(homes):
    r=2*np.random.rand(homes.shape[0],homes.shape[1])-1
    x= homes+r
    return x
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
    return positions,states,length


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

#####
class animMovementIndivid(object):

    def __init__(self, N=50):
        self.N = N

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=200,frames=epochs, init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        pos_x, pos_y= updatePosition(homes).T
        self.scat = self.ax.scatter(pos_x, pos_y,s=1)
        self.ax.axis([0, 50, 0,50])
        return self.scat,


    def update(self, i):
        self.scat.set_offsets(updatePosition(homes))
        #self.scat.set_array(newinfo[:, 2])
        return self.scat,


a = animMovementIndivid()
plt.show()

#####
class animDisease(object):

    def __init__(self, N=50):
        self.N = N
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=0.1,frames=epochs, init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        self.ax.axis([0, 50, 0, 50])
        self.scat = self.ax.scatter(positions[:,0],positions[:,1],s=1)

        return self.scat,


    def update(self, i):
        pos,sta,len=evolution(d,states,positions,homes,length,limit)
        self.scat.set_offsets(pos)
        self.scat.set_array(states)
        return self.scat,


a = animDisease()
plt.show()