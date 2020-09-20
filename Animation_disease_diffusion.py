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

    fractions[0]= np.mean([1 if states[i]==1 else 0 for i in range(len(states))])
    fractions[1]= np.mean([1 if states[i]==0 else 0 for i in range(len(states))])
    return fractions


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

history=np.zeros((positions.shape[0],positions.shape[1],epochs))
historystates=np.zeros((states.shape[0],epochs))

fr=[]
for i in range(epochs):
    history[:,:,i]=positions
    historystates[:,i]=states
    fr.append(evolution(d,states,positions,homes,length,limit))

##

# Create new Figure and an Axes which fills it.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
N,epochs=50,200
homes=np.array([np.array([k%N,k//N]) for k in range(N*N)])
history=np.zeros((homes.shape[0],homes.shape[1],epochs))
historystates=np.zeros((N*N,epochs))


fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(0, 1))

pos=history[:,:,0]
#scat = ax.scatter(homes[:, 0]/N, homes[:, 1]/50,s=0.05)


def init():
    population=ax.scatter(pos[:,0],pos[:,1],s=0.2)
    return population,

def update(i):
    i=(i+1)%epochs
    pos=history[:,:,i]
    # sta=historystates[:,i]
    # indH=(sta==0)
    # indS=(sta==1)
    population.set_offsets(pos)
    return population,

anim = animation.FuncAnimation(fig, update, init_func=init, frames=epochs, interval=20, blit=True)


#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

#####
class animDisease(object):

    def __init__(self, N=50):
        self.N = N
        self.history = history
        self.histstates=historystates

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        pos_x, pos_y, col = self.data_stream(0).T
        self.scat = self.ax.scatter(pos_x, pos_y, c=col, s=1, vmin=0, vmax=1, cmap="jet", edgecolor="k")
        self.ax.axis([0, 1, 0, 1])
        return self.scat,

    def data_stream(self,i):
        pos = self.history[:,:,i]
        col = self.histstates[:,i].T
        return np.c_[pos[:,0], pos[:,1], col]

    def update(self, i):
        newinfo = self.data_stream(i)
        self.scat.set_offsets(newinfo[:, :2])
        self.scat.set_array(newinfo[:, 2])
        return self.scat,


a = animDisease()
plt.show()