import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from myPack.constitution import *
from myPack.output_format import Format
Format.params()


ndiv = 1000
total_time = 20 + 1
nseq = total_time * ndiv
amp = 0.1
x = np.zeros(nseq)
# points = np.random.normal(0,1,2*total_time)
points = np.random.rand(2*total_time)*2 - 1
points[0] = 0
points = points / np.max(np.abs(points))
div = math.floor(ndiv/2)
for n in range(2*total_time-2):
    x[ndiv+n*div:ndiv+(n+1)*div] = np.linspace(points[n],points[n+1],div)
x *= amp


model = Slip_Bilinear2(2000e3,3)
for i in range(nseq):
    model.shear(x[i])
    model.push()

fig,ax = plt.subplots(figsize=(5,6))
ax.plot(x,model.slip.F,label='slip') #label
ax.plot(x,model.bilinear.F,label='bilinear') #label
ax.plot(x,model.F,label='slip-bilinear') #label
ax.grid()
ax.legend(bbox_to_anchor=(1,0),loc='lower right',borderaxespad=0)
fig.savefig('constitution.png')
plt.close(fig)


# fig = plt.figure(figsize=(5,6))
# ims = []
# k = 500
# y = model.F
# ni = int(np.floor(nseq/k))
# for i in range(k):
#     start = (i-3)*ni if i-3>0 else 0
#     end = i*ni
#     im1 = plt.plot(x[:end],y[:end],color='black',lw=1)
#     im2 = plt.plot(x[start:end],y[start:end],color='red',lw=2)
#     im3 = plt.plot(x[end],y[end],marker='.',markersize=10,color='red')

#     ims.append(im1+im2+im3)

# ani = animation.ArtistAnimation(fig,ims,interval=20)
# ani.save('constitution.gif')
