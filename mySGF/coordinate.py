import numpy as np
import parameter

def norm(r,axis=None):
    return np.sqrt((r**2).sum(axis=axis))

def bottom(t1,t2,dip,w):
    e0 = (t2-t1)/norm(t2-t1)
    eb = -np.array([-e0[1],e0[0],-np.tan(dip)])
    eb = eb/norm(eb)*w
    b1 = t1+eb
    b2 = t2+eb
    return b1,b2

class Coordinate:
    def __init__(self,rtop,rbottom):
        # r[point,xyz]
        self.rtop,self.rbottom = rtop,rbottom
        xvec = self.rtop[1]-self.rtop[0]
        yvec = self.rbottom[0]-self.rtop[0]
        self.l = norm(xvec)
        self.w = norm(yvec)
        self.ex,self.ey = xvec/self.l,yvec/self.w
        pos = np.array([10000,0,0])
        x,y = np.dot(pos,self.ex),np.dot(pos,self.ey)
        zez = pos - x*self.ex - y*self.ey
        self.ez = zez/norm(zez)

    def pos(self,X,Y):  # X,Yは平面になおした座標系
        def pos(x,y):
            return self.rtop[0] + x*self.ex + y*self.ey
        try:
            n,m = len(X),len(Y)
            position = np.zeros([n,m,3])
            for i in range(n):
                for j in range(m):
                    position[i,j] = pos(X[i],Y[j])
            return position
        except TypeError:
            return pos(X,Y)

    def xyz(self,pos):
        x = np.dot(pos,self.ex)
        y = np.dot(pos,self.ey)
        z = np.dot(pos,self.ez)
        return np.array(x,y,z)

    def mesh_pos(self,x0,y0,l,n,pos_obs):
        dl = l/n
        xlist = x0 + dl/2 + np.linspace(0,l-dl,n)
        ylist = y0 + dl/2 + np.linspace(0,l-dl,n)
        pos = self.pos(xlist,ylist)
        r = norm(pos-pos_obs,axis=2)
        return pos,r

    def mesh_hypo(self,pos1,pos2,xy1,xy2,l1,l2):
        (x1,y1),(x2,y2) = xy1,xy2
        xmin = min(x1,x2)
        ymax = max(y1+l1,y2+l2)
        xy_hypo = xmin,ymax
        pos_hypo = self.pos(xmin,ymax)
        xi1 = norm(pos1-pos_hypo,axis=2)
        xi2 = norm(pos2-pos_hypo,axis=2)
        return xy_hypo,xi1,xi2


# (x,y,z)のデカルト座標系、単位はkm
# (x,y)座標は日本測地系の6系（大阪は6）での平面直角座標を採用
# zは鉛直下向きにとる
top1 = np.array([-124737,-28509,2000])*1e-3
top2 = np.array([-162511,-36189,2000])*1e-3
dip = parameter.dip  # 傾斜角
w = parameter.Wm  # 断層幅[km]
bottom1,bottom2 = bottom(top1,top2,dip,w)
c = Coordinate(np.stack([top1,top2]),np.stack([bottom1,bottom2]))

pos_obs = np.array([-159913,-36170,0])*1e-3
pos_hypo0 = np.array([-162445,-32398,11000])*1e-3
r0 = norm(pos_obs-pos_hypo0)