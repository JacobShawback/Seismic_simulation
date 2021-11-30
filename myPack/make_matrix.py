# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

class House():
    def __init__(self,m1,m2,mf,If,k1,k2,kh,kth,l1,l2,h1=0.05,h2=0.2,cf=None,cth=None):
        self.m = [m1,m2,mf,If]
        self.k = [k1,k2,kh,kth]
        self.l = [l1,l2]
        self.h = [h1,h2]
        self.l1,self.l2 = l1,l2
        self.kh,self.kth = kh,kth
        self.dof = 4
        M = np.array([[m1,0,m1,m1*l1],
                    [0,m2,m2,m2*l2],
                    [m1,m2,mf+m1+m2,m2*l2+m1*l1],
                    [m1*l1,m2*l2,m1*l1+m2*l2,If+m1*l1**2+m2*l2**2]])

        K = np.array([[k1,-k1,0,0],
                    [-k1,k1+k2,0,0],
                    [0,0,kh,0],
                    [0,0,0,kth]])
        self.M,self.K = M,K
        self.F = np.array([m1,m2,mf+m1+m2,m1*l1+m2*l2])

        # EIGENVALUE ANALYSIS ---------------------
        m,k = M[0:2,0:2],K[0:2,0:2]
        w,v = LA.eig(np.dot(LA.inv(m),k))
        omega = np.sqrt(w)
        period = 2*np.pi/omega
        print('period, 2dof',period)
        eigen_vec = v
        beta = 2*h1/omega.sum()
        alpha = beta*omega.prod()
        C1 = alpha*m + beta*k
        print(m)
        print(k)
        print(C1)

        # print('omega of top:',omega[0])
        # print('period of top:',2*np.pi/omega[0])

        m_all = M[2,2]
        I_all = M[3,3]
        omega_f = np.sqrt(kh/m_all)
        omega_th = np.sqrt(kth/I_all)
        C2 = 2*omega_f*h2*m_all
        C3 = 2*omega_th*h2*I_all
        if cf is not None:
            C2,C3 = cf,cth
        # print('omega of base:',omega_f,omega_th)
        # print('period of base:',2*np.pi/omega_f,2*np.pi/omega_th)

        self.C = np.array(
            [[C1[0,0],C1[0,1],0,0],
            [C1[1,0],C1[1,1],0,0],
            [0,0,C2,0],
            [0,0,0,C3]]
        )
        # self.C = alpha[0]*M + beta[0]*K

        w,v = LA.eig(np.dot(LA.inv(M),K))
        self.omega = np.sqrt(w)
        print('period, 4dof',2*np.pi/self.omega)
        self.modal_vec = v
        # print('omega:',self.omega)
        # print('period:',2*np.pi/self.omega)

        # print('mode shape:',v)
        # print('\n\n')

        fig,ax = plt.subplots(figsize=[3,5])
        # ax.set_xlabel()
        # ax.set_ylabel(, labelpad=4.0)
        y_tics = [0,1,2,3,4]
        y_tics = [0,1,2,3]
        mode = [1,2,3,0]
        for i in range(4):
            vv = self.to_relative(v[mode[i]])
            # if vv.max() > -vv.min():
            #     mv = vv.max()
            # else:
            #     mv = vv.min()
            mv = 1
            vv = np.concatenate([vv[:3],[0]]) / mv
            ax.plot(vv,y_tics,label='mode: {}'.format(i+1)) #label
        ax.legend(bbox_to_anchor=(1,0),loc='lower right')
        # plt.yticks(y_tics,[r'$x_1$',r'$x_2$',r'$y$',r'$\theta$','ground'])
        plt.yticks(y_tics,[r'$x_1$',r'$x_2$',r'$y$','ground'])
        # ax.set_ylim(4.2,-0.2)
        ax.set_ylim(3.2,-0.2)
        # ax.set_xlim(-1,1.3)
        # ax.semilogx()
        fig.savefig('fig/mode_shape.png')
        plt.close(fig)

    def to_relative(self,x):
        y = np.zeros_like(x)
        y[0] = x[0]+x[2]+x[3]*self.l1
        y[1] = x[1]+x[2]+x[3]*self.l2
        y[2] = x[2]
        y[3] = x[3]
        return y

    def to_abs(self,x):
        y = self.to_relative(x)
        y[0:3] += self.wave.accAfter
        return y

class House_NL(House):
    def __init__(self,m1,m2,mf,If,kc,kw,kh,kth,l1,l2,h1=0.02,h2=0.2,cf=None,cth=None,alpha_c=0.05,alpha_w=0.05,slip_rate_c=0.85,slip_rate_w=0.85):
        k1 = kc[0]+kw[0]
        k2 = kc[1]+kw[1]
        self.alpha_c,self.alpha_w = alpha_c,alpha_w
        self.slip_rate_c,self.slip_rate_w = slip_rate_c,slip_rate_w
        self.kc,self.kw = kc,kw
        self.k = np.array([k1,k2])
        super().__init__(m1,m2,mf,If,k1,k2,kh,kth,l1,l2,h1=h1,h2=h2,cf=cf,cth=cth)
