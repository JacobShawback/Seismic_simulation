from enum import Enum,auto
import numpy as np
from numpy.core.function_base import linspace
import numpy.linalg as LA
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from myPack.constitution import Slip,Linear,Slip_Bilinear,Combined, Slip_Bilinear2, Slip_Bilinear3
from myPack.make_matrix import House,House_NL

class CType(Enum):
    LINEAR = auto()
    SLIP = auto()
    SLIP_BILINEAR = auto()


def make_gif(x,y,fig,ax,path,total_time=50):
    npage = int(total_time*5)
    ims = []
    ni = int(np.floor(len(x)/npage))
    for i in range(npage):
        start = (i-3)*ni if i-3>0 else 0
        end = i*ni
        im = None
        im = ax.plot(x[:end],y[:end],color='black',lw=0.3)
        im += ax.plot(x[start:end],y[start:end],color='red',lw=0.6)
        im += ax.plot(x[end-1],y[end-1],marker='.',markersize=5,color='red')
        ims.append(im)
    interval = total_time/npage*1000
    ani = animation.ArtistAnimation(fig,ims,interval=interval)
    ani.save(path)


# Linear
class Response:
    def __init__(self,house,wave,fname='model1'):
        self.house,self.wave = house,wave
        self.M = house.M
        self.C = house.C
        self.K = house.K
        self.F = house.F
        self.l1,self.l2=house.l
        self.acc = wave.acc0
        self.dt = wave.dt0
        self.time = wave.time0
        self.fname = fname

    def to_relative(self,x):
        y = np.zeros_like(x)
        y[0] = x[0]+x[2]+x[3]*self.l1
        y[1] = x[1]+x[2]+x[3]*self.l2
        y[2] = x[2]
        y[3] = x[3]
        return y

    def to_abs(self,x):
        y = self.to_relative(x)
        y[0:3] += self.acc
        return y

    # x: u0 to u3
    # z[0]:1st floor, z[1]:2nd floor
    def to_angle(self,x):
        y = self.to_relative(x)
        shape = list(y.shape)
        shape[0] = 2
        z = np.zeros(shape)

        z[0] = (y[1]-y[2])/self.l2
        z[1] = (y[0]-y[1])/(self.l1-self.l2)
        return z

    # def NewmarkB(self):
    #     # NEWMARK PARAMETER  --------------------------------------
    #     dof = self.M.shape[0]  # degree od freedom
    #     # beta = 1/6  # linear acceleration
    #     beta = 1/4 # average acceleration
    #     # beta = 0 # explicit acceleration
    #     dt = self.dt
    #     N = len(self.acc)  # time step

    #     # MAKE ZERO ARRAY--------------------------------------
    #     Dis = np.zeros([dof,N])
    #     Vel = np.zeros([dof,N])
    #     Acc = np.zeros([dof,N])

    #     # MAIN PART --------------------------------------
    #     for i in range(1, N):
    #         a1 = self.M + 0.5*dt*self.C + beta*dt**2*self.K
    #         a21 = -np.dot(self.F,self.acc[i])  # acc[i]はスカラーだがこれが高速かも？
    #         a22 = -np.dot(self.C,(Vel[:,i-1] + 0.5*dt*Acc[:,i-1]))
    #         a23 = -np.dot(self.K,(Dis[:,i-1] + dt*Vel[:,i-1] + (0.5-beta)*dt**2*Acc[:,i-1]))
    #         a2 = a21+a22+a23
    #         Acc[:,i] = np.dot(LA.inv(a1),a2)
    #         Vel[:,i] = Vel[:,i-1] + dt*(Acc[:,i-1] + Acc[:,i])/2
    #         Dis[:,i] = Dis[:,i-1] + dt*Vel[:,i-1] + (0.5-beta)*dt**2*Acc[:,i-1] + beta*dt**2*Acc[:,i]

    #     self.Dis = self.to_relative(Dis)
    #     self.Vel = self.to_relative(Vel)
    #     self.Acc = self.to_abs(Acc)
    #     self.u_time = [self.Dis,self.Vel,self.Acc]
    #     self.angle = self.to_angle(Dis)

    def NewmarkB(self):
        # NEWMARK PARAMETER  --------------------------------------
        a0,dt = self.acc,self.dt
        dof = self.M.shape[0]  # degree od freedom
        n = len(a0)  # time step

        # MAKE ZERO ARRAY--------------------------------------
        Dis = np.zeros([dof,n])
        Vel = np.zeros([dof,n])
        Acc = np.zeros([dof,n])

        # MAIN PART --------------------------------------
        Mbydt2,Cby2dt = self.M/dt**2,self.C/2/dt
        K_hat_inv = LA.inv(Mbydt2+Cby2dt)
        dnext = np.zeros(dof)

        k = int(n/20)
        for i in range(1,n-1):
            if i%k == 0:
                print(f'\t{int(100*i/n)}%')

            Dis[:,i] = dnext
            R = -np.dot(self.F,a0[i])
            F = np.dot(self.K,Dis[:,i])

            Rhat = R - F + np.dot(Mbydt2,2*Dis[:,i]-Dis[:,i-1]) + np.dot(Cby2dt,Dis[:,i-1])
            dnext = np.dot(K_hat_inv,Rhat)
            Acc[:,i] = (Dis[:,i-1]-2*Dis[:,i]+dnext)/dt**2
            Vel[:,i] = (-Dis[:,i-1]+dnext)/2/dt

        self.Dis = self.to_relative(Dis)
        self.Vel = self.to_relative(Vel)
        self.Acc = self.to_abs(Acc)
        self.u_time = [self.Dis,self.Vel,self.Acc]
        self.angle = self.to_angle(Dis)

    def FftMethod(self):
        # FFT PARAMETER  --------------------------------------
        dof = self.M.shape[0]  # degree od freedom
        fftAcc = fft(self.acc)
        N = len(fftAcc)  # time step
        dt = self.dt

        omega = fftfreq(N, d=dt) * 2 * np.pi

        # MAIN PART --------------------------------------
        M = self.M.flatten()[None,:]
        K = self.K.flatten()[None,:]
        C = self.C.flatten()[None,:]
        F = self.F
        P = ((omega**2*M.T).T - (omega*C.T*1j).T - K).reshape(N,dof,dof)
        Pinv = LA.inv(P)

        iomega_mat = 1j * np.eye(dof)[None,:,:] * omega[:,None,None]
        iomega2_mat = -np.eye(dof)[None,:,:] * omega[:,None,None]**2


        H_dis = Pinv @ F
        H_vel = Pinv @ iomega_mat @ F
        H_acc = Pinv @ iomega2_mat @ F

        self.H = np.abs(H_dis) * self.house.omega[0]

        U_dis = H_dis.T * fftAcc
        U_vel = H_vel.T * fftAcc
        U_acc = H_acc.T * fftAcc

        Dis = np.real(ifft(U_dis))
        Vel = np.real(ifft(U_vel))
        Acc = np.real(ifft(U_acc))

        self.Dis_f = self.to_relative(Dis)
        self.Vel_f = self.to_relative(Vel)
        self.Acc_f = self.to_abs(Acc)
        self.u_freq = [self.Dis_f,self.Vel_f,self.Acc_f]

    def plot(self):
        wave,house = self.wave,self.house
        name_tag = ['2nd floor','1st floor','Base','Rocking']
        fname_tag = ['2nd','1st','base','rock']
        unit_list = ['[m]','[m/sec]','[m/sec^2]']
        type_tag = ['Displacement','Velocity','Acceleration']
        domain = ['Time domain','Frequency domain']

        xlim = 1e-3,6e1
        Nt = len(self.acc)
        N0 = self.wave.N0
        tstart,tend = int((Nt-1.1*N0)/2),int((Nt+1.5*N0)/2)
        Nshow = tend-tstart
        for i,(name,fname) in enumerate(zip(name_tag,fname_tag)):
            # # ================ Amplitude ================
            # fig,ax = plt.subplots()
            # ax.set_title(name)
            # ax.set_xlabel('Frequency [Hz]')
            # ax.set_ylabel('Amplitude factor', labelpad=6.0)
            # ax.plot(wave.freqList[:nn],self.H[:nn,i],label=domain[0]) #label
            # ax.semilogx()
            # ax.set_xlim(*xlim)
            # # ax.semilogy()
            # fig.savefig('fig/amp/{}_{}.png'.format(self.fname,fname))
            # plt.close(fig)

            for j,(tt,unit) in enumerate(zip(type_tag,unit_list)):
                u = self.u_time[j],self.u_freq[j]
                U = [np.abs(fft(ui)) for ui in u]

                # # ================ Fourier Spectrum ================
                # fig,ax = plt.subplots()
                # ax.set_title(name)
                # ax.set_xlabel('Frequency [Hz]')
                # ax.set_ylabel('Fourier Spectrum of {}'.format(tt), labelpad=6.0)
                # ax.plot(wave.freqList[:nn],U[0][i,:nn],label=domain[0]) #label
                # ax.plot(wave.freqList[:nn],U[1][i,:nn],label=domain[1]) #label
                # ax.legend(bbox_to_anchor=(0,0),loc='lower left',borderaxespad=0)
                # ax.semilogx()
                # ax.semilogy()
                # ax.set_xlim(*xlim)
                # fig.savefig('fig/freq/{}_{}_{}.png'.format(self.fname,tt,fname))
                # plt.close(fig)

                # ================ Wave ================

                fig,ax = plt.subplots()
                ax.set_title(name)
                ax.set_xlabel('Time [sec]')
                ax.set_ylabel('{} {}'.format(tt,unit), labelpad=6.0)
                umax = [', max:{:.3g}'.format(np.abs(u[k][i]).max()) for k in range(2)]
                ax.plot(np.arange(Nshow)*wave.dt0,u[0][i,tstart:tend],label=domain[0]+umax[0]) #label
                ax.plot(np.arange(Nshow)*wave.dt0,u[1][i,tstart:tend],label=domain[1]+umax[1]) #label
                ax.legend()
                fig.savefig('fig/wave/{}_{}_{}.png'.format(self.fname,tt,fname))
                plt.close(fig)

                # # magnified wave
                # if i == 0 and j == 0:
                #     fig,ax = plt.subplots()
                #     ax.set_title(name)
                #     ax.set_xlabel('Time [sec]')
                #     ax.set_ylabel('{} {}'.format(tt,unit), labelpad=6.0)
                #     tstart,tend = int((Nt-wave.N)/2),int((Nt+wave.N)/2)
                #     umax = [', max:{:.3g}'.format(np.abs(u[k][i]).max()) for k in range(2)]
                #     ax.plot(np.arange(wave.N)*wave.dt,u[0][i,tstart:tend],label=domain[0]+umax[0]) #label
                #     ax.plot(np.arange(wave.N)*wave.dt,u[1][i,tstart:tend],label=domain[1]+umax[1]) #label
                #     ax.set_xlim(2,10)
                #     ax.set_ylim(0,0.08)
                #     fig.savefig('fig/wave/{}_{}_{}_magnify.png'.format(self.fname,tt,fname))
                #     plt.close(fig)

        # angle
        floor = ['1st floor','2nd floor']
        fig,ax = plt.subplots()
        ax.set_title('Angle')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Angle', labelpad=6.0)
        label = [floor[i]+', max:{:.3g}'.format(np.abs(self.angle[i]).max()) for i in range(2)]
        ax.plot(np.arange(Nshow)*wave.dt0,self.angle[0,tstart:tend],label=label[0])
        ax.plot(np.arange(Nshow)*wave.dt0,self.angle[1,tstart:tend],label=label[1])
        ax.plot([0,wave.N*wave.dt0],[1/120,1/120],color='red')
        ax.plot([0,wave.N*wave.dt0],[-1/120,-1/120],color='red')
        ax.legend()
        ax.set_xlim(0,wave.N*wave.dt0)
        fig.savefig('fig/wave/{}_angle.png'.format(self.fname))
        plt.close(fig)

    def DrawFRF(self, mode, fname):
        tag = ['2nd floor', '1st floor', 'base','rock']
        dof = self.M.shape[0]  # degree od freedom
        EndPeriod = 50  # frequency window
        N = len(self.acc)  # time step
        freqlist = np.linspace(0,EndPeriod,N)*2*np.pi  # circular frequency list
        A = freqlist**2

        # MAKE ZERO ARRAY --------------------------------------
        FRF = np.zeros((dof, N), dtype='c8')

        # MAIN PART --------------------------------------
        for i in range(1, N):
            M1 = -A[i] * self.M + freqlist[i] * self.C * 1j + self.K
            FRF[:, i] = -np.dot(LA.inv(-M1), self.F).T

        self.FRF = np.abs(FRF[:, :])
        plt.figure()
        plt.title(tag[mode])
        plt.xlabel('Frequency')
        plt.ylabel('Amplification factor')
        plt.plot(np.linspace(0, EndPeriod, N) * 2 * np.pi, self.FRF[mode, :])
        plt.xscale('log')
        plt.grid()
        plt.savefig(
            'fig/%s_FRF.jpeg' %
            fname,
            bbox_inches="tight",
            pad_inches=0.05)

    def FourierSpectrum(self, mode, fname):
        tag = ['2nd floor', '1st floor', 'base','rock']
        N = len(self.acc)  # time step
        nn = int(N/2-1)
        freqList = fftfreq(N, d=self.dt)
        #print(self.Dis[0, 0])
        # self.Dis[mode,1000:]=0
        self.spectrum_n = fft(self.Dis[mode])
        self.spectrum_f = fft(self.Dis_f[mode])
        #self.spectrum_sa = fft(self.Dis[mode]-self.Dis_f[mode])
        # print(self.spectrum_n[0,0])
        plt.figure()
        plt.title(tag[mode])
        plt.xlabel('Frequency[Hz]')
        plt.ylabel('Fourier spectrum of Displacement')
        plt.plot(freqList[:nn],
                 np.abs(self.spectrum_n[:nn]),
                 label='Time domain',
                 linewidth=0.3)
        plt.plot(freqList[:int(N / 2 - 1)],
                 np.abs(self.spectrum_f[:nn]),
                 label='Frequency domain',
                 linewidth=0.3)
        # plt.plot(freqList[:int(N/2-1)],np.abs(self.spectrum_sa[:int(N/2-1)]),label='sabun',linewidth=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([0, 100])
        plt.ylim([10**-7, 10**7])
        plt.legend()
        plt.grid()
        plt.savefig(
            'fig/%s_FourierSpectrum.jpeg' %
            fname,
            bbox_inches="tight",
            pad_inches=0.05)


# NL: Non-Linear
class NL_4dof(Response):
    def __init__(self, house:House_NL, wave, fname='4dof'):
        super().__init__(house, wave, fname=fname)
        self.C_linear = house.C
        self.C_nonlinear = house.Cnl
        # c,w: column, wall
        self.kc = house.kc  # kc[floor][2 for bilinear]
        self.kw = house.kw
        self.kh = house.kh
        self.kth = house.kth

        self.data_path = 'data/non-linear/'
        self.fig_path = 'fig/'

    def get_model(self,cmodel):
        if cmodel == Linear:
            k = self.house.k
            model = Combined(
                [[Linear(k[0])],[Linear(k[1])],[Linear(self.kh)],[Linear(self.kth)]]
            )
        elif cmodel == Slip_Bilinear2:
            l = np.array([self.l1-self.l2,self.l2])
            kc,kw = self.kc,self.kw
            model = Combined(
                [[Slip_Bilinear3(kc[0],l[0]),Slip_Bilinear2(kw[0],l[0])],
                [Slip_Bilinear3(kc[1],l[1]),Slip_Bilinear2(kw[1],l[1])],
                [Linear(self.kh)],
                [Linear(self.kth)]]
            )
        else:
            l = np.array([self.l1-self.l2,self.l2])
            dyield_c = l/40  # shape: floor
            dyield_w = l/250  # shape: floor
            alpha_c,alpha_w = self.house.alpha_c,self.house.alpha_w
            slip_rate_c = self.house.slip_rate_c
            slip_rate_w = self.house.slip_rate_w

            kc,kw = self.kc,self.kw
            if cmodel == Slip:
                model = Combined(
                    [[cmodel(kc[0],kc[0]*alpha_c,dyield_c[0]),cmodel(kw[0],kw[0]*alpha_w,dyield_w[0])],
                    [cmodel(kc[1],kc[1]*alpha_c,dyield_c[1]),cmodel(kw[1],kw[1]*alpha_w,dyield_w[1])],
                    [Linear(self.kh)],
                    [Linear(self.kth)]]
                )
            else:
                model = Combined(
                    [[cmodel(kc[0],alpha_c,dyield_c[0],slip_rate_c),cmodel(kw[0],alpha_w,dyield_w[0],slip_rate_w)],
                    [cmodel(kc[1],alpha_c,dyield_c[1],slip_rate_c),cmodel(kw[1],alpha_w,dyield_w[1],slip_rate_w)],
                    [Linear(self.kh)],
                    [Linear(self.kth)]]
                )
        return model

    def dis_to_x(self,dis):
        x = [dis[0]-dis[1],dis[1],dis[2],dis[3]]
        return np.array(x)

    def x_to_dis(self,x):
        dis = [x[0],x[1]-x[0],x[2],x[3]]
        return np.array(dis)

    def save_txt(self,Acc,Vel,Dis,model,cmodel):
        lnl = '_l' if cmodel==Linear else '_s' if cmodel==Slip else '_sb'
        np.savetxt(f'{self.data_path}{self.fname}{lnl}_Acc',Acc)
        np.savetxt(f'{self.data_path}{self.fname}{lnl}_Vel',Vel)
        np.savetxt(f'{self.data_path}{self.fname}{lnl}_Dis',Dis)
        np.savetxt(f'{self.data_path}{self.fname}{lnl}_x',model.X)
        np.savetxt(f'{self.data_path}{self.fname}{lnl}_f',model.F)

    def different(self,cmodel):
        a0,dt = self.acc,self.dt
        n = len(a0)
        model = self.get_model(cmodel)

        C = self.C_linear if cmodel==Linear else self.C_nonlinear
        M,K = self.M,self.K

        dof,_ = M.shape
        Acc = np.zeros([dof,n])
        Vel = np.zeros([dof,n])
        Dis = np.zeros([dof,n])

        Mbydt2,Cby2dt = M/dt**2,C/2/dt
        K_hat_inv = LA.inv(Mbydt2+Cby2dt)
        dnext = np.zeros(dof)

        k = int(n/5)
        for i in range(1,n-1):
            if i % k == 0:
                print(f'\t{int(100*i/n)}%')

            Dis[:,i] = dnext
            R = -np.dot(self.F,a0[i])
            if cmodel == Linear:
                F = np.dot(K,Dis[:,i])
            else:
                x = self.dis_to_x(Dis[:,i])
                f = model.sheer(x)
                F = self.x_to_dis(f)
                model.push()

            Rhat = R - F + np.dot(Mbydt2,2*Dis[:,i]-Dis[:,i-1]) + np.dot(Cby2dt,Dis[:,i-1])
            dnext = np.dot(K_hat_inv,Rhat)
            Acc[:,i] = (Dis[:,i-1]-2*Dis[:,i]+dnext)/dt**2
            Vel[:,i] = (-Dis[:,i-1]+dnext)/2/dt
        self.save_txt(Acc,Vel,Dis,model,cmodel)

    def newmark(self,cmodel):
        def newmark(a1,a2,v,d,dt,beta):
            v_next = v + dt*(a1+a2)/2
            d_next = d + dt*v + ((0.5-beta)*a1+beta*a2)*dt**2
            return v_next,d_next

        small = 1e-8
        a0,dt = self.acc,self.dt
        n = len(a0)
        model = self.get_model(cmodel)
        C = self.C_linear if cmodel==Linear else self.C_nonlinear
        M,K = self.M,self.K

        beta = 1/6
        dof,_ = self.M.shape
        Acc = np.zeros([dof,n])
        Vel = np.zeros([dof,n])
        Dis = np.zeros([dof,n])

        k = int(n/20)
        for i in range(n-1):
            if i % k == 0:
                print(f'\t{int(100*i/n)}/100')

            A1 = Acc[:,i]
            V1,D1 = newmark(A1,A1,Vel[:,i],Dis[:,i],dt,beta)
            # x: バネの伸び
            x = self.dis_to_x(D1)
            # kx: バネの反力
            kx = model.sheer(x)
            # Kx： 運動方程式における剛性の項（ベクトル）
            Kx = self.x_to_dis(kx)
            Fa0 = np.dot(self.F,a0[i+1])
            A2 = np.dot(LA.inv(M),np.dot(-C,V1)-Kx-Fa0)
            delta = np.abs(A2-A1)
            if (delta<small).sum() == dof:
                model.push()
                Acc[:,i+1] = A2
                Vel[:,i+1] = V1
                Dis[:,i+1] = D1
            else:
                A3 = A1
                while (delta<small).sum() < dof:
                    V3,D3 = newmark(Acc[:,i],A3,Vel[:,i],Dis[:,i],dt,beta)
                    # x: バネの伸び
                    x = self.dis_to_x(D3)
                    # kx: バネの反力
                    kx = model.sheer(x)
                    # Kx： 運動方程式における剛性の項（ベクトル）
                    Kx = self.x_to_dis(kx)
                    A4 = np.dot(LA.inv(M),np.dot(-C,V3)-Kx-Fa0)
                    delta = np.abs(A4-A3)
                    A3 = A4 + 0.99*(A3-A4)
                model.push()
                Acc[:,i+1] = A4
                Vel[:,i+1] = V3
                Dis[:,i+1] = D3
        self.save_txt(Acc,Vel,Dis,model,cmodel)

    def get_names(self):
        floor_name = ['2nd floor','1st floor','Sway','Rocking']
        floor_file = ['2f','1f','sway','rock']
        unit_list = ['[m]','[m/sec]','[m/sec^2]']
        type_tag = ['Displacement','Velocity','Acceleration']
        models = ['2f column','2f wall','1f column','1f wall','Sway','Rocking']
        return floor_name,floor_file,unit_list,type_tag,models

    def get_u_angle(self,cmodel):
        lnl = '_l' if cmodel==Linear else '_s' if cmodel==Slip else '_sb'

        # Acc.shape: dof,nstep
        Dis = np.loadtxt(f'{self.data_path}{self.fname}{lnl}_Dis')
        Vel = np.loadtxt(f'{self.data_path}{self.fname}{lnl}_Vel')
        Acc = np.loadtxt(f'{self.data_path}{self.fname}{lnl}_Acc')

        angle = self.to_angle(Dis)
        Dis = self.to_relative(Dis)
        Vel = self.to_relative(Vel)
        Acc = self.to_abs(Acc)
        u = Dis,Vel,Acc
        return {'u':u,'angle':angle}

    def plot(self,cmodel,title='',second=None,gif=False):
        floor_name,floor_file,unit_list,type_tag,models = self.get_names()
        lnl = '_l' if cmodel==Linear else '_s' if cmodel==Slip else '_sb'

        u_angle = self.get_u_angle(cmodel)
        u_list,angle = u_angle['u'],u_angle['angle']
        Dis,Vel,Acc = u_list
        if second is not None:
            u_list2,angle2 = second['u'],second['angle']
            tag1,tag2 = second['tag1'],second['tag2']
        else:
            tag1,tag2 = '',''

        Nt = len(self.acc)
        N0 = self.wave.N0
        tstart,tend = int((Nt-1.1*N0)/2),int((Nt+1*N0)/2)
        ymergin = 1.1
        for i,(name,fname) in enumerate(zip(floor_name,floor_file)):
            for j,(tt,unit) in enumerate(zip(type_tag,unit_list)):
                u = u_list[j]
                if second is not None:
                    u2 = u_list2[j]

                # ================ Wave ================
                fig,ax = plt.subplots()
                ax.set_title(title+name)
                ax.set_xlabel('Time [sec]')
                ax.set_ylabel('{} {}'.format(tt,unit), labelpad=6.0)
                umax = 'max:{:.3g}'.format(np.abs(u[i]).max())
                ax.plot(self.time[tstart:tend],u[i,tstart:tend],label=tag1+umax) #label
                ymax = np.abs(u[i]).max() * ymergin
                if second is not None:
                    umax2 = 'max:{:.3g}'.format(np.abs(u2[i]).max())
                    ax.plot(self.time[tstart:tend],u2[i,tstart:tend],label=tag2+umax2) #label
                    ymax2 = np.abs(u[i]).max() * ymergin
                    ymax = max(ymax,ymax2)
                ax.set_xlim(self.time[tstart],self.time[tend])
                ax.set_ylim(-ymax,ymax)
                ax.legend()
                if second is not None:
                    fig.savefig(f'{self.fig_path}wave/compare{lnl}_{tt}_{fname}.png')
                else:
                    fig.savefig(f'{self.fig_path}wave/{self.fname}{lnl}_{tt}_{fname}.png')
                plt.close(fig)

        # =============== Angle ===============
        floor = ['1st floor','2nd floor']
        fig,ax = plt.subplots()
        ax.set_title('Angle')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Angle', labelpad=6.0)
        label = [floor[i]+', max: {:.3g}'.format(np.abs(angle[i]).max()) for i in range(2)]
        ax.plot(self.time[tstart:tend],angle[0,tstart:tend],label=label[0])
        ax.plot(self.time[tstart:tend],angle[1,tstart:tend],label=label[1])
        ax.plot([self.time[tstart],self.time[tend]],[1/120,1/120],color='black')
        ax.plot([self.time[tstart],self.time[tend]],[-1/120,-1/120],color='black')
        ax.plot([self.time[tstart],self.time[tend]],[1/40,1/40],color='red')
        ax.plot([self.time[tstart],self.time[tend]],[-1/40,-1/40],color='red')
        ax.legend()
        ax.set_xlim(self.time[tstart],self.time[tend])
        ymax = np.abs(angle).max() * ymergin
        ymax = max(ymax,1/40*ymergin)
        ax.set_ylim(-ymax,ymax)
        fig.savefig(f'{self.fig_path}wave/{self.fname}{lnl}_Angle.png')
        plt.close(fig)

        # =============== Mode ================
        tmax = np.argmax(np.abs(angle[0]))
        fig,ax = plt.subplots(figsize=[2,3])
        mode = Dis[:3,tmax]
        xmax = np.abs(mode).max() * ymergin
        ax.set_xlim(-xmax,xmax)
        ax.set_ylim(2.2,-0.2)
        y_tics = [0,1,2]
        ax.plot(mode,y_tics,lw=1)
        plt.yticks(y_tics,[r'$x_1$',r'$x_2$',r'$y$'])
        fig.savefig(f'{self.fig_path}{self.fname}{lnl}_mode.png')

        # =============== Constitution ===============
        gstart,gend = int((Nt-N0)/2),int((Nt+0*N0)/2)
        gtime = self.time[gend]-self.time[gstart]
        if cmodel != Linear:
            x = np.loadtxt(f'{self.data_path}{self.fname}{lnl}_x')
            f = np.loadtxt(f'{self.data_path}{self.fname}{lnl}_f')

            if gif:
                fig,ax = plt.subplots(figsize=(5,1))
                # ax.set_title('Input acc')
                ymax = np.abs(self.acc).max()*ymergin
                ax.set_ylim(-ymax,ymax)
                ax.set_ylabel('[N/sec^2]')
                fname = f'{self.fig_path}constitution/acc.gif'
                make_gif(self.time[gstart:gend],self.acc[gstart:gend],fig,ax,fname,gtime*2)
            for i,m in enumerate(models):
                fig,ax = plt.subplots(figsize=(3,3))
                # ax.set_title('elongation - force')
                ax.set_xlabel('Elongation of spring [m]')
                ax.set_ylabel('Reaction force [N]', labelpad=6.0)
                ax.plot(x[i],f[i],label=m)
                ax.legend(bbox_to_anchor=(1,0),loc='lower right',borderaxespad=0)
                fname = f'{self.fig_path}constitution/{self.fname}{lnl}_{i+1}'
                fig.savefig(fname+'.png')
                plt.close(fig)
                if gif and (i==1 or i==3):
                    fig,ax = plt.subplots(figsize=(3,3))
                    ax.set_title(m)
                    ax.set_xlabel('Elongation of spring [m]')
                    ax.set_ylabel('Reaction force [N]', labelpad=6.0)
                    make_gif(x[i][gstart:gend],f[i][gstart:gend],fig,ax,fname+'.gif',gtime*2)




class NL_2dof(NL_4dof):
    def __init__(self, house: House_NL, wave, fname='2dof'):
        super().__init__(house, wave, fname=fname)
        self.M = self.M[0:2,0:2]
        self.C = self.C[0:2,0:2]
        self.K = self.K[0:2,0:2]
        self.F = self.F[0:2]

    def get_model(self,cmodel):
        if cmodel == Linear:
            k = self.house.k
            model = Combined(
                [[Linear(k[0])],[Linear(k[1])],[Linear(self.kh)],[Linear(self.kth)]]
            )
        else:
            l = np.array([self.l1-self.l2,self.l2])
            dyield_c = l/40  # shape: floor
            dyield_w = l/250  # shape: floor
            alpha_c,alpha_w = self.house.alpha_c,self.house.alpha_w
            slip_rate_c = self.house.slip_rate_c
            slip_rate_w = self.house.slip_rate_w

            kc,kw = self.kc,self.kw
            if cmodel == Slip:
                model = Combined(
                    [[cmodel(kc[0],kc[0]*alpha_c,dyield_c[0]),cmodel(kw[0],kw[0]*alpha_w,dyield_w[0])],
                    [cmodel(kc[1],kc[1]*alpha_c,dyield_c[1]),cmodel(kw[1],kw[1]*alpha_w,dyield_w[1])],
                    [Linear(self.kh)],
                    [Linear(self.kth)]]
                )
            else:
                model = Combined(
                    [[cmodel(kc[0],alpha_c,dyield_c[0],slip_rate_c),cmodel(kw[0],alpha_w,dyield_w[0],slip_rate_w)],
                    [cmodel(kc[1],alpha_c,dyield_c[1],slip_rate_c),cmodel(kw[1],alpha_w,dyield_w[1],slip_rate_w)],
                    [Linear(self.kh)],
                    [Linear(self.kth)]]
                )
        return model

    def dis_to_x(self, dis):
        x = [dis[0]-dis[1],dis[1]]
        return np.array(x)

    def x_to_dis(self, x):
        dis = [x[0],x[1]-x[0]]
        return np.array(dis)

    def to_relative(self,x):
        return x

    def to_abs(self,x):
        y = self.to_relative(x)
        return y + self.acc

    def to_angle(self, x):
        y = self.to_relative(x)
        z = np.zeros_like(y)

        z[0] = y[1]/self.l2
        z[1] = (y[0]-y[1])/(self.l1-self.l2)
        return z

    def get_names(self):
        floor_name = ['2nd floor','1st floor']
        floor_file = ['2nd','1st']
        unit_list = ['[m]','[m/sec]','[m/sec^2]']
        type_tag = ['Displacement','Velocity','Acceleration']
        models = ['2f column','2f wall','1f column','1f wall']
        return floor_name,floor_file,unit_list,type_tag,models