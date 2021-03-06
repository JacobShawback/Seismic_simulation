import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
from myPack.response_method import Response

def integration(wave,dt,low=0.2,high=50):
    high = 1/dt/2
    w = np.fft.fft(wave)
    freq = np.fft.fftfreq(len(wave),d=dt)
    df = freq[1] - freq[0]

    nt = 10
    low0  = max(0,low - nt*df)
    high0 = high + nt*df

    w2 = np.ones_like(w)
    for (i,f) in enumerate(freq):
        coef = (0.0+1.0j)*2.0*np.pi*f
        if np.abs(coef) <= 0.0:
            w2[i] = 0.0j
        elif abs(f) < low0:
            w2[i] = 0.0 + 0.0j
        elif abs(f) < low:
            w2[i] = w[i] * (abs(f)-low0)/(low-low0) / coef
        elif abs(f) <= high:
            w2[i] = w[i] / coef
        elif abs(f) <= high0:
            w2[i] = w[i] * (abs(f)-high0)/(high-high0) / coef
        else:
            w2[i] = 0.0 + 0.0j

    wave_int = np.real(np.fft.ifft(w2))
    return wave_int

class Data(object):
    def __init__(self,path,dt=0.01,fname='eathquake1',div=1):
        csv_input = np.loadtxt(path)
        # print(csv_input.shape)
        # print(csv_input[:5])
        self.time = csv_input[:,0]
        self.acc = csv_input[:,1] * 1e-2
        self.acc -= self.acc.mean()
        self.dt = dt
        self.N = len(self.acc)
        if self.N%2 == 1:
            self.time = self.time[:-1]
            self.acc = self.acc[:-1]
            self.N -= 1
        self.fname = fname

        self.time0 = self.time[:]
        self.acc0 = self.acc[:]
        self.dt0 = dt
        self.N0 = self.N

        self.Process()
        if div > 1:
            self.Interpolation(div)

    def Process(self):
        n = 5
        # while 2**(n-3)<self.N:
        while 2**(n-1)<self.N:
            n += 1
        acc0 = self.acc
        zeros = np.zeros(2**(n-1)-int(self.N/2))
        acc0 = np.concatenate([zeros,acc0,zeros])
        t0 = self.time0[0] - self.dt0*len(zeros)
        tend = self.time0[-1] + self.dt0*len(zeros)
        self.time0 = np.linspace(t0,tend,len(acc0))
        self.freqList = fftfreq(len(acc0), d=self.dt)
        self.period = 2*np.pi/self.freqList
        self.fftAcc = fft(acc0)
        self.fftAccAfter = np.copy(self.fftAcc)
        fs = 0.2
        fn = (1/self.dt)/2
        # self.fftAccAfter[np.abs(self.freqList) < fs] = 0  # ??????????????????????????????
        # self.fftAccAfter[np.abs(self.freqList) > fn] = 0  # ??????????????????????????????????????????
        accAfter = np.real(ifft(self.fftAccAfter))
        self.acc0 = accAfter

    def Interpolation(self,div=10,to_acc0=True):
        self.dtInterpolated = self.dt / div
        n = len(self.acc0)
        acc = np.zeros((div*n))
        time = np.linspace(self.time0[0],self.time0[-1],div*n)

        for i in range(n-1):
            a,b = self.acc0[i:i+2]
            for j in range(div):
                acc[div*(i-1)+j] = a+(b-a)*(j-1)/div
        self.accInterpolated = acc
        if to_acc0:
            self.time0 = time
            self.acc0 = acc
            self.dt0 = self.dtInterpolated
            self.N0 = self.N * div

    def Output(self,other_acc=None,other_dt=0.01,other_tag='before'):
        N = len(self.acc0)
        freqList = fftfreq(N, d=self.dt0)

        plt.figure(figsize=[4,2])
        plt.plot(self.time0, self.acc0,label='after')
        if other_acc is not None:
            plt.plot(self.time0, other_acc,label=other_tag)
            plt.legend()
        plt.xlabel('Time [sec]')
        plt.ylabel(r'acceleration [m/sec^2]')
        plt.xlim(0,self.time[-1])
        plt.xlim([10,60])
        # plt.grid()
        plt.savefig(
            'fig/%s_input_acc.png' %
            self.fname,
            bbox_inches="tight",
            pad_inches=0.05)

        plt.figure(figsize=[4,2])
        v = integration(self.acc0,self.dt0)
        plt.plot(self.time0, v,label='after')
        if other_acc is not None:
            other_v = integration(other_acc,other_dt)
            plt.plot(self.time0, other_v,label=other_tag)
            plt.legend()
        plt.xlabel('Time [sec]')
        plt.ylabel(r'velocity [m/sec]')
        plt.xlim(0,self.time[-1])
        plt.xlim([10,60])
        # plt.grid()
        plt.savefig(
            'fig/%s_input_vel.png' %
            self.fname,
            bbox_inches="tight",
            pad_inches=0.05)

        plt.figure(figsize=[4,2])
        d = integration(v,self.dt0)
        plt.plot(self.time0, d,label='after')
        if other_acc is not None:
            other_d = integration(other_v,other_dt,other_dt)
            plt.plot(self.time0, other_d,label=other_tag)
            plt.legend()
        plt.xlabel('Time [sec]')
        plt.ylabel(r'displacement [m]')
        plt.xlim(0,self.time[-1])
        plt.xlim([10,60])
        # plt.grid()
        plt.savefig(
            'fig/%s_input_disp.png' %
            self.fname,
            bbox_inches="tight",
            pad_inches=0.05)


        plt.figure(figsize=[3,3])
        plt.plot(freqList[:int(N / 2 - 1)],
            np.abs(self.fftAccAfter[:int(N / 2 - 1)]))
        if other_acc is not None:
            plt.plot(freqList[:int(N / 2 - 1)],
                np.abs(fft(other_acc)[:int(N / 2 - 1)]))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Fourier spectrum of acceleration')
        plt.xlim([5e-1,5e1])
        plt.ylim([1e-4,1e4])
        # plt.grid()
        plt.savefig('fig/%s_inputacc_fourierSpectrum.png' %
                    self.fname, bbox_inches="tight", pad_inches=0.05)

        plt.figure(figsize=[3,3])
        plt.plot(freqList[:int(N / 2 - 1)],
            np.abs(self.fftAccAfter[:int(N / 2 - 1)]),label='before')
        if other_acc is not None:
            plt.plot(freqList[:int(N / 2 - 1)],
                np.abs(fft(other_acc)[:int(N / 2 - 1)]),label=other_tag)
            plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Fourier spectrum of acceleration')
        plt.xlim([5e-1,5e1])
        plt.ylim([1e-4,1e4])
        # plt.grid()
        plt.savefig('fig/%s_inputacc_fourierSpectrum.png' %
                    self.fname, bbox_inches="tight", pad_inches=0.05)

        if other_acc is not None:
            plt.figure(figsize=[3,3])
            y1 = np.abs(self.fftAccAfter[:int(N / 2 - 1)])
            y2 = np.abs(fft(other_acc)[:int(N / 2 - 1)])
            plt.plot(freqList[:int(N / 2 - 1)],y1/y2)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Fourier spectrum of acceleration')
            plt.xlim([5e-1,5e1])
            # plt.ylim([1e-4,1e4])
            # plt.grid()
            plt.savefig('fig/%s_inputacc_processed_fourierSpectrum.png' %
                        self.fname, bbox_inches="tight", pad_inches=0.05)

    def ResponseSpectrum(self,MaxDis2=None):
        m = 1
        h = 0.02
        acc = self.acc0
        NT = 1000
        N = len(acc)
        dt = 0.01
        beta = 1 / 6
        MaxDis = []
        MaxVel = []
        MaxAcc = []
        Dis = np.zeros((N))
        Vel = np.zeros((N))
        Acc = np.zeros((N))
        Zacc = np.zeros((N))
        for T in np.linspace(0.01, 10, NT):
            w = 2 * np.pi / T
            k = m * w**2
            c = 2 * m * w * h
            for i in range(1, N):
                Acc1 = m + 0.5 * dt * c + beta * dt * dt * k
                Acc2 = -m * acc[i] - c * (Vel[i - 1] + 0.5 * dt * Acc[i - 1]) - k * (
                    Dis[i - 1] + dt * Vel[i - 1] + (0.5 - beta) * dt * dt * Acc[i - 1])
                Acc[i] = Acc2 / Acc1
                Vel[i] = Vel[i - 1] + dt * (Acc[i - 1] + Acc[i]) / 2
                Dis[i] = Dis[i - 1] + dt * Vel[i - 1] + \
                    (0.5 - beta) * dt * dt * Acc[i - 1] + beta * dt * dt * Acc[i]
            Zacc = Acc + acc
            MaxDis.append(np.max(np.abs(Dis)))
            MaxVel.append(np.max(np.abs(Vel)))
            MaxAcc.append(np.max(np.abs(Zacc)))
        print(np.max(np.abs(Dis)))
        plt.figure()
        plt.plot(np.linspace(0.01, 10, NT), MaxAcc)
        plt.title(r'2%damped')
        plt.xlabel('Period [sec]')
        plt.ylabel(r'Absolute acceleration response spectrum [m/sec^2]')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which='major')
        plt.grid(which='minor', linestyle='dotted')
        plt.savefig(
            'fig/%s_input_Sa.png'%self.fname,
            bbox_inches="tight",
            pad_inches=0.05)

        plt.figure()
        plt.plot(np.linspace(0.01, 10, NT), MaxVel)
        plt.title(r'2%damped')
        plt.xlabel('Period [sec]')
        plt.ylabel('Velocity response spectrum [m/sec]')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which='major')
        plt.grid(which='minor', linestyle='dotted')
        plt.savefig(
            'fig/%s_input_Sv.png' %
            self.fname,
            bbox_inches="tight",
            pad_inches=0.05)

        plt.figure()
        plt.plot(np.linspace(0.01, 10, NT), MaxDis)
        if MaxDis2 is not None:
            plt.plot(np.linspace(0.01, 10, NT), MaxDis2)
        plt.title(r'2%damped')
        plt.xlabel('Period [sec]')
        plt.ylabel('Displacement response spectrum [m]')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which='major')
        plt.grid(which='minor', linestyle='dotted')
        plt.savefig(
            'fig/%s_input_Sd.png' %
            self.fname,
            bbox_inches="tight",
            pad_inches=0.05)

        return MaxDis
