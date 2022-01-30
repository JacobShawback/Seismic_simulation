import numpy as np

def cut_filter(wave,dt,low=0.2):
    high = 1/dt/2
    w = np.fft.fft(wave)
    freq = np.fft.fftfreq(len(wave),d=dt)
    df = freq[1] - freq[0]

    nt = 10
    low0  = max(0,low - nt*df)
    high0 = high + nt*df

    w2 = np.ones_like(w)
    for (i,f) in enumerate(freq):
        if abs(f) < low0:
            w2[i] = 0.0 + 0.0j
        elif abs(f) < low:
            w2[i] = w[i] * (abs(f)-low0)/(low-low0)
        elif abs(f) <= high:
            w2[i] = w[i]
        elif abs(f) <= high0:
            w2[i] = w[i] * (abs(f)-high0)/(high-high0)
        else:
            w2[i] = 0.0 + 0.0j

    wave2 = np.real(np.fft.ifft(w2))
    return wave2 - wave2.mean()

def integrate(acc,dt=0.01,beta=1/6):
    n = len(acc)
    vel = np.zeros(n)
    dis = np.zeros(n)

    for i in range(1,n):
        vel[i] = vel[i-1] + dt*(acc[i-1]+acc[i])/2
        dis[i] = dis[i-1] + dt*vel[i-1] + (0.5-beta)*dt**2*acc[i-1] + beta*dt**2*acc[i]
    return dis,vel

def crac(acc0,dt,):
    n = len(acc0)
    acc0 = acc0 - acc0[0]
    dis0,vel0 = integrate(acc0,dt)
    T = n*dt

    a1_int = 0
    for i in range(n):
        t = i*dt
        a1_int += vel0[i]*(3*T*t**2 - 2*t**3)*dt

    a1 = 28/13/T**2 * (2*vel0[-1] - 15/T**5*a1_int)
    a0 = vel0[-1]/T - a1*T/2
    t = np.linspace(0,n*dt,n)

    dis = dis0 - (0.5*a0*t**2 + 1/6*a1*t**3)
    vel = vel0 - (a0*t + 0.5*a1*t**2)
    acc = acc0 - (a0 + a1)
    return dis,vel,acc


dt = 0.01
scale_factor = 2000/8388608
csv_input = np.loadtxt('acc_before.txt')
acc0 = csv_input.flatten() * scale_factor
acc0 = cut_filter(acc0,dt,low=0.2)

n = len(acc0)
time_list = np.linspace(0,n*dt,n)
dis,vel,acc = crac(acc0,dt)

np.savetxt('acc0.txt',np.stack([time_list,acc],1),fmt='%.5e')
