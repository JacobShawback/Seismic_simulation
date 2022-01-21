import numpy as np

scale_factor = 2000/8388608
csv_input = np.loadtxt('acc_before.txt')
acc = csv_input.flatten() * scale_factor
acc -= acc.mean()

print(acc.max(),acc.min(),acc.mean())

dt = 0.01
n = len(acc)
time_list = np.linspace(0,n*dt,n)
np.savetxt('acc0.txt',np.stack([time_list,acc],1),fmt='%.5e')


scale_factor = 3920/6182761
csv_input = np.loadtxt('acc_before2.txt')
acc = csv_input.flatten() * scale_factor
acc -= acc.mean()
print(acc.max(),acc.min(),acc.mean())
dt = 0.01
n = len(acc)
time_list = np.linspace(0,n*dt,n)
np.savetxt('acc02.txt',np.stack([time_list,acc],1),fmt='%.5e')