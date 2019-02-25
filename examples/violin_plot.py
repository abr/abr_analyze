import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from abr_analyze.utils import DataHandler

dat = DataHandler('dewolf2018neuromorphic')
test = 'friction_post_tuning/nengo_cpu_friction_55_0/session000/run'
colors = ['r', 'g', 'b', 'm', 'y', 'k', 'tab:grey', 'tab:orange', 'tab:red', 'tab:pink', 'tab:yellow']
for ii in range(0,50):
    tmp = dat.load(
            parameters=['input_signal'],
            save_location='%s%03d'%(test, ii))['input_signal']
    if ii == 0:
        input_signal = tmp
    else:
        input_signal = np.vstack((input_signal,tmp))
input_signal = np.array(input_signal).T

fig = plt.figure()
ax = fig.add_subplot(3,1,1)
plt.grid()
handles = []
for ii in range(0, 5):
    data = input_signal[ii]
    violin_parts = ax.violinplot(
        data, [ii],
        showmeans=True, showextrema=False, points=1000)
    violin_parts['cmeans'].set_edgecolor('black')#colors[ii])
    violin_parts['cmeans'].set_linewidth(3)
    #violin_parts['bodies'][0].set_facecolor(colors[ii])
    violin_parts['bodies'][0].set_alpha(.8)

ax = fig.add_subplot(312)
for ii in range(5, len(input_signal)):
    data = input_signal[ii]
    violin_parts = ax.violinplot(
        data, [ii],
        showmeans=True, showextrema=False, points=1000)
    violin_parts['cmeans'].set_edgecolor('black')#colors[ii])
    violin_parts['cmeans'].set_linewidth(3)
    #violin_parts['bodies'][0].set_facecolor(colors[ii])
    violin_parts['bodies'][0].set_alpha(.8)

plt.title('input_signal')


ax = fig.add_subplot(313)
mag = []
for ii, x in enumerate(input_signal.T):
    print('%.2f%% complete'%(ii/len(input_signal.T)*100), end='\r')
    mag.append(np.linalg.norm(x))
ax.plot(range(0,len(input_signal.T)), mag)
# ax.set_ylim(-1,1)
# plt.savefig('q-dq_means_and_scales.pdf')
plt.show()
