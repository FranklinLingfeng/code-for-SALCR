import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator


metrics = ['ari', 'fmi', 'ami', 'v_measure']
metric_names = ['Adjusted Rand Index', 'Fowlkes-Mallows Index', 'Adjusted Mutual Info Score', 'V-measure Score']
plt.figure(figsize=(13, 16))
plt.subplots_adjust(wspace =0.23, hspace =0.30)
text_size=16
label_size=20
axis_size=16
color1 = '#16304B'
color2 = "#DC541B" 
x_plot, y_plot = 4, 1
linewidth = 2.5


for i in range(x_plot * y_plot):
    plt.subplot(x_plot, y_plot, i+1)

    #折线图
    x = np.array(list(range(40))) + 40
    y_base_v = np.load('hlf_ReID/USL/cluster_metric/base/' + metrics[i] + '_v.npy')
    y_base_r = np.load('hlf_ReID/USL/cluster_metric/base/' + metrics[i] + '_r.npy')
    y_final_v = np.load('hlf_ReID/USL/cluster_metric/final/' + metrics[i] + '_v.npy')
    y_final_r = np.load('hlf_ReID/USL/cluster_metric/final/' + metrics[i] + '_r.npy')

    if i <=1:
        x_major_locator=MultipleLocator(5)
        y_major_locator=MultipleLocator(0.08)
    else:
        x_major_locator=MultipleLocator(5)
        y_major_locator=MultipleLocator(0.04)

    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.plot(x,y_base_v,'--',color = color1,label="Visible pseudo-labels (DAGL)", linewidth=linewidth)#s-:方形
    plt.plot(x,y_base_r,'--',color = color2,label="Infrared pseudo-labels (DAGL)", linewidth=linewidth)#o-:圆形
    plt.plot(x,y_final_v,'-',color = color1,label="Visible pseudo-labels (SALCR)", linewidth=linewidth)#s-:方形
    plt.plot(x,y_final_r,'-',color = color2,label="Infrared pseudo-labels (SALCR)", linewidth=linewidth)#o-:圆形

    if i <= 1:
        y_min = min(y_base_v.min(), y_base_r.min(), y_final_v.min(), y_final_r.min())-0.08
        y_max = max(y_base_v.max(), y_base_r.max(), y_final_v.max(), y_final_r.max())+0.02
    else:
        y_min = min(y_base_v.min(), y_base_r.min(), y_final_v.min(), y_final_r.min())
        y_max = max(y_base_v.max(), y_base_r.max(), y_final_v.max(), y_final_r.max())+0.015

    plt.xlim((40, 80))
    plt.ylim((y_min, y_max))

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Epoch", fontsize=18)#横坐标名字
    plt.ylabel(metric_names[i], fontsize=18)#纵坐标名字
    plt.legend(loc = "lower center", fontsize=16, ncol=2)#图例
    plt.grid(ls='--',alpha=0.8, axis='y')


plt.savefig('hlf_ReID/USL/cluster_metric/plot_acc.pdf', bbox_inches='tight')


