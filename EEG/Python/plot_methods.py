import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(x,y,title_name,sub_num,x_label,y_label, regression=True, label = None,markersize=3):
    # Make a scatter plots with the given values
    ax = plt.subplot(sub_num)
    ax.set_title(title_name)

    plt.scatter(x,y,label=label,s=markersize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    
    if regression:
        valid_inds = np.isfinite(x)
        valid_inds = list((np.where(valid_inds)[0]))
        x_fit = [x[i] for i in valid_inds]
        y_fit = [y[i] for i in valid_inds]
        b,m=np.polyfit(x_fit,y_fit,1)
    
        plt.plot(x_fit,b+m*np.array(x_fit),'-')
    if label != None:
        legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

def plot_scatters(x,y_arr,title_names,sub_nums,x_label,y_label,regression=True,labels=None,markersize=12):
    if labels ==None:
        for y,title_name,sub_num in zip(y_arr,title_names,sub_nums):
            plot_scatter(x,y,title_name,sub_num,x_label,y_label,regression,markersize=markersize)
    else:
        for y,title_name,sub_num,label in zip(y_arr,title_names,sub_nums,labels):
            plot_scatter(x,y,title_name,sub_num,x_label,y_label,regression,label=label,markersize=markersize)