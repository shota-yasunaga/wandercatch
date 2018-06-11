import matplotlib.pyplot as plt

def plot_scatter(x,y,title_name,sub_num,x_label,y_label):
    ax = plt.subplot(sub_num)
    ax.set_title(title_name)
    plt.scatter(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    valid_inds = np.isfinite(x)
    valid_inds = list((np.where(valid_inds)[0]))
    
    x_fit = [x[i] for i in valid_inds]
    y_fit = [y[i] for i in valid_inds]
    b,m=polyfit(x_fit,y_fit,1)
    
    plt.plot(x_fit,b+m*np.array(x_fit),'-')
    

def plot_scatters(x,y_arr,title_names,sub_nums,x_label,y_label):
    for y,title_name,sub_num in zip(y_arr,title_names,sub_nums):
        plot_scatter(x,y,title_name,sub_num,x_label,y_label)