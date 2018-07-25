import os
from mat2python import getFreqValuesVec,getChanLocs, getNumLabels,folderIterator,readOneFeatures,getSubsampledFeatures
import numpy as np
import matplotlib.pyplot as plt


def normalize_power(power):
    ''' normalize the power (calculate the relative power against the mean across frequency)
    input: numpy nd array (1,2,3) If 2, it should be (~, frequency), if 3,(~,~,frequency)
    output: numpy nd array with the same shape
    '''
    print(power.shape)
    if power.ndim == 1:
        power /= np.mean(power)
    elif power.ndim == 2:
        for i in range(power.shape[0]):
            power[i,:] = power[i,:]/np.mean(power[i,:])
    elif power.ndim == 3:
        for i in range(power.shape[0]):
            for j in range(power.shape[1]):
                power[i,j,:] /= np.mean(power[i,j,:])

        # original_shape = power.shape
        # power = power.reshape((-1,power.shape[-1]))
        # print(power.shape)
        # power = np.transpose(power)
        # print(np.mean(power,axis=0).shape)
        # power = power /np.mean(power,axis = 0)

        # print(power.shape)
        # power = np.transpose(power)
        # power = power.reshape(original_shape)
        # print(power.shape)

    return power

def plot_cond(conds_dir,cond,loc_path,color,label_itr,end_freq=30):
    '''
    I thin this is going to be depriciated
    '''
    freq_path = conds_dir +'/' +cond
    freq_dir = os.fsencode(freq_path)
    loc_dir  = os.fsencode(loc_path)
    subplot_values = 1
    for files in zip(os.listdir(freq_dir),os.listdir(loc_dir)):
        print(files) # Sanity Check. 
        freq_var = os.path.join(freq_dir, files[0])
        loc_var  = os.path.join(loc_dir, files[1])
        plt.subplot(4,6,subplot_values)
        power,spectrum = getFreqValuesVec(freq_var)
        chanlocs = getChanLocs(loc_var)
        label_file = next(label_itr)
        if ('Oz' in chanlocs):
            oz_ind = chanlocs.index('Oz')
            spectrum = list(np.array(spectrum).flatten())
            last_ind = spectrum.index(end_freq) # TODO: escape when there is no such value
            
            labels_count = getNumLabels(label_file)  # Make it if statement
            plt.plot(spectrum[0:last_ind], power[oz_ind][0:last_ind],color,label=cond+':'+str(labels_count[cond])+'Epochs')
            plt.legend(loc='upper right', fontsize='smaller')
            freq_var = freq_var.decode("utf-8")
            plt.title('Participant: '+ freq_var[-7:-4])
        else:
            print(files[0], 'did not have Oz')
            input('understood?')
        subplot_values +=1


def plot_all_ppt(cond_dir,conds,loc_path,color,label_itr,end_freq):
    '''
    Input:
        cond_dir:
        conds:
        loc_path:
        color:
        label_itr:
        end_freq:
    Output: 
        Warnings. It can only do 2 conditions now
    '''
    loc_itr = folderIterator(loc_path)
    c0_itr = folderIterator(cond_dir+'/'+conds[0])
    c1_itr = folderIterator(cond_dir+'/'+conds[1])
    for cond0,cond1 in zip(c0_itr,c1_itr):
        pass
        
    
def plot_all_epochs(power_path,cond,loc_path,color,end_freq=30,chan='Oz',subplot=False):
    '''
    In the future, allow subplot optino and ppt option
    '''
    power_itr = folderIterator(power_path)
    loc_itr   = folderIterator(loc_path)
    subplot_values=1
    for power_file,loc_file in zip(power_itr,loc_itr):
        print(power_file)
        features,freqVec = readOneFeatures(power_file,end_freq)
        print('Dimension of all of the powers')     
        print(features.shape) # sanity check
        plt.figure(subplot_values)
        
        
        chanlocs = getChanLocs(loc_file)
        if (chan in chanlocs):
            chan_ind = chanlocs.index(chan)
            # chan_powers = normalize_power(features[:,chan_ind,:])
            chan_powers = features[:,chan_ind,:]
            
            labels_count = features.shape[0]
        
            mean = np.mean(chan_powers,0)
            std = np.std(chan_powers,axis=0)

            plt.plot(freqVec, mean,color,label=cond+':'+str(labels_count)+'Epochs')
            plt.plot(freqVec,mean-std,color)
            plt.plot(freqVec,mean+std,color)

            plt.legend(loc='upper right', shadow=True, fontsize='x-large')
            power_file = power_file.decode("utf-8")
            plt.title('Participant: '+ power_file[-7:-4]+'\n Chan: '+chan)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
        else:
            print(files[0], ' did not have ', chan) 
            input('understood?')
        subplot_values +=1

# How am I flattening the features


def main():
    
    loc_dir  = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Chanlocs'
    # base_dir   = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/freqValues'
    conds_dir  = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Features'
    labels_dir = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Labels_all'
    conditions = ['On', 'MW']
    colors     = ['r','b']


    # Plot the baseline(after probe)
    plt.figure()
    plt.suptitle('Power Spectrum: ON vs MW')
    # plot_cond(base_dir,loc_dir,'k')

    # for cond,color in zip(conditions,colors):
    #     label_itr = folderIterator(labels_dir)
    #     plot_cond(conds_dir,cond,loc_dir,color,label_itr)


    for cond,color in zip(conditions,colors):
        power_path = conds_dir+'/'+cond
        plot_all_epochs(power_path,cond,loc_dir,color,chan='P8')

    plt.show()



if __name__ == "__main__":
    main()
