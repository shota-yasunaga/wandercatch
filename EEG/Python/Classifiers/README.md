# Classifiers
This folder contains srcipts/functions/classes for classification. 

In this project, I used more rigorous way of veryfying the results. This is partially because I did not have a big dataset.
<!---
TODO: I should probably insert a figure to describe what I'm doing here. 
-->

## Classes (SVM.py)
SVM.py contains useful classes to do the classification verification. It might not be very straightforwad to understand at the beggining, but those classes make it a lot easier to deal with verification. 
Currently, this class only supprots binary classification. 

### Tutorial
In this tutorial, I will explain how to use the classes I have implemented with one of the scripts that I have written. Here is the whole code.  This is meant to understand the code and not to be able to run the tutorial code as it is. 

```python
# Make a list of Classifiers
clfs = []
for kernel in ['linear']:
    for C in [0.00001,0.0001,0.001,0.1]:
        titles.append(kernel+ '(C='+str(C)+')')
        clfs.append(SVC(kernel=kernel,C=C))
clf_fold = clfFolder(clfs)
clf_fold.setAccuracyFun(accuracy_score,normalize=True)

dim  = 0
adc = acrossDimClassifier(clf_fold,dim)
k_fold   = kFolder(adc,num_split=5)

subs = subSampler(k_fold,num_sub_samples=10)
subs.set_subsample_func(getPartialSubsampledFeatures)
subs.init_sub_sample(on_path,mw_path,max_freq = 40,feature_type='freq', partial_func = partial_func)
subs.progress_bar()
scores = subs.score()
scores.save_plot_init(saving_plot_path)
# plot all of the scores
suptitle = 's'+ppt
y_label  ='Accuracy'
x_label  ='Classifier'
x = 0
scores.plot(x,across_dim=None,clf_dim=3,suptitle=suptitle,titles = titles,x_label=x_label,y_label=y_label,plt_func=plt.errorbar)
```
At the first 7 liners, I am creationg a list of classifiers and passing it to clfFolder. You can have any classifiers that supoort fit and predict. At this point (07-24-2018), this step is necessary (if you want to try one classifier, you pass a list that contains one classifier). The last line is setting what function you want to use for measuring the accuracy. You can change this, for example, to f1_score. setAccuracyFun can also take other key word inputs and it will pass them when it evaluates the accuracy. 
```python
clfs = []
for kernel in ['linear']:
    for C in [0.00001,0.0001,0.001,0.1]:
        titles.append(kernel+ '(C='+str(C)+')')
        clfs.append(SVC(kernel=kernel,C=C))
clf_fold = clfFolder(clfs)
clf_fold.setAccuracyFun(accuracy_score,normalize=True)
```
acrossDimClassifier is quite simple to deal with. You just pass the clfFolder that you have just created and specify which dimension you want to run classifiers as a function of (Read explanation of acrossDimClassifier). If you do not want to run classifiers with respect to a certain dimension, you can set dim to be 0. 

```python
dim  = 0
adc = acrossDimClassifier(clf_fold,dim)
```

kFolder is also quite simple. You just pass the acrossDimClassifier and specify the number of slipts (k).
```python
k_fold   = kFolder(adc,num_split=5)

```
subSampler is a little bit more complited. 

First, you specify the number of subSamples (for the meaning of this, look at the explanation of the [subSamples](#subsampler))
```python
subs = subSampler(k_fold,num_sub_samples=10)
```
Then, you initialize the subsample. This requires 2 steps. The first step is to set which function you use to do the subSamplings. You can look up on the README of mat2python.py for the different functions. 
```python
subs.set_subsample_func(getPartialSubsampledFeatures)
```

The secoond step is to actually subSample. The values passed to init_sub_samples are going to be passed to the function that you set with set_subsample_func. So, it is beneficial to see what each function supports although the basic interface is the same. 

```
subs.set_subsample_func(getPartialSubsampledFeatures)
subs.init_sub_sample(on_path,mw_path,max_freq = 40,feature_type='freq', partial_func = partial_func)
```

progress_bar is optional. It just shows the progress bar on the temrinal to make you feel better while waiting. 
```python
subs.progress_bar()
```
subs.score evokes all of the classifier iterator folder that you have passed up to here. It does all of the steps specified by those layers. This returns the scoreSummary object (read [scoreSummary](#scoresummary))
```python
scores = subs.score()
```
Then, you need to set where you want to save the plot. If you do not run this command, it does neither save nor display the plot (this is TODO for me).
```python
scores.save_plot_init(saving_plot_path)
```
Finally, we plot the results outputted to scores. For the detail of the specification for plot, look at [scoreSummary](#scoresummary). There are some things you need to take care here because of my laziness. Especially, the number of classifiers should be divisible by subplot_dims(x*y).Then, you need to specify which dimension, acrossDimClassifier and clfFolder were. 
```python
# plot all of the scores
suptitle = 's'+ppt
y_label  ='Accuracy'
x_label  ='Classifier'
x = 0
scores.plot(x,subplot_dims=[2,3],across_dim=None,clf_dim=3,suptitle=suptitle,titles = titles,x_label=x_label,y_label=y_label,plt_func=plt.errorbar)
```

### ClfItrFolder (*Classiffier Iterator Folder*)

This class is mostly for the interface (to show what kind of operations other classes should contain)

### acrossDimClassifier (across dimension classifier)

This class is to run classifications as a function of a feature dimension. For example, if the original features have 3 dimensions (samples, channels, frequency), you can run the classification with respect to channels, for example. This means that classification uses all of the frequencies but for one channel and does this for all of the channels. 

### ClfFolder
This class is to run different kind of classifiers at the same time. 

Currently, you need to have this at the lowest layer of the ClfItrFolders and I thihnk I will not develop upon this version. 

\* **ClfFolderOpposite** is basically the same class as ClfFolder, but it predicts the opposite in the test. I was getting some weird results and made this. You should probably not use this class.

### subSampler
This class sub-samples from the orignal samples on each category based on the function specified. 

### ScoreSummary
ScoreSummary deals with the results omitted from the Classifiers. Classifiers are dependent on this class. 

You can get ScoreSummary object from anyCltItrFolder(acrossDimClassifier, clfFolder,subSampler)

Look at [mat2python](https://github.com/andrillon/wandercatch/tree/master/EEG/Python#mat2python) for the functions to pass to ScoreSummary objects.

# Functions (Modules)


# Scripts
Scripts in general are to be modified according to your need. It has basic structure so that it's easier to modify. 

Looking at the [Tutorial](#tutorial) will help you understand the code. 

- SVM_all_pts_together.py
  
  run SVM with all of the participants' features together. 

- SVM_ppts_loop.py

  run classifications on different participants in a loop

- find_coefficients.py

  Run SVM with linear kernel and see the coefficients 

If you understand those, all you have to do for other projects is to modify those scripts. However, here are some other scripts that I have written.  TODO: decide what I will include. 