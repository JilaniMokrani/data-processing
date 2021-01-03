#!/usr/bin/env python
# coding: utf-8

# In[2]:


import scipy.io.wavfile as wavfile
import numpy
import os.path
from os import walk
from scipy import stats
import numpy as np
import librosa 
import numpy as np
from scipy.stats import norm
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def snr(name):
    _wavs = []
    snr_data = []
    for (_,_,filenames) in walk(name):
        _wavs.extend(filenames)
        break
    for _wav in _wavs:
        # read audio samples
        if(".wav" in _wav ): 
            file = name+"/" + _wav
            data, rate = librosa.load(file)
            duration = librosa.get_duration(y=data, sr=rate)
            print ("- file: "+ file+", duration: "+str(duration))

            singleChannel = data
            try:
                singleChannel = numpy.sum(data, axis=1)
            except:
                # was mono after all
                pass

            norm = singleChannel / (max(numpy.amax(singleChannel), -1 * numpy.amin(singleChannel)))
            res = signaltonoise(norm)
            snr_data.append(res)
            print ("- "+_wav+" : "+str(res))
    return snr_data

def getSNR(folder,output, types):
    for type in types:
        #- Signal to Noise ratio
        snr_data = snr(folder+"/"+type)
        pickle.dump(snr_data, open(output+"/"+type+"/snr.data", 'wb')) 
        snr_data = pickle.load(open(output+"/"+type+"/snr.data", 'rb'))


# In[34]:


def getHIST(data, type, imagePath, x_txt, y_txt):
    # matplotlib histogram
    plt.hist(data, edgecolor = 'black',bins = int(180/binwidth))
    # Add labels
    plt.title('Histogram of '+type+' class, total='+str(len(data)))
    plt.xlabel(x_txt)
    plt.ylabel(y_txt)
    plt.savefig(imagePath+'/Histogram of '+type+".png")
    plt.show()
def getHIST_KDE(data, type, imagePath, x_txt, y_txt):
    # Density Plot and Histogram of all arrival delays
    sns.distplot(
        data, 
        hist=True, 
        kde=True, 
        rug=True, 
        rug_kws={"color": "g"}, 
        kde_kws={"color": "red", "lw": 3, "label": "KDE"},
        #hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"},
        bins=int(180/5), 
        color = 'darkblue', 
        hist_kws={'edgecolor':'black'},
        #kde_kws={'linewidth': 4}
        )
    # Add labels
    plt.title('Histogram of '+type+' class, total='+str(len(data)))
    plt.xlabel(x_txt)
    plt.ylabel(y_txt)
    plt.savefig(imagePath+'/Histogram_KDE of '+type+".png")
    plt.show()
def getRug_KDE(data, type, imagePath, x_txt, y_txt):
    # Density Plot with Rug Plot
    sns.distplot(
        data, 
        hist = False, 
        kde = True, 
        rug = True,
        color = 'darkblue', 
        kde_kws={'linewidth': 3, "label": "KDE"},
        rug_kws={'color': 'black'})
    # Add labels
    plt.title('Density plot of '+type+' class, total='+str(len(data)))
    plt.xlabel(x_txt)
    plt.ylabel(y_txt)
    plt.savefig(imagePath+'/Density plot of '+type+".png")
    plt.show()

def PlotScatter(data, type, color_type, imagePath, x_txt, y_txt):
    plt.scatter(
        np.arange(len(data)), 
        data, 
        alpha=0.5, 
        color = color_type, 
        label="PCG Sample: "+type+" class")
    plt.title('PCG sample plot of '+type+' class, total='+str(len(data)))
    plt.xlabel(x_txt)
    plt.ylabel(y_txt)
    plt.savefig(imagePath+'/PCG sample plot of '+type+".png")
    plt.show()


# In[35]:



# In[49]:


import librosa.display
import scipy.io.wavfile as wavfile
import numpy
import os
import os.path
from os import walk
from scipy import stats
import numpy as np
import librosa 
import numpy as np
from scipy.stats import norm
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from pycm import *

def extractMelSpectrogram_features(folder,melFolder,SpectreFolder, types):
    hop_length = 512
    n_fft = 1024
    for nametype in types:
        _wavs = []
        wavs_duration = []
        for (_,_,filenames) in walk(folder+"/"+nametype+"/"):
            _wavs.extend(filenames)
            break
        Mel_Spectrogram = []
        for _wav in _wavs:
            # read audio samples
            if(".wav" in _wav and not os.path.exists(SpectreFolder + "/" + nametype + "/" + _wav.replace(".wav",".png"))): 
                file = folder+"/" +nametype+"/"+_wav
                signal, rate = librosa.load(file)  
                #The Mel Spectrogram
                S = librosa.feature.melspectrogram(signal, sr=rate, n_fft=n_fft, hop_length=hop_length)
                S_DB = librosa.power_to_db(S, ref=np.max)
                #Mel_Spectrogram.append(S_DB)
                #print (S_DB)
                librosa.display.specshow(S_DB, sr=rate, hop_length=hop_length)
                print(nametype + "/" +  _wav)
                plt.savefig(SpectreFolder + "/" + nametype + "/" + _wav.replace(".wav",".png") )
                pickle.dump(S_DB,open(melFolder+"/"+nametype+"/"+_wav.replace(".wav",".mel"),"wb"))
                
folder = os.environ["INPUT_DATA_PATH"]
output = os.environ["OUTPUT_DATA_PATH"]
types = os.listdir(folder)

FiguresPath = output + "/Figures"
if not os.path.exists(FiguresPath):
    os.mkdir(FiguresPath)
    for type in types:
        os.mkdir(FiguresPath + "/" + type)

MelSpectrogramPath = output + "/MelSpectrograms"
if not os.path.exists(MelSpectrogramPath):
    os.mkdir(MelSpectrogramPath)
    for type in types:
        os.mkdir(MelSpectrogramPath + "/" + type)

SpectresPath = output + "/Spectres"
if not os.path.exists(SpectresPath):
    os.mkdir(SpectresPath)
    for type in types:
        os.mkdir(SpectresPath + "/" + type)

getSNR(folder=folder,output = FiguresPath, types = types)

dataset_type = ""
binwidth = 5
for type in types:
    file_snr = FiguresPath+'/'+type+"/snr.data"
    snr_data = pickle.load(open(file_snr, 'rb'))
    
    #"SNR"
    print ("SNR of "+type+"...")
    getHIST(snr_data, type, FiguresPath + "/" + type, 'Signal to Noise Ratio', 'PCG data')
    getHIST_KDE(snr_data, type, FiguresPath + "/" + type, 'Signal to Noise Ratio', 'Density')
    getRug_KDE(snr_data, type, FiguresPath + "/" + type, 'Signal to Noise Ratio', 'Density')
    PlotScatter(snr_data, type, "red", FiguresPath + "/" + type, "PCG sample", "Signal to Noise Ratio")


extractMelSpectrogram_features(folder, MelSpectrogramPath, SpectresPath, types)

