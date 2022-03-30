"""# RECOLA Preprocessing and Featurization

## Physio Featurization Pipeline

### Data Paths
"""

import json
import os
import math
import pandas as pd
import numpy as np
import datetime
import tables

import matplotlib.pyplot as plt
import IPython
import urllib.request

import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import datetime
import math
import arff

import pydub
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import make_chunks

from scipy.io import wavfile
import soundfile as sfile

plt.rcdefaults()
print(" Extracting and formatting Labels ...")

"""## Label Extraction"""
## Annotation Data Path
a_label_path = Path('recola/raw/baseline_features/labels/gs_1/arousal/')
v_label_path = Path('recola/raw/baseline_features/labels/gs_1/valence/')


plist = os.listdir(a_label_path)
selected_pid = [x[:-5] for x in plist]
selected_pid.sort()

selected_pid = ['dev_1', 'dev_2', 'dev_3', 'dev_4']# 'dev_5', 'dev_6', 'dev_7', 'dev_8'] 

## Combined Dataset Locations
feat = 'recola/features'
os.makedirs(feat,exist_ok=True)

## Cleaning Labels by availablability of audio
def label_extractor(pid, ap, vp, spath):
    labels = pd.DataFrame()
    
    ## Label Stack with corresponding participants
    f_name = pid + '.arff'
   
    ## Extract Arousal Labels
    p = os.path.join(ap, f_name)
    dataset = arff.loads(open(p), 'r')
    df_a = pd.DataFrame(dataset['data'], columns=['pid','frameTime','GoldStandard'])
    df_a = df_a.rename(columns={'GoldStandard': 'a_gs'})

    ## Extract Valence Labels
    p = os.path.join(vp, f_name)
    dataset = arff.loads(open(p), 'r')
    df_v = pd.DataFrame(dataset['data'], columns=['pid','frameTime','GoldStandard'])
    df_v = df_v.rename(columns={'GoldStandard': 'v_gs'})
    df_v = df_v.drop(columns=['frameTime', 'pid'])

    ## Concatenate Gold Standard Label series
    df = pd.concat([df_a, df_v], axis = 1)
    
    a_b = pd.DataFrame()
    v_b = pd.DataFrame()

    for i in range(0, math.floor(df.shape[0]/10)+1,1):
        start = i * 10
        end = (i + 1) * 10
        temp = df[start: end].copy()
        a_mean = temp['a_gs'].mean()
        v_mean = temp['v_gs'].mean()

        if a_mean < 0:
          a_mean = 0
        elif a_mean > 0:
          a_mean = 1
        
        if v_mean < 0:
          v_mean = 0
        elif v_mean > 0:
          v_mean = 1

        a_b = a_b.append([a_mean], ignore_index=True)
        v_b = v_b.append([v_mean], ignore_index=True)
    
    a_b = a_b.rename(columns={0: "arousal"})
    v_b = v_b.rename(columns={0: "valence"})   
    
    ## Concatenate new scaled labels of Arousal and Valence
    labels = pd.concat([a_b, v_b], axis = 1)
    labels = pd.DataFrame(labels, index=range(0, math.floor(df.shape[0]/10),1))

    # # Dumping Path
    feat_path = spath + '/' + pid + '/'
    os.makedirs(feat_path,exist_ok=True)

    p1 = os.path.join(feat_path,"labels.hdf5")
    labels.to_hdf(p1,  key='df', mode='w')
    tables.file._open_files.close_all()

for i in selected_pid:
    label_extractor(i, a_label_path,v_label_path, feat)
print(" Label Extraction Complete")

"""## Physio Signals"""

physio_path = 'recola/raw/recordings_physio'

#Feature write path
feat = 'recola/features/'
os.makedirs(feat,exist_ok=True)

def feat_slice(df, f_data, seg_len, duration):
    df_feat = pd.DataFrame()
    start = 0
    while start + int(f_data * seg_len) < duration:
        a = df[start : start + int(f_data * seg_len )].copy()
        a = a.reset_index(drop=True)
        df_feat = df_feat.append(a, ignore_index=True)
        start = start + int(f_data * seg_len )      
    return df_feat

### Running Feature Extraction on All Participant segments
def physio_read(physio_path, pid, feat_path ):
      df = pd.DataFrame()

      # Extracting segments path for Specific Participant
      f_name = physio_path + '/' + pid + '.csv'
      df = pd.read_csv(f_name,sep=';')

      bvp = feat_slice(df['HR'], f_data = 250, seg_len = 0.4, duration = df.shape[0])
      
      eda = feat_slice(df['EDA'], f_data = 250, seg_len = 0.4, duration = df.shape[0])

      # # Dumping Path
      feat_path = feat + '/' + pid
      os.makedirs(feat_path, exist_ok=True)
      
      p1 = os.path.join(feat_path,"eda.hdf5")
      eda.to_hdf(p1,  key='df', mode='w')

      p1 = os.path.join(feat_path,"bvp.hdf5")
      bvp.to_hdf(p1,  key='df', mode='w')

      tables.file._open_files.close_all()
print(" Extracting and formatting Physio Signals ...")
for pid in selected_pid:
    physio_read(physio_path, pid,feat )

print(" Physio Signals Formatted.")

"""## Audio Signals

### Feature Libraries
"""

import librosa
import librosa.display

import pydub
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import make_chunks

import opensmile

import audiofile

import cv2
"""### Data Paths"""

## Audio Data Path
audio_path = 'recola/raw/recordings_audio/'

#Feature write path
feat ='recola/features/'
os.makedirs(feat,exist_ok=True)

"""### Feature Definitions

#### ComParE2016
"""

# ComParE 2016
f_compare = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

# eGeMAPS v02
f_egemaps02 = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

"""#### Spectrograms"""

def spectrogram_melfreq(seg, pid, ind, f_path, my_dpi = 256):

    ## Get Spectrogram
    wave, sr = librosa.load(seg)

    ## #Speech/Music contemporary values
    n_mels = 512      # Typical Mel Bands for speech [Tested for best discrimination]
    n_fft = 2048      # Window Length for FFT (Speech/Music contemporary values)
    hop_length = 512  # = math.floor(sr * 0.015) # Amount of Shift per FFT
    
    S = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=n_mels,  hop_length = hop_length,  n_fft = n_fft) # , hop_length = hop_length, win_length = win_length)
    Sdb = librosa.power_to_db(S, ref=np.max)
    
    ## Plot Spectrogram
    plt.figure(figsize=(1., 1.))
    librosa.display.specshow(Sdb, sr=22050, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig('spec.tif' , dpi=my_dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.clf()
    
    ## Read Spectrogram
    spec_res = my_dpi
    img  = cv2.imread('spec.tif')
    img2 = cv2.resize(img,(spec_res,spec_res))
    img2 = img2/255.00
    return img2
    

"""### Raw Audio Feature Extraction"""

def segment_chunks(signal_n, seg):
    seg_len = seg * 1000                    # pydub calculates in millisec
    chunks = make_chunks(signal_n, seg_len) # Make chunks of 'seg' seconds
    for i in np.arange(0,1,len(chunks)):
        if (chunks[i].duration_seconds != (seg * 1000)):
            del chunks[i]
    return chunks

def aud_feat(pid, feat, audio_path):
          # Feature Dataframe Initializations
          feat_com_t    = pd.DataFrame()
          feat_egemap_t = pd.DataFrame()
          feat_spec_mf_t  = pd.DataFrame()

          ## Extracting Audio from Specific Participant
          aud_file = audio_path +  pid + '.wav'
          signal = AudioSegment.from_file(aud_file , "wav") 

          ## Slicing Audio in Chunks
          aud_chunks = segment_chunks(signal,seg = 0.4)
          
          
          # Feature Dataframe Initializations
          feat_com    = pd.DataFrame()
          feat_egemap = pd.DataFrame()
          feat_spec_mf  = pd.DataFrame()

          ## Looping over all chunks from a Specific Participant 
          for i, s in enumerate(aud_chunks):
                  s.export("segment.wav", format="wav")
                  seg = 'segment.wav'

                  # Extracting and Appending Feature #1        
                  feat_com = feat_com.append(f_compare.process_file(seg))

                  # Extracting and Appending Feature #2
                  feat_egemap = feat_egemap.append(f_egemaps02.process_file(seg))

                  # Extracting and Appending Feature #3
                  feat_spec_mf = feat_spec_mf.append([[spectrogram_melfreq(seg, pid, i , feat, my_dpi = 256)]])     
                              
                  os.remove('segment.wav')        
                  del s, seg

          ## Assign Participant ID to the Samples
          feat_com['pid'] = pid
          feat_egemap['pid'] = pid
          feat_spec_mf['pid'] = pid

          # Reset Column Titles
          feat_spec_mf = feat_spec_mf.rename(columns={0: "MelFreq Spectrogram"})

          ## Reset Indices for the Dataframes
          feat_com.reset_index(drop=True, inplace=True)
          feat_egemap.reset_index(drop=True, inplace=True)
          feat_spec_mf.reset_index(drop=True, inplace=True)


          # Dumping Path
          p1 = os.path.join(feat, pid,  "compar.hdf5")
          feat_com.to_hdf(p1,  key='df', mode='w') #, format='table')

          p1 = os.path.join(feat, pid,  "egemap.hdf5")
          feat_egemap.to_hdf(p1,  key='df', mode='w') #, format='table')

          p1 = os.path.join(feat, pid,  "mat_melfreq.hdf5")
          feat_spec_mf.to_hdf(p1,  key='df', mode='w')#, format='table')

          tables.file._open_files.close_all()

          del feat_com, feat_egemap, feat_spec_mf, aud_file, signal, aud_chunks



print(" Extracting and formatting Audio Features ...")
for pid in selected_pid:
    print(' Generating Audio Features for ' + pid)
    aud_feat(pid, feat, audio_path)

print(" Audio Features Generated")

"""### Noisy Audio Feature Extraction"""

import glob
demand_data = 'demand/' 
noisy_audios = [AudioSegment.from_wav(wav_file) for wav_file in glob.glob(demand_data+'*.wav')]

### Running Feature Extraction on All Participant

## ## Features - eGeMAPsv02 + ComPaRE2016 + MelFreq Spectrograms


def noisy_aud_feat(pid, feat, audio_path ):
      names = ['k','l','o','h']
      cnt = 0
      # Looping over all Participants
      for noise in noisy_audios:
            # Feature Dataframe Initializations
            feat_com_t    = pd.DataFrame()
            feat_egemap_t = pd.DataFrame()
            feat_spec_mf_t  = pd.DataFrame()

            ## Extracting Audio from Specific Participant
            aud_file = audio_path +  pid + '.wav'
            signal_n = AudioSegment.from_file(aud_file , "wav") 

            signal = signal_n.overlay(noise, gain_during_overlay = 0)

            ## Slicing Audio in Chunks
            aud_chunks = segment_chunks(signal,seg = 0.4)

            # Feature Dataframe Initializations
            feat_com    = pd.DataFrame()
            feat_egemap = pd.DataFrame()
            feat_spec_mf  = pd.DataFrame()

            ## Looping over all chunks from a Specific Participant 
            for i, s in enumerate(aud_chunks):
                  s.export("segment.wav", format="wav")
                  seg = 'segment.wav'

                  # Extracting and Appending Feature #1        
                  feat_com = feat_com.append(f_compare.process_file(seg))

                  # Extracting and Appending Feature #2
                  feat_egemap = feat_egemap.append(f_egemaps02.process_file(seg))

                  # Extracting and Appending Feature #3
                  feat_spec_mf = feat_spec_mf.append([[spectrogram_melfreq(seg, pid, i , feat, my_dpi = 256)]])     
                              
                  os.remove('segment.wav')        
                  del s, seg

            ## Assign Participant ID to the Samples
            feat_com['pid'] = pid
            feat_egemap['pid'] = pid
            feat_spec_mf['pid'] = pid

            # Reset Column Titles
            feat_spec_mf = feat_spec_mf.rename(columns={0: "MelFreq Spectrogram"})

            ## Reset Indices for the Dataframes
            feat_com.reset_index(drop=True, inplace=True)
            feat_egemap.reset_index(drop=True, inplace=True)
            feat_spec_mf.reset_index(drop=True, inplace=True)


            # Dumping Path
            p1 = os.path.join(feat, pid, names[cnt] + "_compar.hdf5")
            feat_com.to_hdf(p1,  key='df', mode='w') #, format='table')

            p1 = os.path.join(feat, pid, names[cnt] + "_egemap.hdf5")
            feat_egemap.to_hdf(p1,  key='df', mode='w') #, format='table')

            p1 = os.path.join(feat, pid, names[cnt] + "_mat_melfreq.hdf5")
            feat_spec_mf.to_hdf(p1,  key='df', mode='w')#, format='table')

            tables.file._open_files.close_all()
            
            cnt =  cnt + 1
            
            del feat_com, feat_egemap, feat_spec_mf, aud_file, signal, aud_chunks
print(" Extracting and formatting Noisy Audio Features ...")

for pid in selected_pid:
    print(' Generating Noisy Audio Features for ' + pid)
    noisy_aud_feat(pid, feat, audio_path)
    

print(" Noisy Audio Features Generated")



"""## Baseline Features"""

## Baseline Features Path
base_path = 'recola/raw/baseline_features/features/'

## Combined Dataset Locations
feat = 'recola/features'
os.makedirs(feat,exist_ok=True)

## Cleaning Labels by availablability of audio
def feature_extractor(feat_path, f_path, tag, pid):
    df = pd.DataFrame()
    if tag == 'audio/DeepSpectrum': 
            ##  Extract Feature
            f_name = pid + '_' + str(0.0) +  '_0.4.arff'
            p = os.path.join(f_path,f_name)
            dataset = arff.loads(open(p), 'r')
            df = pd.DataFrame(dataset['data'], columns=dataset['attributes'])
            df.loc[:,'pid'] = pid        
            
            ##  Dumping Path
            t = tag.split('/')[1]
            k = feat_path + '/' + pid + '/'
            os.makedirs(k, exist_ok=True)
            p1 = k + '/' +  t + "_b.hdf5"
            df.to_hdf(p1,  key='df', mode='w')
            tables.file._open_files.close_all()
    
    elif tag == 'audio/eGeMAPSfunct' :
            tavg = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            for j in tavg:
                ##  Extract Feature
                f_name = pid + '_' +  str(j) +  '_0.4.arff'
                p = os.path.join(f_path,f_name)
                dataset = arff.loads(open(p), 'r')
                df = pd.DataFrame(dataset['data'], columns=dataset['attributes'])
                df.loc[:,'pid'] = pid
                df.loc[:,'t_a'] = j  
              
                ##  Dumping Path
                t = tag.split('/')[1]
                k = feat_path + '/' + pid + '/'
                os.makedirs(k, exist_ok=True)
                p1 = k + '/' +  t + '_' + str(j) + "_b.hdf5"
                df.to_hdf(p1,  key='df', mode='w')  
                tables.file._open_files.close_all()
    else:          
            tavg = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
            for j in tavg:
                ##  Extract Feature
                f_name = pid + '_' +  str(j) +  '_0.4.arff'
                p = os.path.join(f_path,f_name)
                dataset = arff.loads(open(p), 'r')
                df = pd.DataFrame(dataset['data'], columns=dataset['attributes'])
                df.loc[:,'pid'] = pid
                df.loc[:,'t_a'] = j  

                ##  Dumping Path
                t = tag.split('/')[1]
                k = feat_path + '/' + pid + '/'
                os.makedirs(k, exist_ok=True)
                p1 = k + '/' +  t + '_' + str(j) + "_b.hdf5"
                df.to_hdf(p1,  key='df', mode='w')  
                tables.file._open_files.close_all()

base_path = 'recola/raw/baseline_features/features/'
t1 = ['physio/EDA', 'physio/HRHRV', 'audio/eGeMAPSfunct', 'audio/DeepSpectrum']

print(" Extracting and formatting Baseline  Features ...")
for i in selected_pid:
    for m in t1: 
        f1 = os.path.join(base_path,m)
        feature_extractor(feat, f1,m,i)




"""# Check List"""

import glob 
import pandas as pd
import os

import math
import numpy as np
import tables


## Feature Path
features = 'recola/features/'
plist = os.listdir(features)
plist.sort()

feat_set = ['',  'k_', 'l_', 'o_', 'h_']

for f in plist:
    for pid in feat_set:
        f_path = features + f + '/' 
        
        ## Annotation Labels Load
        specm = pd.read_hdf(os.path.join(f_path,pid + 'mat_melfreq.hdf5'))        
        compar  =  pd.read_hdf(os.path.join(f_path,pid + 'compar.hdf5'))
        egemap =  pd.read_hdf(os.path.join(f_path,pid + 'egemap.hdf5'))
        eda = pd.read_hdf(os.path.join(f_path,'eda.hdf5'))
        bvp = pd.read_hdf(os.path.join(f_path,'bvp.hdf5'))
        labels = pd.read_hdf(os.path.join(f_path,'labels.hdf5'))

        print(" Feature size check before corrections")
        print( f + "  " +  str(specm.shape[0]) + "  " +  str(egemap.shape[0])+ "  " +  str(compar.shape[0])+ "  " +  str(eda.shape[0])+ "  " +  str(bvp.shape[0])+ "  " +  str(labels.shape[0]))

        s = min(specm.shape[0], egemap.shape[0], compar.shape[0],  eda.shape[0], bvp.shape[0],  labels.shape[0])

        specm = specm[:s]
        compar = compar[:s]
        egemap = egemap[:s]
        eda = eda[:s]
        bvp = bvp[:s]
        labels = labels[:s]
        
        print(" Feature size check after corrections")
        print( f + "  " +  str(specm.shape[0]) + "  " +  str(egemap.shape[0])+ "  " +  str(compar.shape[0])+ "  " +  str(eda.shape[0])+ "  " +  str(bvp.shape[0])+ "  " +  str(labels.shape[0]))

        f_path = features + f + '/'
        os.makedirs(f_path, exist_ok=True)
        
        p1 = os.path.join(f_path,"bvp.hdf5")
        bvp.to_hdf(p1,  key='df', mode='w')

        p1 = os.path.join(f_path,"eda.hdf5")
        eda.to_hdf(p1,  key='df', mode='w')

        p1 = os.path.join(f_path,pid + "mat_melfreq.hdf5")
        specm.to_hdf(p1,  key='df', mode='w')#, format='table')
        
        p1 = os.path.join(f_path,pid + "compar.hdf5")
        compar.to_hdf(p1,  key='df', mode='w')#, format='table')
        
        p1 = os.path.join(f_path,pid + "egemap.hdf5")
        egemap.to_hdf(p1,  key='df', mode='w')#, format='table')
        
        p1 = os.path.join(f_path,"labels.hdf5")
        labels.to_hdf(p1,  key='df', mode='w')#, format='table')

        tables.file._open_files.close_all()        




# ## Count of Features and Labels

# list_f = glob.glob(features + '*/bvp.hdf5')
# list_f.sort()
# for i in list_f:
    # v = pd.read_hdf(i)
    # print(i + '  :  ' +str(v.shape[0]))

# list_f = glob.glob(features + '*/eda.hdf5')
# list_f.sort()
# for i in list_f:
    # v = pd.read_hdf(i)
    # print(i + '  :  ' +str(v.shape[0]))

# list_f = glob.glob(features + '*/labels.hdf5')
# list_f.sort()
# for i in list_f:
    # v = pd.read_hdf(i)
    # print(i + '  :  ' +str(v.shape[0]))

# list_f = glob.glob(features + '*/*compar.hdf5')
# list_f.sort()
# for i in list_f:
    # v = pd.read_hdf(i)
    # print(i + '  :  ' +str(v.shape[0]))

# list_f = glob.glob(features + '*/*mat_melfreq.hdf5')
# list_f.sort()
# for i in list_f:
    # v = pd.read_hdf(i)
    # print(i + '  :  ' +str(v.shape[0]))
    
# list_f = glob.glob(features + '*/*egemap.hdf5')
# list_f.sort()
# for i in list_f:
    # v = pd.read_hdf(i)
    # print(i + '  :  ' +str(v.shape[0]))    
    

# import pandas as pd
# import os



# """#### 1s Label Distribution"""

# ## Combined Dataset Locations

# features = 'recola/features/'
# feat = 'recola/'


# list_f = glob.glob(features + '*/labels.hdf5')
# list_f.sort()


# df = pd.DataFrame()

# for pid in list_f:
    # ## Load Data
    # labels = pd.read_hdf(pid)
    
    # sa = labels['arousal']
    # sa_z = sa[sa == 0].count()
    # sa_o = sa[sa == 1].count()

    # sv = labels['valence']
    # sv_z = sv[sv == 0].count()
    # sv_o = sv[sv == 1].count()

    # df = df.append([[pid, sa_z, sa_o, sv_z, sv_o]], ignore_index=True)


# df = df.rename(columns={0: "pid", 1: "A_Self_0", 2: "A_Self_1", 3: "V_Self_0",  4: "V_Self_1"})
# df.reset_index(inplace=True)
# df = df.drop(columns=['index'])


# p1 = os.path.join(feat,"label.csv")
# df.to_csv(p1)      