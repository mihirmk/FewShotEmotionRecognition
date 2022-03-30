"""

Task 1 -  For K-EmoCon

Steps in this code :
1. Preprocessing
    a. Audio Diarization
    b. Physiological data segmentation 
    c. Participant-wise annotation segmentation

2. Featurization
    a. Audio Featurization (Natural Audio and Nosie-imputed Audio)
    b. Physiological Data slicing
    d. Annotation Conversion to Binary Labels

3. Verification
    a. Corection of all the generated labels and features by participants
    b. Editing adn discarding of missing labels and features

"""





"""# K-EmoCon PREPROCESSING

## Audio Data
"""

import json
import os
import math
import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import IPython
import urllib.request

import pydub
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import make_chunks

from scipy.io import wavfile
import soundfile as sfile
import tables

import cv2

import librosa
import librosa.display

import pydub
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import make_chunks

import opensmile

import audiofile

debate_audio_path = 'kemocon/raw/debate_audios'
vox_pair_path = 'kemocon/metadata/vox_pair_dia'

vox_json_path = 'kemocon/metadata/vox_dia'
os.makedirs(vox_json_path,exist_ok=True)

# Vox-Sort based Audio Segmentation
vox_audio_path = 'kemocon/processed/audio_data_p'
os.makedirs(vox_audio_path,exist_ok=True)

pid_vox_list = os.listdir(vox_pair_path)
pid_vox_list.sort()
aud_list = os.listdir(debate_audio_path)

# Labels from Vox-Sort Diarization with Off-line verification
vox_labels = { 'pid' : ['p1.p2~.srt', 'p3.p4~.srt','p5.p6~.srt','p7.p8~.srt','p9.p10~.srt','p11.p12~.srt','p13.p14~.srt','p15.p16~.srt','p17.p18~.srt','p19.p20~.srt','p21.p22~.srt','p23.p24~.srt','p25.p26~.srt','p27.p28~.srt','p29.p30~.srt','p31.p32~.srt', ],
                 '1' : [1, 3, 5, 8, 9, 11, 13, 15, 17, 20, 22, 24, 26, 27, 30, 31],
                 '2' : [2, 4, 6, 7, 10, 12, 14, 16, 18, 19, 21, 23, 25, 28, 29, 32]}

vox_labels = pd.DataFrame(vox_labels)

## List of Participant IDs
debate_physio_path = 'kemocon/raw/e4_data'
pid_list = os.listdir(debate_physio_path)
rejected_pid = ['2','3','6','7','11','12','17','18','26','27','28']
selected_pid = [int(item) for item in pid_list if item not in rejected_pid]
selected_pid.sort()

"""### Diarization"""
print("Starting Audio Diarizations")
# loop through all files in folder
for filename in pid_vox_list:
    ### Diarization Segment Generation
    path1 = os.path.join(vox_pair_path,filename)
    a = pd.read_csv(path1, header=None,names=['dia'])

    # Separate Speakers and Segments 
    df1 = a[a.index % 2 == 0].copy()
    df1 = df1.reset_index(drop=True)
    df2 = a[a.index % 2 == 1].copy()
    df2 = df2.reset_index(drop=True)

    # create dataframe for adjustment of time
    df = pd.concat([df1, df2], axis=1)
    df = df.set_axis(["speaker", "segment"], axis=1)


    # Convert Segments string to start and end times
    dfone = pd.DataFrame(index=list(df.index),columns=['speaker','start','end'])
    for row in list(df.index):
      dfone.loc[row,'speaker'] = df['speaker'][row]
      dfone.loc[row,'start'] = df['segment'][row][:-16]
      dfone.loc[row,'end'] = df['segment'][row][16:]


    z = datetime.datetime.strptime(dfone.start[0],"%H:%M:%S.%f")
    # Convert String datetime to absolute milliseconds
    dfone['start'] = dfone['start'].apply(lambda d : datetime.datetime.strptime(d,"%H:%M:%S.%f"))
    dfone['start'] = dfone['start'].apply(lambda d : (d - z).total_seconds()*1000)

    dfone['end'] = dfone['end'].apply(lambda d : datetime.datetime.strptime(d,"%H:%M:%S.%f"))
    dfone['end'] = dfone['end'].apply(lambda d : (d - z).total_seconds()*1000)


    # Slicing Diarization for individual Speakers according to Identified Labels
    p_i = int(vox_labels.loc[vox_labels['pid'] == filename]['1'])
    df_1 = dfone[dfone.speaker == '1:'].copy()
    df_1.loc[:,'speaker'] = p_i
    df_1 = df_1.reset_index(drop=True)
    # Saving First Diarization Label
    p1 = os.path.join(vox_json_path,str(p_i)+".json")
    df_1.to_json(p1)


    p_j = int(vox_labels.loc[vox_labels['pid'] == filename]['2'])
    df_2 = dfone[dfone.speaker == '2:'].copy()
    df_2.loc[:,'speaker'] = p_j
    df_2 = df_2.reset_index(drop=True)
    # Saving Second Diarization Label
    p1 = os.path.join(vox_json_path,str(p_j)+".json")
    df_2.to_json(p1)

    # Speaker Labels 128 and 256 correspond to the Non-speech sounds and silences respectively, which have been removed completely


    # Audio Segmentation based on the Generated Diarization Files

    # Loading Corresponding Audio for the generated Diarization
    file = os.path.join(debate_audio_path,filename[:-5]+".wav")
    fs, signal = wavfile.read(file)

    if(p_i in selected_pid): 
              ################################################################
              #       Slicing Audio by First Speaker Label from Diarization  #
              ################################################################
              sample_n = np.empty((1,2))
              for part in list(np.arange(0,len(df_1),1)):
                  seg_size_t = (df_1.end[part] - df_1.start[part])/1000
                  seg_size_s = math.floor(seg_size_t * fs)
                  seg_start = math.floor((df_1.start[part]/1000) * fs)
                  sample_n = np.append(sample_n, signal[seg_start : seg_start + seg_size_s], axis = 0)
              
              
              # Writing First Speaker Diarization Audio
              sfile.write(vox_audio_path+"/pid_{0}.wav".format(p_i), sample_n, fs)

    if(p_j in selected_pid):
              ################################################################
              #       Slicing Audio by Second Speaker Label from Diarization  #
              ################################################################
              sample_n = np.empty((1,2))
              for part in list(np.arange(0,len(df_2),1)):
                  seg_size_t = (df_2.end[part] - df_2.start[part])/1000
                  seg_size_s = math.floor(seg_size_t * fs)
                  seg_start = math.floor((df_2.start[part]/1000) * fs)
                  sample_n = np.append(sample_n, signal[seg_start : seg_start + seg_size_s], axis = 0)
              

              # Writing First Speaker Diarization Audio
              sfile.write(vox_audio_path+"/pid_{0}.wav".format(p_j), sample_n, fs)    


print("Audio Diarizations Completed.")


"""## Physio Data"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import datetime
import os
import math
import soundfile as sfile
import soundfile
# Dataset Paths

# Audio files Path
aud_path = 'kemocon/processed/audio_data_p'

# Diarization Data
vox_json_path = "kemocon/metadata/vox_dia"

# Processed Physio Write Path
write_physio_path = Path('kemocon/processed/physio_data_p')
os.makedirs(write_physio_path,exist_ok=True)

# Timestamp adjustments
meta_path = 'kemocon/metadata/'
time_stamp = pd.read_csv(os.path.join(meta_path,"subjects.csv"))
time_stamp = time_stamp.set_index(np.arange(1,33,1))

## List of Participant IDs
debate_physio_path = 'kemocon/raw/e4_data/'
pid_list = os.listdir(debate_physio_path)
rejected_pid = ['2','3','6','7','11','12','17','18','26','27','28']
selected_pid = [int(item) for item in pid_list if item not in rejected_pid]
selected_pid.sort()

print("Starting Physio data Separations")

# loop through all files in folder
for pid in selected_pid:
    # # Extracting Annotation Length from file
    # f_name = os.path.join(aud_path,'pid_'+str(pid)+'.wav')
    # aud = sfile.SoundFile(f_name)
    # aud_len = math.floor(aud.frames/aud.samplerate)

    # Extract Diarization File from Vox_Sort Method for per Pair Speaker Segmentation by Audio
    dia_path = os.path.join(vox_json_path,str(pid)+'.json')
    segments = pd.read_json(dia_path) #, orient='split')

    # Extract Physio Folder
    deb_path = os.path.join(debate_physio_path,str(pid))

    # Extracting raw participant data from Physio Folder
    hr = pd.read_csv(deb_path + "/E4_HR.csv")
    eda = pd.read_csv(deb_path + "/E4_EDA.csv")
    bvp = pd.read_csv(deb_path + "/E4_BVP.csv")

    if (pid == 29 or pid == 31):
        hra = hr[hr['device_serial'] == 'A013E1'].copy()
        hra.reset_index(drop=True)
        hr = hra.copy()
        
        edaa = eda[eda['device_serial'] == 'A013E1'].copy()
        edaa.reset_index(drop=True)
        eda = edaa.copy()
        
        bvpa = bvp[bvp['device_serial'] == 'A013E1'].copy()
        bvpa.reset_index(drop=True)
        bvp = bvpa.copy()                

    elif (pid == 30 or pid == 32):
        hra = hr[hr['device_serial'] == 'A01A3A'].copy()
        hra.reset_index(drop=True)
        hr = hra.copy()
        
        edaa = eda[eda['device_serial'] == 'A01A3A'].copy()
        edaa.reset_index(drop=True)
        eda = edaa.copy()
        
        bvpa = bvp[bvp['device_serial'] == 'A01A3A'].copy()
        bvpa.reset_index(drop=True)
        bvp = bvpa.copy()


    ## 1. Cut 1 -  Extracting Pair-wise Debate Time stamps - to cut non-debate time segments 
    pid_start = time_stamp.loc[int(pid)]["startTime"]
    pid_end   = time_stamp.loc[int(pid)]["endTime"]

    # Cutting non-Debate time from Physio Data
    hr_a = hr.loc[hr['timestamp'].between(pid_start, pid_end)].copy()
    eda_a = eda.loc[eda['timestamp'].between(pid_start, pid_end)].copy()
    bvp_a = bvp.loc[bvp['timestamp'].between(pid_start, pid_end)].copy()

    # Resetting Index for Looping on pid later
    hr_p = hr_a.reset_index(drop=True)
    eda_p = eda_a.reset_index(drop=True)
    bvp_p = bvp_a.reset_index(drop=True)
      
    # # Setting Start of Physio data to relative timestamp
    hr_p['timestamp'] -= hr_p.timestamp[0]
    eda_p['timestamp'] -= eda_p.timestamp[0]
    bvp_p['timestamp'] -=  bvp_p.timestamp[0]

    hr_n = pd.DataFrame(columns=['timestamp','value'])
    eda_n = pd.DataFrame(columns=['timestamp','value'])
    bvp_n = pd.DataFrame(columns=['timestamp','value'])


    ### 2. Cut 2 - Extracting Participant Physio Data corresponding to Audio Segments from Diarization
    for seg in list(segments.index):
        pd_start = segments['start'][seg].copy()
        pd_end   = segments['end'][seg].copy()
        a = hr_p.loc[hr_p.timestamp.between(pd_start, pd_end)].copy() 
        hr_n = hr_n.append(a, ignore_index = True)
        a = eda_p.loc[eda_p.timestamp.between(pd_start, pd_end)].copy()
        eda_n = eda_n.append(a, ignore_index = True)
        a = bvp_p.loc[bvp_p.timestamp.between(pd_start, pd_end)].copy()
        bvp_n = bvp_n.append(a, ignore_index = True)

    hr_n = hr_n[['timestamp','value']]
    eda_n = eda_n[['timestamp','value']]
    bvp_n = bvp_n[['timestamp','value']]


    hr_n = hr_n.reset_index(drop=True)
    eda_n = eda_n.reset_index(drop=True)
    bvp_n = bvp_n.reset_index(drop=True)

    # print(str(pid)  + "  " + str(segments.shape) + "  " +  str(hr_n.shape) + "  " + str(eda_n.shape) +  "   " +  str(bvp_n.shape) )

    # Dumping Path
    write_path = os.path.join(write_physio_path,str(pid))
    os.makedirs(write_path,exist_ok=True)

    # Writing to json
    with open(os.path.join(write_path,"hr.json"), "w") as k:
        k.write(hr_n.to_json(orient="split"))

    with open(os.path.join(write_path,"eda.json"), "w") as k:
        k.write(eda_n.to_json(orient="split"))

    with open(os.path.join(write_path,"bvp.json"), "w") as k:
        k.write(bvp_n.to_json(orient="split"))

print("Physio data Separations Completed.")


"""## Annotation Data


"""

import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import datetime
import math

# Dataset Paths

## Annotation Origin Paths

self_annotation_path = 'kemocon/raw/emotion_annotations/self_annotations'
part_annotation_path = 'kemocon/raw/emotion_annotations/partner_annotations'
extr_annotation_path = 'kemocon/raw/emotion_annotations/external_annotations'
aggr_annotation_path = 'kemocon/raw/emotion_annotations/aggregated_external_annotations'

## Diarization Data
vox_json = "kemocon/metadata/vox_dia"

## Timestamp Reading for Individual Participants
meta_path = 'kemocon/metadata/'
time_stamp = pd.read_csv(os.path.join(meta_path,"subjects.csv"))
time_stamp = time_stamp.set_index(np.arange(1,33,1))

## List of Participant IDs
debate_physio_path = 'kemocon/raw/e4_data'
pid_list = os.listdir(debate_physio_path)
rejected_pid = ['2','3','6','7','11','12','17','18','26','27','28']
selected_pid = [int(item) for item in pid_list if item not in rejected_pid]
selected_pid.sort()

## Annotation Write Path
annotation_json_path = 'kemocon/processed/annotation_data/'
os.makedirs(annotation_json_path,exist_ok=True)


"""#### Modified Annotation Segmentation [ 1s, 2.5s, 5 ]"""
print("Starting Annotation data Separations")
sl = [1]
for seg_len in sl :
      cnt = int(5/seg_len)
      for pid in selected_pid:
            # Number of Seconds of Annotation Samples
            sec = int((time_stamp.endTime[int(pid)] -  time_stamp.startTime[int(pid)])/1000)

            ### Segmentation of Annotation Data based on Diarization

            # Load Segmentation for Current Participant ID
            dia_path = os.path.join(vox_json,str(pid)+'.json')
            segments = pd.read_json(dia_path)

            # Trimming Diarization to resolution of seconds
            segments.start = segments.start/1000
            segments.end  = segments.end/1000

            
            ### Extraction of Self Annotation 
            
            
            # Load File
            s_path = os.path.join(self_annotation_path,'P'+str(pid)+'.self.csv')
            s = pd.read_csv(s_path)
            # Select Arousal and Valence
            s = s[['seconds','arousal','valence']]
            s.index += 1
            df_s = pd.DataFrame(index=np.arange(1,(len(s))*cnt + 1,1),columns=s.columns)
            # Extrapolate cnt seconds
            for i in list(s.index):
                pins = range((i - 1)* cnt + 1,(i - 1)* cnt + cnt + 1,1)
                for j in pins:
                    df_s.loc[j] = s.loc[i]

            end_sec = np.asarray(s.seconds)[-1]    
            df_s['seconds'] = np.arange(seg_len ,end_sec + seg_len, seg_len)
            df_s.columns = ['seconds','arousal','valence']

            df_n = pd.DataFrame(columns=df_s.columns) 

            # Extracting Annotation Segments relevent to Participant ID 
            for seg in list(segments.index):
                pd_start = segments['start'][seg]
                pd_end   = segments['end'][seg]
                a = df_s.loc[df_s.seconds.between(pd_start, pd_end, inclusive="both")].copy()
                df_n = df_n.append(a, ignore_index = True)

            # Dumping Path
            p_path = annotation_json_path + '/' +  str(seg_len) + '/self'
            os.makedirs(p_path, exist_ok=True)
            p1 = os.path.join(p_path,str(pid)+"_a.json")
            df_n.to_json(p1)
            del df_n, df_s

            
            
            ### Extraction of External  Annotation 


            # Load File
            s_path = os.path.join(aggr_annotation_path,'P'+str(pid)+'.external.csv')
            s = pd.read_csv(s_path)
            # Select Arousal and Valence
            s = s[['seconds','arousal','valence']]
            s.index += 1
            df_s = pd.DataFrame(index=np.arange(1,(len(s))*cnt +1,1),columns=s.columns)
            # Extrapolate cnt seconds
            for i in list(s.index):
                pins = range((i - 1)* cnt + 1,(i - 1)* cnt + cnt + 1,1)
                for j in pins:
                    df_s.loc[j] = s.loc[i]

            end_sec = np.asarray(s.seconds)[-1]    
            df_s['seconds'] = np.arange(seg_len ,end_sec + seg_len, seg_len)
            df_s.columns = ['seconds','arousal','valence']

            df_n = pd.DataFrame(columns=df_s.columns) 

            # Extracting Annotation Segments relevent to Participant ID 
            for seg in list(segments.index):
                pd_start = segments['start'][seg]
                pd_end   = segments['end'][seg]
                a = df_s.loc[df_s.seconds.between(pd_start, pd_end, inclusive="both")].copy()
                df_n = df_n.append(a, ignore_index = True)

            # Dumping Path
            p_path = annotation_json_path +  '/' +  str(seg_len) + '/extr'
            os.makedirs(p_path, exist_ok=True)
            p1 = os.path.join(p_path,str(pid)+"_a.json")
            df_n.to_json(p1)
            del df_n, df_s

print("Annotation data Separations Completed.")





"""# K-EmoCon FEATURIZATION

## Physio Featurization Pipeline

### Data Paths
"""

print(" Creating Physio Features ")
import json
import os
import math
import pandas as pd
import numpy as np
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import IPython
import urllib.request

import pydub
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import make_chunks

from scipy.io import wavfile
import soundfile as sfile


# Data Path Definitions
physio_data = 'kemocon/processed/physio_data_p'
# Audio files Path
audio_data = 'kemocon/processed/audio_data_p'

# Diarization Data
vox_json_path = "kemocon/metadata/vox_dia"
debate_physio_path = 'kemocon/raw/e4_data'
pid_list = os.listdir(debate_physio_path)
rejected_pid = ['2','3','6','7','11','12','17','18','26','27','28']
selected_pid = [int(item) for item in pid_list if item not in rejected_pid]
selected_pid.sort()
selected_pid = [15, 23, 30, 31]

#Feature write path
features ='kemocon/features/'
os.makedirs(features,exist_ok=True)

def feat_slice(df, f_data, seg_len, duration):
    df_feat = pd.DataFrame()
    start = 0
    duration  = duration * f_data * seg_len
    while start + int(f_data * seg_len) < duration:
        a = df[start : start + int(f_data * seg_len )].copy()
        a = a.reset_index(drop=True)
        df_feat = df_feat.append(a, ignore_index=True)
        start = start + int(f_data * seg_len )      
    return df_feat

"""### Feature Extraction"""

### Running Feature Extraction on All Participant segments
import warnings
warnings.filterwarnings("ignore")

f_bvp = 64
f_eda = 4

sl = [1]
for seg_len in sl :

      df_bvp = pd.DataFrame()
      df_eda = pd.DataFrame()

      df_p_b = pd.DataFrame()
      df_p_e = pd.DataFrame()

      # Looping over all Participants
      for pid in selected_pid:
          # Extracting segments path for Specific Participant
          pid_path = os.path.join(physio_data,str(pid))
          
          aud_file = os.path.join(audio_data,'pid_' + str(pid) + '.wav')
          signal = AudioSegment.from_file(aud_file , "wav")

          # Extracting BVP Data
          bvp_path = pid_path + '/bvp.json'
          bvp = pd.read_json(bvp_path,orient = 'split')
          bvp = bvp['value']

          # Extracting EDA Data
          eda_path = pid_path + '/eda.json' 
          eda = pd.read_json(eda_path,orient = 'split')
          eda = eda['value']   

          a_len = len(signal)/(seg_len * 1000)         
          s_bvp = math.floor(bvp.shape[0]/f_bvp/seg_len)
          s_eda = math.floor(eda.shape[0]/f_eda/seg_len)
          s_len = min(s_bvp, s_eda, a_len)

          # Feature Extractions
          feat_bvp = feat_slice(bvp,f_bvp, seg_len, s_len)
          feat_eda = feat_slice(eda,f_eda, seg_len, s_len)
                  
          pid_b = [pid] * len(feat_bvp)
          pid_e = [pid] * len(feat_bvp)

          df_p_b = df_p_b.append(pid_b, ignore_index=True)
          df_p_e = df_p_e.append(pid_e, ignore_index=True)

          df_bvp = df_bvp.append(feat_bvp, ignore_index=True)
          df_eda = df_eda.append(feat_eda, ignore_index=True)

          # print(str(pid) + "  " + str(a_len) + "  " + str(s_bvp) + "   " + str(s_eda))
          

      df_p_b = df_p_b.rename(columns={0: "pid"})
      df_p_e = df_p_e.rename(columns={0: "pid"})

      df_bvp = pd.concat([df_p_b, df_bvp], axis=1)
      df_eda = pd.concat([df_p_e, df_eda], axis=1)


      feat = features + '_' + str(seg_len) + 's' 
      os.makedirs(feat, exist_ok=True)

      # Dumping Path
      p1 = os.path.join(feat,"bvp.hdf5")
      df_bvp.to_hdf(p1,  key='df', mode='w')

      p1 = os.path.join(feat,"eda.hdf5")
      df_eda.to_hdf(p1,  key='df', mode='w')

      names = ['k','l','o','h']

      for i in names:
          f_path = feat + '_' + i
          os.makedirs(f_path,exist_ok=True)
          
          p1 = os.path.join(f_path,"bvp.hdf5")
          df_bvp.to_hdf(p1,  key='df', mode='w')

          p1 = os.path.join(f_path,"eda.hdf5")
          df_eda.to_hdf(p1,  key='df', mode='w')

      del df_eda, df_bvp, feat_bvp, feat_eda, df_p_b, df_p_e
      
      
print(" Physio Featurization Complete.")

print(" Creating Audio Features ")
"""## Audio Featureization Pipeline

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

"""### Data Paths"""

# Data Path Definitions
audio_processed = 'kemocon/processed/audio_data_p/'
demand_data = 'demand/'
debate_physio_path = 'kemocon/raw/e4_data'
pid_list = os.listdir(debate_physio_path)
rejected_pid = ['2','3','6','7','11','12','17','18','26','27','28']
selected_pid = [int(item) for item in pid_list if item not in rejected_pid]
selected_pid.sort()
selected_pid = [15, 23, 30, 31]

#Feature write path
features ='kemocon/features/' 
os.makedirs(features,exist_ok=True)

"""### Feature Definitions

#### ComParE2016 _ eGeMAPSv02 
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

def spectrogram_melfreq(seg, my_dpi = 256):
    
    ## #Speech/Music contemporary values
    n_mels = 512      # Typical Mel Bands for speech [Tested for best discrimination]
    n_fft = 2048      # Window Length for FFT (Speech/Music contemporary values)
    hop_length = 512  # Amount of Shift per FFT
    win_length = n_fft # math.floor(sr * 0.03)
    
    ## Get Spectrogram
    wave, sr = librosa.load(seg)
    S = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=n_mels , hop_length = hop_length, n_fft = n_fft)
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
    img2 = img2/255.0
    
    return img2


"""### Raw Audio Feature Extraction"""

def segment_chunks(signal_n, seg):
    seg_len = seg * 1000                    # pydub calculates in millisec
    chunks = make_chunks(signal_n, seg_len) # Make chunks of 'seg' seconds
    for i in np.arange(0,1,len(chunks)):
        if (chunks[i].duration_seconds != (seg * 1000)):
            del chunks[i]
    return chunks



### Running Feature Extraction on All Participant

## ## First  set of Features - eGeMAPsv02 + ComPaRE2016 + FBanks

sl = [1]
for seg_len in sl :
        # Feature Dataframe Initializations
        feat_com_t    = pd.DataFrame()
        feat_egemap_t = pd.DataFrame()
        feat_spec_mf_t  = pd.DataFrame()

        # Looping over all Participants
        for pid in selected_pid:

              ## Extracting Audio from Specific Participant
              aud_file = audio_processed +  'pid_' + str(pid) + '.wav'
              signal = AudioSegment.from_file(aud_file , "wav") 
              
              ## Slicing Audio in Chunks
              aud_chunks = segment_chunks(signal,seg = seg_len)
              
              
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
                      feat_spec_mf = feat_spec_mf.append([[spectrogram_melfreq(seg, my_dpi = 256)]])    
                                  
                      os.remove('segment.wav')        
                      del s, seg

              ## Assign Participant ID to the Samples
              feat_com['pid'] = pid
              feat_egemap['pid'] = pid
              feat_spec_mf['pid'] = pid

              ## Append the Collected Features from samples to Master 
              feat_com_t = feat_com_t.append(feat_com)
              feat_egemap_t = feat_egemap_t.append(feat_egemap)
              feat_spec_mf_t = feat_spec_mf_t.append(feat_spec_mf)

              del feat_com, feat_egemap, feat_spec_mf, aud_file, signal, aud_chunks

        ## Assign Participant ID to the Samples
        feat_com_t['pid'] = feat_com_t['pid'].astype(int)
        feat_egemap_t['pid'] = feat_egemap_t['pid'].astype(int)
        feat_spec_mf_t['pid'] = feat_spec_mf_t['pid'].astype(int)


        # Reset Column Titles
        feat_spec_mf_t = feat_spec_mf_t.rename(columns={0: "MelFreq Spectrogram"})

        ## Reset Indices for the Dataframes
        feat_com_t.reset_index(drop=True, inplace=True)
        feat_egemap_t.reset_index(drop=True, inplace=True)
        feat_spec_mf_t.reset_index(drop=True, inplace=True)


        feat = features + '_' + str(seg_len) + 's' 
        os.makedirs(feat, exist_ok=True)

        # Dumping Path
        p1 = os.path.join(feat,"compar.hdf5")
        feat_com_t.to_hdf(p1,  key='df', mode='w') #, format='table')

        p1 = os.path.join(feat,"egemap.hdf5")
        feat_egemap_t.to_hdf(p1,  key='df', mode='w') #, format='table')

        p1 = os.path.join(feat,"mat_melfreq.hdf5")
        feat_spec_mf_t.to_hdf(p1,  key='df', mode='w')#, format='table')

        del feat_com_t, feat_egemap_t, feat_spec_mf_t



print(" Audio Featurization Complete.")




"""### Noisy Audio Feature Extraction"""

import glob
noisy_audios = [AudioSegment.from_wav(wav_file) for wav_file in glob.glob(demand_data+'*.wav')]

### Running Feature Extraction on All Participant

## ## Features - eGeMAPsv02 + ComPaRE2016 + MelFreq Spectrograms

names = ['k','l','o','h']
sl = [1]

print(" Creating Noisy Audio Features.  ")

for seg_len in sl :
    cnt = 0
    # Looping over all Participants
    for noise in noisy_audios:
     
          # Feature Dataframe Initializations
          feat_com_t    = pd.DataFrame()
          feat_egemap_t = pd.DataFrame()
          feat_spec_mf_t  = pd.DataFrame()

          for pid in selected_pid:

                ## Extracting Audio from Specific Participant
                aud_file = audio_processed +  'pid_' + str(pid) + '.wav'
                signal_n = AudioSegment.from_file(aud_file , "wav") 

                signal = signal_n.overlay(noise, gain_during_overlay = 0)

                ## Slicing Audio in Chunks
                aud_chunks = segment_chunks(signal,seg = seg_len)
                          
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

                        # Extracting Spectrograms and Appending location names #5
                        feat_spec_mf = feat_spec_mf.append([[spectrogram_melfreq(seg,  my_dpi = 256)]])        
                                    
                        os.remove('segment.wav')        
                        del s, seg

                ## Assign Participant ID to the Samples
                feat_com['pid'] = pid
                feat_egemap['pid'] = pid
                feat_spec_mf['pid'] = pid

                ## Append the Collected Features from samples to Master 
                feat_com_t = feat_com_t.append(feat_com)
                feat_egemap_t = feat_egemap_t.append(feat_egemap)
                feat_spec_mf_t = feat_spec_mf_t.append(feat_spec_mf)

                del feat_com, feat_egemap, feat_spec_mf, aud_file, signal, aud_chunks
          
          ## Assign Participant ID to the Samples
          feat_com_t['pid'] = feat_com_t['pid'].astype(int)
          feat_egemap_t['pid'] = feat_egemap_t['pid'].astype(int)
          feat_spec_mf_t['pid'] = feat_spec_mf_t['pid'].astype(int)
          
          # Reset Column Titles
          feat_spec_mf_t = feat_spec_mf_t.rename(columns={0: "MelFreq Spectrogram"})

          ## Reset Indices for the Dataframes
          feat_com_t.reset_index(drop=True, inplace=True)
          feat_egemap_t.reset_index(drop=True, inplace=True)
          feat_spec_mf_t.reset_index(drop=True, inplace=True)

          # Dumping Path
          f_path = features + '_' + str(seg_len) + 's' + '_' + names[cnt]
          os.makedirs(f_path, exist_ok=True)

          p1 = os.path.join(f_path,"compar.hdf5")
          feat_com_t.to_hdf(p1,  key='df', mode='w') #, format='table')

          p1 = os.path.join(f_path,"egemap.hdf5")
          feat_egemap_t.to_hdf(p1,  key='df', mode='w') #, format='table')

          p1 = os.path.join(f_path,"mat_melfreq.hdf5")
          feat_spec_mf_t.to_hdf(p1,  key='df', mode='w')#, format='table')

          cnt = cnt + 1

          del feat_com_t, feat_egemap_t, feat_spec_mf_t

print(" Noisy Audio Featurization Complete.")



"""## Label Extraction"""

print(" Label Extraction. ")

debate_physio_path = 'kemocon/raw/e4_data'
pid_list = os.listdir(debate_physio_path)
rejected_pid = ['2','3','6','7','11','12','17','18','26','27','28']
selected_pid = [int(item) for item in pid_list if item not in rejected_pid]
selected_pid.sort()
selected_pid = [15, 23, 30, 31]

## Combined Dataset Locations

## Annotation Path
annotation_json_path = 'kemocon/processed/annotation_data/'

## Feature Path
features = 'kemocon/features/'
os.makedirs(features,exist_ok=True)

"""### Binary Labels"""

def binary_def(val):
    if val < 3:
        l = 0
    if val > 3:
        l = 1
    if val == 3:
        l = np.nan
    return l

## Cleaning Labels by availablability of audio

sl = [1]
for seg_len in sl :
  for ann in ['self', 'extr']:
      labels = pd.DataFrame()
      for pid in selected_pid:
          ## Label Stack with corresponding participants
          p_path = annotation_json_path + '/' + str(seg_len) + '/' + ann + '/' 
          path1 = os.path.join(p_path,str(pid)+"_a.json")
          df = pd.read_json(path1)
          df['pid'] = int(pid)
          labels = labels.append(df, ignore_index=True)

      labels['binary_arousal'] = labels.apply(lambda labels : binary_def(labels.arousal), axis = 1)
      labels['binary_valence'] = labels.apply(lambda labels : binary_def(labels.valence), axis = 1)

      ## ## Writing Quadrant Labels to Drive
      f_path = features + '_' + str(seg_len) + 's' + '/'
      p1 = os.path.join(f_path,ann+"_labels.hdf5")
      labels.to_hdf(p1,  key='df', mode='w')

      names = ['k','l','o','h']
      for i in names:
          f_path = features +'/_' + str(seg_len) + 's' '_' + i
          p1 = os.path.join(f_path,ann + "_labels.hdf5")
          labels.to_hdf(p1,  key='df', mode='w')

      del labels, df

print(" Featurization Complete. ")



""" K-EmoCon Feature And Label Corrections

Check Dimensions of Features and Annotation Data and Correct for inconsistencies per participant ###


"""
## Check List

import glob 
import pandas as pd
import os 
import tables
import gc

## Feature Path
features = 'kemocon/features/'

list_f = glob.glob(features + '*/*.hdf5')
list_f.sort()

for i in list_f:
    v = pd.read_hdf(i)
    print(i + '  :  ' +str(v.shape[0]))
 


"""#### Label and Feature Corrections"""

# Data Path Definitions

features_c = 'kemocon/features_c/'
os.makedirs(features_c,exist_ok=True)

physio_data = 'kemocon/processed/physio_data_p'
debate_physio_path = 'kemocon/raw/e4_data'
pid_list = os.listdir(debate_physio_path)
rejected_pid = ['2','3','6','7','11','12','17','18','26','27','28']
selected_pid = [int(item) for item in pid_list if item not in rejected_pid]
selected_pid.sort()
selected_pid = [15, 23, 30, 31]

print("Correcting Basic Features") 
sl = [1]
for seg_len in sl:
        ## ## Writing Quadrant Labels to Drive
        f_path = features + '_' + str(seg_len) + 's'
        
        ## Annotation Labels Load
        specm = pd.read_hdf(os.path.join(f_path,'mat_melfreq.hdf5'))        
        compar  =  pd.read_hdf(os.path.join(f_path,'compar.hdf5'))
        egemap =  pd.read_hdf(os.path.join(f_path,'egemap.hdf5'))
        eda = pd.read_hdf(os.path.join(f_path,'eda.hdf5'))
        bvp = pd.read_hdf(os.path.join(f_path,'bvp.hdf5'))
        self_labels = pd.read_hdf(os.path.join(f_path,'self_labels.hdf5'))
        extr_labels = pd.read_hdf(os.path.join(f_path,'extr_labels.hdf5'))
        
        print(" Raw Feature count per participant")
        print('I :   sl   el   pm    cp    eg     ed    bv')
        for pid in selected_pid:
          print(str(pid) + " :  " +  str(len(self_labels[self_labels['pid'] == pid])) + "   "+  str(len(extr_labels[extr_labels['pid'] == pid])) + "    " + str(len(specm[specm['pid'] == pid])) + "    " + str(len(compar[compar['pid'] == pid]))+ "    " + str(len(egemap[egemap['pid'] == pid]))  + "    " +str(len(eda[eda['pid'] == pid]))       + "    " + str(len(bvp[bvp['pid'] == pid])))


        slabels_t = pd.DataFrame()
        elabels_t = pd.DataFrame()
        specm_t = pd.DataFrame()
        compar_t = pd.DataFrame()
        egemap_t = pd.DataFrame()
        eda_t = pd.DataFrame()
        bvp_t = pd.DataFrame()

        print("Modifying Feature-Label Count per participant")
        for pid in selected_pid:
          lengths = [len(self_labels[self_labels['pid'] == pid]),len(extr_labels[extr_labels['pid'] == pid]), len(specm[specm['pid'] == pid]), len(egemap[egemap['pid'] == pid]), len(compar[compar['pid'] == pid]),  len(eda[eda['pid'] == pid]), len(bvp[bvp['pid'] == pid])]
          s = min(lengths)
          slabels_t = slabels_t.append(self_labels[self_labels['pid'] == pid][:s])
          elabels_t = elabels_t.append(extr_labels[extr_labels['pid'] == pid][:s])
          specm_t = specm_t.append(specm[specm['pid'] == pid][:s])
          compar_t = compar_t.append(compar[compar['pid'] == pid][:s])
          egemap_t = egemap_t.append(egemap[egemap['pid'] == pid][:s])
          eda_t  = eda_t.append(eda[eda['pid'] == pid][:s])
          bvp_t  = bvp_t.append(bvp[bvp['pid'] == pid][:s])

        ## Reset Indices for the Dataframes
        specm_t.reset_index(drop=True, inplace=True)
        compar_t.reset_index(drop=True, inplace=True)
        egemap_t.reset_index(drop=True, inplace=True)       
        eda_t.reset_index(drop=True, inplace=True)
        bvp_t.reset_index(drop=True, inplace=True)
        slabels_t.reset_index(drop=True, inplace=True)
        elabels_t.reset_index(drop=True, inplace=True)



        print('I :   sl   el   pm    cp    eg     ed    bv')
        for pid in selected_pid:
          print(str(pid) + " :  " +  str(len(slabels_t[slabels_t['pid'] == pid])) + "   "+  str(len(elabels_t[elabels_t['pid'] == pid])) + "    " + str(len(specm_t[specm_t['pid'] == pid])) + "    " + str(len(compar_t[compar_t['pid'] == pid]))+ "    " + str(len(egemap_t[egemap_t['pid'] == pid]))  + "    " +str(len(eda_t[eda_t['pid'] == pid]))       + "    " + str(len(bvp_t[bvp_t['pid'] == pid])))

        f_path = features_c + '_' + str(seg_len) + 's'
        os.makedirs(f_path, exist_ok=True)
        
        p1 = os.path.join(f_path,"bvp.hdf5")
        bvp_t.to_hdf(p1,  key='df', mode='w')

        p1 = os.path.join(f_path,"eda.hdf5")
        eda_t.to_hdf(p1,  key='df', mode='w')

        p1 = os.path.join(f_path,"mat_melfreq.hdf5")
        specm_t.to_hdf(p1,  key='df', mode='w')#, format='table')
        
        p1 = os.path.join(f_path,"compar.hdf5")
        compar_t.to_hdf(p1,  key='df', mode='w')#, format='table')
        
        p1 = os.path.join(f_path,"egemap.hdf5")
        egemap_t.to_hdf(p1,  key='df', mode='w')#, format='table')
        
        p1 = os.path.join(f_path,"self_labels.hdf5")
        slabels_t.to_hdf(p1,  key='df', mode='w')#, format='table')

        p1 = os.path.join(f_path,"extr_labels.hdf5")
        elabels_t.to_hdf(p1,  key='df', mode='w')#, format='table')

        tables.file._open_files.close_all()

        del specm, eda, bvp, self_labels, extr_labels 
        del specm_t, eda_t, bvp_t, slabels_t, elabels_t
        tables.file._open_files.close_all()
        

print("Correcting Noisy Features") 
sl = [1]
names = ['k','l','o','h']
for n in names:
      for seg_len in sl:
            ## ## Writing Quadrant Labels to Drive
            f_path = features + '_' + str(seg_len) + 's' + '_' + n

            ## Annotation Labels Load
            specm = pd.read_hdf(os.path.join(f_path,'mat_melfreq.hdf5'))        
            compar  =  pd.read_hdf(os.path.join(f_path,'compar.hdf5'))
            egemap =  pd.read_hdf(os.path.join(f_path,'egemap.hdf5'))
            eda = pd.read_hdf(os.path.join(f_path,'eda.hdf5'))
            bvp = pd.read_hdf(os.path.join(f_path,'bvp.hdf5'))
            self_labels = pd.read_hdf(os.path.join(f_path,'self_labels.hdf5'))
            extr_labels = pd.read_hdf(os.path.join(f_path,'extr_labels.hdf5'))
            print(" Raw Feature count per participant")
            print('I :   sl   el   pm    cp    eg     ed    bv')
            for pid in selected_pid:
              print(str(pid) + " :  " +  str(len(self_labels[self_labels['pid'] == pid])) + "   "+  str(len(extr_labels[extr_labels['pid'] == pid])) + "    " + str(len(specm[specm['pid'] == pid])) + "    " + str(len(compar[compar['pid'] == pid]))+ "    " + str(len(egemap[egemap['pid'] == pid]))  + "    " +str(len(eda[eda['pid'] == pid]))       + "    " + str(len(bvp[bvp['pid'] == pid])))


            slabels_t = pd.DataFrame()
            elabels_t = pd.DataFrame()
            specm_t = pd.DataFrame()
            compar_t = pd.DataFrame()
            egemap_t = pd.DataFrame()
            eda_t = pd.DataFrame()
            bvp_t = pd.DataFrame()


            print("Modifying Feature-Label Count per participant")
            for pid in selected_pid:
              lengths = [len(self_labels[self_labels['pid'] == pid]),len(extr_labels[extr_labels['pid'] == pid]), len(specm[specm['pid'] == pid]), len(egemap[egemap['pid'] == pid]), len(compar[compar['pid'] == pid]),  len(eda[eda['pid'] == pid]), len(bvp[bvp['pid'] == pid])]
              s = min(lengths)
              slabels_t = slabels_t.append(self_labels[self_labels['pid'] == pid][:s])
              elabels_t = elabels_t.append(extr_labels[extr_labels['pid'] == pid][:s])
              specm_t = specm_t.append(specm[specm['pid'] == pid][:s])
              compar_t = compar_t.append(compar[compar['pid'] == pid][:s])
              egemap_t = egemap_t.append(egemap[egemap['pid'] == pid][:s])
              eda_t  = eda_t.append(eda[eda['pid'] == pid][:s])
              bvp_t  = bvp_t.append(bvp[bvp['pid'] == pid][:s])

            ## Reset Indices for the Dataframes
            specm_t.reset_index(drop=True, inplace=True)
            compar_t.reset_index(drop=True, inplace=True)
            egemap_t.reset_index(drop=True, inplace=True)       
            eda_t.reset_index(drop=True, inplace=True)
            bvp_t.reset_index(drop=True, inplace=True)
            slabels_t.reset_index(drop=True, inplace=True)
            elabels_t.reset_index(drop=True, inplace=True)



            print('I :   sl   el   pm    cp    eg     ed    bv')
            for pid in selected_pid:
              print(str(pid) + " :  " +  str(len(slabels_t[slabels_t['pid'] == pid])) + "   "+  str(len(elabels_t[elabels_t['pid'] == pid])) + "    " + str(len(specm_t[specm_t['pid'] == pid])) + "    " + str(len(compar_t[compar_t['pid'] == pid]))+ "    " + str(len(egemap_t[egemap_t['pid'] == pid]))  + "    " +str(len(eda_t[eda_t['pid'] == pid]))       + "    " + str(len(bvp_t[bvp_t['pid'] == pid])))

            f_path = features_c + '_' + str(seg_len) + 's' + '_' + n
            os.makedirs(f_path, exist_ok=True)
            
            p1 = os.path.join(f_path,"bvp.hdf5")
            bvp_t.to_hdf(p1,  key='df', mode='w')

            p1 = os.path.join(f_path,"eda.hdf5")
            eda_t.to_hdf(p1,  key='df', mode='w')

            p1 = os.path.join(f_path,"mat_melfreq.hdf5")
            specm_t.to_hdf(p1,  key='df', mode='w')#, format='table')
            
            p1 = os.path.join(f_path,"compar.hdf5")
            compar_t.to_hdf(p1,  key='df', mode='w')#, format='table')
            
            p1 = os.path.join(f_path,"egemap.hdf5")
            egemap_t.to_hdf(p1,  key='df', mode='w')#, format='table')
            
            p1 = os.path.join(f_path,"self_labels.hdf5")
            slabels_t.to_hdf(p1,  key='df', mode='w')#, format='table')

            p1 = os.path.join(f_path,"extr_labels.hdf5")
            elabels_t.to_hdf(p1,  key='df', mode='w')#, format='table')

            tables.file._open_files.close_all()

            del specm, eda, bvp, self_labels, extr_labels 
            del specm_t, eda_t, bvp_t, slabels_t, elabels_t
            tables.file._open_files.close_all()
