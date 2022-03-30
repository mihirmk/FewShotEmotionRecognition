""" RECOLA Few Shot Learning - Part 1 - All Models


"""

## Library Imports

### Tensorflow Utils


import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Subtract, Multiply, Add, Average
from tensorflow.keras.layers import Input, LSTM, GRU, Lambda, Dense, Dropout, Flatten,Concatenate,Reshape,Embedding
from tensorflow.keras.layers import ConvLSTM2D, Conv1D, MaxPool1D,Conv2D, MaxPool2D, TimeDistributed,SimpleRNN, Conv2DTranspose,UpSampling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LeakyReLU, ReLU, BatchNormalization, InputSpec,RepeatVector, Layer, Reshape, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, History, LearningRateScheduler, ModelCheckpoint

from tensorflow.keras import initializers, optimizers, losses

from tensorflow.keras.callbacks import Callback


"""### General Utils"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import IPython
import urllib.request
import datetime
import os, errno
from pathlib import Path
import uuid

import glob
import gc

import re
import random
from PIL import Image

import math
import time
import shutil
import tqdm

import pickle as pkl

import h5py
import tables

# import os,sys,humanize,psutil,GPUtil
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_recall_fscore_support, roc_curve
from sklearn.metrics import  auc
from sklearn_pandas import DataFrameMapper
import seaborn as sns




import librosa
import librosa.display

"""# FSL Data Creation"""

"""#### Create Dataset"""

   
def create_dataset(df_spec, df_eda, df_bvp, df_ege, labels, shots, pid, f_path, label_tag):

    df_a = df_spec['MelFreq Spectrogram']; df_e = df_eda; df_b = df_bvp; df_eg = df_ege.drop(columns=['pid']); labels_n = labels[label_tag]
    
    
    df_e.columns = df_e.columns.map(str)
    df_b.columns = df_b.columns.map(str)

    eg_m = DataFrameMapper([(df_eg.columns,[MinMaxScaler(),StandardScaler()])])
    e_m = DataFrameMapper([(df_e.columns,[MinMaxScaler(),StandardScaler()])])
    b_m = DataFrameMapper([(df_b.columns,[MinMaxScaler(),StandardScaler()])])
    
    df_e2 = e_m.fit_transform(df_e.copy())
    df_b2 = b_m.fit_transform(df_b.copy())    
    df_eg2 = eg_m.fit_transform(df_eg.copy())

    df_a = df_a.copy()/255.0
    df_e = pd.DataFrame(df_e2, index = df_e.index, columns=df_e.columns)
    df_b = pd.DataFrame(df_b2, index = df_b.index, columns=df_b.columns)
    df_eg = pd.DataFrame(df_eg2, index = df_eg.index, columns=df_eg.columns)   
    
    
    aud_tr = []
    eda_tr = []
    bvp_tr = []
    ege_tr = []
    y_tr = []

    aud_ts = []
    eda_ts = []
    bvp_ts = []
    ege_ts = []
    y_ts = []
    

    for i in labels_n.unique().tolist():
            
          l = labels_n[labels_n == i]
          a = df_a[labels_n == i]
          e = df_e[labels_n == i]
          b = df_b[labels_n == i]
          eg = df_eg[labels_n == i]

          tr = random.choices(l.index,k=shots)

          ts = list(set(l.index) - set(tr))

          a_tr = a.loc[tr]
          e_tr = e.loc[tr]
          b_tr = b.loc[tr]
          eg_tr = eg.loc[tr]
          l_tr = l.loc[tr]
          
          a_ts = a.loc[ts]
          e_ts = e.loc[ts]
          b_ts = b.loc[ts]
          eg_ts = eg.loc[ts]
          l_ts = l.loc[ts]


          ### Training Data Set-Up
          aud_tr.extend(a_tr.values.tolist())
          eda_tr.extend(e_tr.values.tolist())
          bvp_tr.extend(b_tr.values.tolist())
          ege_tr.extend(eg_tr.values.tolist())
          y_tr.extend(l_tr.values.tolist())

          ### Testing Data Set-Up
          aud_ts.extend(a_ts.values.tolist())
          eda_ts.extend(e_ts.values.tolist())
          bvp_ts.extend(b_ts.values.tolist())
          ege_ts.extend(eg_ts.values.tolist())
          y_ts.extend(l_ts.values.tolist())


    aud_tr = np.asarray(aud_tr)
    eda_tr = np.asarray(eda_tr)
    bvp_tr = np.asarray(bvp_tr)
    ege_tr = np.asarray(ege_tr)
    y_tr = np.asarray(y_tr).astype('int32')

    aud_ts = np.asarray(aud_ts)
    eda_ts = np.asarray(eda_ts)
    bvp_ts = np.asarray(bvp_ts)
    ege_ts = np.asarray(ege_ts)    
    y_ts = np.asarray(y_ts).astype('int32')


    return aud_tr, aud_ts, eda_tr, eda_ts, bvp_tr, bvp_ts, ege_tr, ege_ts, y_tr, y_ts


"""#### Make Pairs"""

def make_pairs(a, e, b, eg, y):
    
    c = np.unique(y)
    c_count = len(c)
    digit_indices = [np.where(y == i)[0] for i in c]

    a_pairs = []
    e_pairs = []
    b_pairs = []
    eg_pairs = []
    labels = []


    for idx1 in range(len(y)):      
        a1 = a[idx1]
        e1 = e[idx1]
        b1 = b[idx1]
        eg1 = eg[idx1] 
        label1 = y[idx1]

        ## add a matching example
        idx2 = random.choice(digit_indices[label1])
        a2 = a[idx2]
        e2 = e[idx2]
        b2 = b[idx2]
        eg2 = eg[idx2]
        a_pairs += [[a1, a2]]
        e_pairs += [[e1, e2]]
        b_pairs += [[b1, b2]]
        eg_pairs += [[eg1, eg2]]
        labels += [1]


        # add a non-matching example
        label2 = random.randint(0, c_count - 1)
        while label2 == label1:
            label2 = random.randint(0, c_count - 1)
        
        idx2 = random.choice(digit_indices[label2])
        a2 = a[idx2]
        e2 = e[idx2]
        b2 = b[idx2]
        eg2 = eg[idx2]
        a_pairs += [[a1, a2]]
        e_pairs += [[e1, e2]]
        b_pairs += [[b1, b2]]
        eg_pairs += [[eg1, eg2]]
        labels += [0]
    
    ## Formatting to Array
    a_pairs = np.asarray(a_pairs)
    b_pairs = np.asarray(b_pairs)
    e_pairs = np.asarray(e_pairs)
    eg_pairs = np.asarray(eg_pairs)
    labels = np.asarray(labels).astype('float32')

    return a_pairs, e_pairs, b_pairs, eg_pairs, labels

"""### Reshape pairs"""

def reshape_pairs(ra, re, rb, ree,  labels):
      ##  Rehasing as individial pairs
      a1 = ra[:,0]
      a2 = ra[:,1]

      e_1 = re[:,0]
      e_2 = re[:,1]
      ## Correcting size of Arrays
      e1 = e_1.reshape((e_1.shape[0], e_1.shape[1],1))
      e2 = e_2.reshape((e_2.shape[0], e_2.shape[1],1))

      b_1 = rb[:,0]
      b_2 = rb[:,1]
      ## Correcting size of Arrays
      b1 = b_1.reshape((b_1.shape[0], b_1.shape[1],1))
      b2 = b_2.reshape((b_2.shape[0], b_2.shape[1],1))

      eg_1 = ree[:,0]
      eg_2 = ree[:,1]
      ## Correcting size of Arrays
      eg1 = eg_1.reshape((eg_1.shape[0], eg_1.shape[1],1))
      eg2 = eg_2.reshape((eg_2.shape[0], eg_2.shape[1],1))
      labels = labels.reshape((labels.shape[0], 1))
      
      return a1, a2, e1, e2, b1, b2, eg1, eg2, labels

"""# NxK Shot Learning - Siamese Nets with Contrastive Loss

### Accuracy
"""

def compute_accuracy(lab, y_pr):
    correct_predictions = (np.sum(y_pr == lab) * 100 )/len(y_pr)
    return correct_predictions

"""### Loss Functions"""

def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)
    return contrastive_loss

"""### Distance Metrics"""

def euclidean_distance(vecs):
    x, y = vecs
    sum_square = K.sum(K.square(x - y), axis = 1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def square_euclidean_distance(vecs):
    x, y = vecs
    sum_square = K.sum(K.square(x - y), axis = 1, keepdims=True)
    return K.maximum(sum_square, K.epsilon())

def half_square_euclidean_distance(vecs):
    x, y = vecs
    sum_square = K.sum(K.square(x - y), axis = 1, keepdims=True)
    return max(0.5 * K.maximum(sum_square, K.epsilon())) #, self.e)

"""### Plot Functions"""


def plots1_a1(history, metric1, metric2, label, pid, base_dir, model_name, i, has_valid=True):
    
    fig, ax = plt.subplots(1, 2,figsize=(20,7))

    ax[0].plot(history[metric1], color = 'r', label=metric1)
    if has_valid:
      ax[0].plot(history["val_"+metric1], color = 'b', label="val_"+metric1)
    ax[0].set_xlabel("epochs", fontsize=17)
    ax[0].set_ylabel(metric1, fontsize=17)
    ax[0].set_ylim(ymin=0)
    ax[0].legend(["Support", "Query"], loc="best")
    ax[0].grid(True)

    ax[1].plot(history[metric2], color = 'r', label=metric2)
    if has_valid:
      ax[1].plot(history["val_"+metric2], color = 'b', label="val_"+metric2)
    ax[1].set_xlabel("epochs", fontsize=17)
    ax[1].set_ylabel(metric2, fontsize=17)
    ax[1].set_ylim(ymin=0)
    ax[1].legend(["Support", "Query"], loc="best")
    ax[1].grid(True)

    fig.suptitle(label, fontsize=16)
    plt.savefig(base_dir + model_name + '_' + str(shots) + '_h.png')
    plt.close()


def plots2_cm(lab, y_pr, pid, plot_label, base_dir):


    print("Confusion Matrix on Support Set")
    #Print Confusion Matrix
    cm = confusion_matrix(lab, y_pr)
    labels = ['0', '1']
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin = 0.2);
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(base_dir + plot_label+'_cm.png')
    plt.close()

    #Print Area Under Curve
    false_positive_rate, recall, thresholds = roc_curve(lts, predictions)
    roc_auc = auc(false_positive_rate, recall)
    plt.figure()
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out (1-Specificity)')
    plt.savefig(base_dir + plot_label + '_auc.png')
    plt.close()


def plots3_s(base_dir, model_name, batch_size_variants, set_name):
        ## Plotting Valence Results for a set

        NUM_COLORS = 10
        cm = plt.get_cmap('tab20')

        ## Plot of Contrastive Loss
        fig, ax = plt.subplots(1, 3,figsize=(24, 8))
        ax[0].set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        ax[1].set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        ax[2].set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])


        
        for i in batch_size_variants:
            a = base_dir + model_name  + '_' + str(i) + '_lr.npy'
            lr_new = np.load(a, allow_pickle=True) #.item()
            
            a = base_dir + model_name  + '_' + str(i) + '_sl.npy'
            df = np.load(a, allow_pickle=True) #.item()

            a = base_dir + model_name  + '_' + str(i) + '_lrb.npy'
            lr_best = np.load(a, allow_pickle=True) #.item()           
            
            a = base_dir + model_name  + '_' + str(i) + '_lrv.npy'
            y_best = np.load(a, allow_pickle=True) #.item() 

            lmin_index = np.argmin(df)

            ax[0].plot(lr_new, df, label=  str(i)) 
            ax[0].legend(loc="best")
            ax[0].set_xscale('log')
            # ax[0].set_ylim(ymin=0)
            # ax[0].set_ylim(ymax=0.6) 
            ax[0].grid(True, which="major", ls="-")
            ax[0].set_xlabel("Learning Rate", fontsize=17)
            ax[0].set_ylabel("Support Set Loss", fontsize=17)                
            ax[0].text(lr_best,10,'%d',rotation=90)
            # ax[0].axhspan(0, lmin_index, facecolor='cyan', alpha=0.5)

            a = base_dir + model_name  + '_' + str(i) + '_dr.npy'
            df = np.load(a, allow_pickle=True) #.item()

            ax[1].plot(lr_new, df, label= str(i)) 
            ax[1].legend(loc="best")
            ax[1].set_xscale('log')
            # ax[1].set_ylim(ymin=0)
            # ax[1].set_ylim(ymax=0.6)
            ax[1].grid(True, which="major", ls="-")            
            ax[1].set_xlabel("Learning Rate", fontsize=17)
            ax[1].set_ylabel("Derivative of Support Set Loss", fontsize=17)

            a = base_dir + model_name  + '_' + str(i) + '_ql.npy'
            df = np.load(a, allow_pickle=True) #.item()

            ax[2].plot(lr_new, df, label= str(i)) 
            ax[2].legend(loc="best")
            ax[2].set_xscale('log')
            # ax[2].set_ylim(ymin=0)
            # ax[2].set_ylim(ymax=0.6) 
            ax[2].grid(True, which="major", ls="-")
            ax[2].set_xlabel("Learning Rate", fontsize=17)
            ax[2].set_ylabel("Query Set Loss", fontsize=17)
            ax[2].scatter(x = lr_best, y = y_best, marker = 'o')

        plt.savefig(base_dir + model_name + '_Loss.png')        
        plt.close()
        
        ## Plot of Binary Accuracy
        fig, ax = plt.subplots(1, 2,figsize=(20, 8))
        ax[0].set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        ax[1].set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])


        for i in batch_size_variants:
            a = base_dir + model_name  + '_' + str(i) + '_lr.npy'
            lr_new = np.load(a, allow_pickle=True) #.item()

            a = base_dir + model_name  + '_' + str(i) + '_sa.npy'
            df = np.load(a, allow_pickle=True) #.item()

            a = base_dir + model_name  + '_' + str(i) + '_lrb.npy'
            lr_best = np.load(a, allow_pickle=True) #.item()                    

            a = base_dir + model_name  + '_' + str(i) + '_lrv.npy'
            y_best = np.load(a, allow_pickle=True) #.item()           

            ax[0].plot(lr_new, df, label=  str(i)) 
            ax[0].legend(loc="best")
            ax[0].set_xscale('log')
            ax[0].set_ylim(ymin=0)
            ax[0].set_ylim(ymax=1.1)
            ax[0].set_xlabel("Learning Rate", fontsize=17)
            ax[0].set_ylabel("Support Set Accuracy", fontsize=17)                
            ax[0].text(lr_best,10,'%d',rotation=90)
                
            a = base_dir + model_name  + '_' + str(i) + '_qa.npy'
            df = np.load(a, allow_pickle=True) #.item()

            ax[1].plot(lr_new, df, label=  str(i)) 
            ax[1].legend(loc="best")
            ax[1].set_xscale('log')
            ax[1].set_ylim(ymin=0)
            ax[1].set_ylim(ymax=1.1)
            ax[1].set_xlabel("Learning Rate", fontsize=17)
            ax[1].set_ylabel("Query Set Accuracy", fontsize=17)
            
        plt.savefig(base_dir + model_name +  '_Accuracy.png')
        plt.close()


    
"""### Embeddings

#### EDA Embedding
"""

def eda_embed(input_shape):
    seq = Sequential()
    seq.add(InputLayer((input_shape)))

    # 1st GRU Block
    seq.add(GRU(units = 64, return_sequences = True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", bias_initializer="zeros"))
    seq.add(Activation('tanh'))    
    seq.add(BatchNormalization())

    seq.add(GRU(units = 64, return_sequences = True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", bias_initializer="zeros"))
    seq.add(Activation('tanh'))    
    seq.add(BatchNormalization())

    # FC Block
    seq.add(Flatten())
    seq.add(Dense(64, activation=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer='l2', bias_regularizer='l2'))
    seq.add(BatchNormalization())

    return seq

"""#### BVP Embedding"""

def bvp_embed(input_shape):
    seq = Sequential()
    seq.add(InputLayer((input_shape)))

    # 1st GRU Block
    seq.add(GRU(units = 64, return_sequences = True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", bias_initializer="zeros"))
    seq.add(Activation('tanh'))    
    seq.add(BatchNormalization())

    seq.add(GRU(units = 64, return_sequences = True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", bias_initializer="zeros"))
    seq.add(Activation('tanh'))    
    seq.add(BatchNormalization())

    # FC Block
    seq.add(Flatten())    
    seq.add(Dense(64, activation=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer='l2', bias_regularizer='l2'))
    seq.add(BatchNormalization())

    return seq

"""#### Audio Embeddings - 1 eGeMAPsv02"""

def ege_embed(input_shape):
    seq = Sequential()
    seq.add(InputLayer((input_shape)))

    # 1st FC Block
    seq.add(Dense(32, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer='l2', bias_regularizer='l2'))
    seq.add(Activation('relu'))
    seq.add(BatchNormalization())

    seq.add(Dense(32, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer='l2', bias_regularizer='l2'))
    seq.add(Activation('relu'))    
    seq.add(BatchNormalization())

    # FC Block
    seq.add(Flatten())
    seq.add(Dense(64, activation=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer='l2', bias_regularizer='l2'))
    seq.add(BatchNormalization())

    return seq

"""#### Audio Embedding - 2 Mel-Spectrogram"""

def aud_embed(input_shape):
    """Generates Audio Embedding using Mel-Spectrograms
    """

    # Convolutional Neural Network
    seq = Sequential()
    seq.add(InputLayer((input_shape)))

    #1st Convolutional Layer
    seq.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='same', kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer='l2', bias_regularizer='l2'))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))
    seq.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #2nd Convolutional Layer
    seq.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same', kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer='l2', bias_regularizer='l2'))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))
    seq.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
    
    #3rd Convolutional Layer
    seq.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer='l2', bias_regularizer='l2'))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))
    seq.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
    
    #Passing it to a Fully Connected layer
    seq.add(Flatten())
    # 1st Fully Connected Layer
    seq.add(Dense(4096, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer='l2', bias_regularizer='l2'))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))
    seq.add(Dropout(0.35)) 

    seq.add(Flatten())
    seq.add(Dense(500, activation=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer='l2', bias_regularizer='l2'))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))  
    seq.add(Dropout(0.05)) 
    
    seq.add(Dense(64, activation=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer='l2', bias_regularizer='l2'))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))
    
    return seq

"""### Siamese Network"""

def siamese_network(input_dim_eda,input_dim_bvp, input_dim_aud, input_dim_ege):
    
    ## EDA Embeddings
    eda_a = Input(shape=input_dim_eda)
    eda_b = Input(shape=input_dim_eda)
    
    eda_embedding = eda_embed(input_dim_eda)
    feat_eda_a = eda_embedding(eda_a)
    feat_eda_b = eda_embedding(eda_b)

    ## BVP Embeddings
    bvp_a = Input(shape=input_dim_bvp)
    bvp_b = Input(shape=input_dim_bvp)   
    
    bvp_embedding = bvp_embed(input_dim_bvp)
    feat_bvp_a = bvp_embedding(bvp_a)
    feat_bvp_b = bvp_embedding(bvp_b)

    ## Audio Embeddings - MelSpectrograms
    aud_a = Input(shape=input_dim_aud)
    aud_b = Input(shape=input_dim_aud)

    aud_embedding = aud_embed(input_dim_aud)
    feat_aud_a = aud_embedding(aud_a)
    feat_aud_b = aud_embedding(aud_b)
    
    ## Audio Embeddings - eGeMaPS Features
    ege_a = Input(shape=input_dim_ege)
    ege_b = Input(shape=input_dim_ege)

    ege_embedding = ege_embed(input_dim_ege)
    feat_ege_a = ege_embedding(ege_a)
    feat_ege_b = ege_embedding(ege_b)
    
    # ## Concatenation of Embeddings
    feat_vecs_a = Concatenate()([feat_eda_a, feat_bvp_a, feat_ege_a, feat_aud_a]) 
    feat_vecs_b = Concatenate()([feat_eda_b, feat_bvp_b, feat_ege_b, feat_aud_b]) 

    ## Distance Measure
    distance = Lambda(euclidean_distance)([feat_vecs_a ,feat_vecs_b]) 
    distance = BatchNormalization()(distance)
    distance = Dense(1,activation=None)(distance)    
    prediction = Activation('sigmoid')(distance)

    model = Model(inputs=[eda_a, bvp_a, aud_a, ege_a,  eda_b, bvp_b, aud_b, ege_b], outputs=prediction) 


    return model


"""# A. SiameseNet with Participants """

random.seed(10)

import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np



from matplotlib import pyplot as plt
import math
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np




class LRFinder:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.acc = []
        self.lrs = []
        self.best_loss = 1e9
        self.val_losses = []
        self.val_acc = []
        
        
    def on_epoch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)


        # Log the Loss and Accuracy
        loss = logs['loss']
        acc = logs['binary_accuracy']
        val_loss = logs['val_loss']
        val_acc = logs['val_binary_accuracy']
        
        self.losses.append(loss)
        self.acc.append(acc)
        self.val_losses.append(val_loss)
        self.val_acc.append(val_acc)

        # Check whether the loss got too large or NaN
        if batch > 5 and (math.isnan(val_loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if val_loss < self.best_loss:
            self.best_loss = val_loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, x_train, y_train, x_val, y_val, start_lr, end_lr, batch_size, epochs, **kw_fit):
        # If x_train contains data for multiple inputs, use length of the first input.
        # Assumption: the first element in the list is single input; NOT a list of inputs.
        N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]
        
        # Compute number of batches and LR multiplier
        num_batches = (epochs * N) / batch_size

        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(epochs))
        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_epoch_end=lambda batch, logs: self.on_epoch_end(batch, logs))

        history = self.model.fit(x_train, y_train, validation_data = (x_val, y_val), batch_size=i, epochs=epochs,
                       callbacks=[callback],
                       **kw_fit)

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

        derivatives = self.get_derivatives(sma = 5)
        lr_b = self.get_best_lr(sma = 5)
        lr_index = self.lrs.index(lr_b)
        np.save(base_dir + model_name  + '_' + str(i) + '_lrb.npy', np.array(lr_b))
        np.save(base_dir + model_name  + '_' + str(i) + '_lrv.npy', np.array(self.val_losses[lr_index]))
        np.save(base_dir + model_name  + '_' + str(i) + '_sl.npy', np.array(self.losses))
        np.save(base_dir + model_name  + '_' + str(i) + '_ql.npy', np.array(self.val_losses))
        np.save(base_dir + model_name  + '_' + str(i) + '_sa.npy', np.array(self.acc))
        np.save(base_dir + model_name  + '_' + str(i) + '_qa.npy', np.array(self.val_acc))        
        np.save(base_dir + model_name  + '_' + str(i) + '_lr.npy', np.array(self.lrs))
        np.save(base_dir + model_name  + '_' + str(i) + '_dr.npy', np.array(derivatives))        
        
    def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale(x_scale)
        plt.show()

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(lrs, derivatives)
        plt.xscale('log')
        plt.ylim(y_lim)
        plt.show()

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.val_losses[i] - self.val_losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]



feat = 'recola/features/'
selected_pid = os.listdir(feat)
selected_pid = ['dev_1', 'dev_2', 'dev_3', 'dev_4']#[x for x in selected_pid if x != "train_8" ]
selected_pid.sort()


base_model_p = 'recola/models_a/'
base_feat = 'recola/features/'

f1s = 'recola/features/' 
t_folder = 'r_hp_4/'
feat = [''] 

shots = 50
epochs = 100
spec_res = 256
margin = 1.00

start_lr = 1e-5
end_lr = 0.5


for pid in selected_pid:

        features = f1s + pid
        ## Features Load
        specm = pd.read_hdf(features+'/mat_melfreq.hdf5')
        eda   = pd.read_hdf(features+'/eda.hdf5')
        bvp   = pd.read_hdf(features+'/bvp.hdf5')
        cp    = pd.read_hdf(features+'/compar.hdf5')
        eg    = pd.read_hdf(features+'/egemap.hdf5')

        ## Annotation Labels Load
        labels = pd.read_hdf(features+'/labels.hdf5')
       
        batch_size_variants = [128, 256, 512]
        
        """### Arousal Trial"""

        #### Arousal Model
        base_dir = 'recola/' + t_folder  + pid
        os.makedirs(base_dir, exist_ok = True)
        model_name = 'EA'

        ## Create Dataset
        aud_tr, aud_ts, eda_tr, eda_ts, bvp_tr, bvp_ts, ege_tr, ege_ts, y_tr, y_ts= create_dataset(specm, eda, bvp, eg, labels, shots, pid, base_dir, 'arousal' )

        ## Make Pairs
        ra, re, rb, ree, tr = make_pairs(aud_tr, eda_tr, bvp_tr, ege_tr, y_tr)
        a1tr, a2tr, e1tr, e2tr, b1tr, b2tr, eg1tr, eg2tr, ltr = reshape_pairs(ra, re, rb, ree, tr)

        sa, se, sb, see, ts = make_pairs(aud_ts, eda_ts, bvp_ts, ege_ts, y_ts)
        a1ts, a2ts, e1ts, e2ts, b1ts, b2ts, eg1ts, eg2ts, lts = reshape_pairs(sa, se, sb, see, ts)

        input_dim_eda = e1tr[0].shape
        input_dim_bvp = b1tr[0].shape
        input_dim_aud = a1tr[0].shape
        input_dim_ege = eg1tr[0].shape


        for i in batch_size_variants:
                model = siamese_network(input_dim_eda,input_dim_bvp, input_dim_aud, input_dim_ege)
                optimizer = optimizers.Adam()
                model.compile(loss=loss(margin=margin), optimizer= optimizer, metrics=["binary_accuracy"])
                
                # Find Most Optimal Learning Rate                
                lr_finder = LRFinder(model)                
                lr_finder.find(x_train = [e1tr, b1tr, a1tr, eg1tr, e2tr, b2tr, a2tr, eg2tr], y_train = ltr, x_val = [e1ts, b1ts, a1ts, eg1ts, e2ts, b2ts, a2ts, eg2ts], y_val =  lts, start_lr = start_lr, end_lr = end_lr, batch_size = i, epochs=epochs)                

                del model
                tf.keras.backend.clear_session()                


        # Plotting Arousal Results for a set
        plots3_s(base_dir, model_name, batch_size_variants, pid)
        
        
        """### Valence Trial"""

        #### Valence Model
        base_dir = 'recola/' + t_folder  + pid
        os.makedirs(base_dir, exist_ok=True)
        model_name = 'EV'

        ## Create Dataset
        aud_tr, aud_ts, eda_tr, eda_ts, bvp_tr, bvp_ts, ege_tr, ege_ts, y_tr, y_ts= create_dataset(specm, eda, bvp, eg, labels, shots, pid, base_dir, 'valence' )

        ## Make Pairs
        ra, re, rb, ree, tr = make_pairs(aud_tr, eda_tr, bvp_tr, ege_tr, y_tr)
        a1tr, a2tr, e1tr, e2tr, b1tr, b2tr, eg1tr, eg2tr, ltr = reshape_pairs(ra, re, rb, ree, tr)

        sa, se, sb, see, ts = make_pairs(aud_ts, eda_ts, bvp_ts, ege_ts, y_ts)
        a1ts, a2ts, e1ts, e2ts, b1ts, b2ts, eg1ts, eg2ts, lts = reshape_pairs(sa, se, sb, see, ts)

        input_dim_eda = e1tr[0].shape
        input_dim_bvp = b1tr[0].shape
        input_dim_aud = a1tr[0].shape
        input_dim_ege = eg1tr[0].shape

        for i in batch_size_variants:
                model = siamese_network(input_dim_eda,input_dim_bvp, input_dim_aud, input_dim_ege)
                optimizer = optimizers.Adam()
                model.compile(loss=loss(margin=margin), optimizer= optimizer, metrics=["binary_accuracy"])
                
                
                ## Find Most Optimal Learning Rate
                lr_finder = LRFinder(model)
                lr_finder.find(x_train = [e1tr, b1tr, a1tr, eg1tr, e2tr, b2tr, a2tr, eg2tr], y_train = ltr, x_val = [e1ts, b1ts, a1ts, eg1ts, e2ts, b2ts, a2ts, eg2ts], y_val =  lts, start_lr = start_lr, end_lr = end_lr, batch_size=i, epochs=epochs)                
             
                
                del model
                tf.keras.backend.clear_session()

        # Plotting Valence Results for a set
        plots3_s(base_dir, model_name, batch_size_variants, pid)