#import load_intan_rhd_format
import src.importrhdutilities as load_intan_rhd_format
import sys, os

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt
from scipy.signal import argrelextrema

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

import pdb
# Parameters
num_channels = 32
sample_freq = 20e3

    #spike time
prespike = 20 #80
postspike =  40 #140
buffer = 1 # so that spikes do not overlap
spike_time = prespike + postspike + buffer

artifact_threshold = 400

    #filtering
passband = [250, 6e3]
filter_order = 1
filter_iter = 11
ftype = 'bandpass'

# Get data:

def retrieve_recording_files(data_location, mouse_id, date):
    mypath = data_location + date + '_' + mouse_id + '/'
    files = []
    for file in os.scandir(mypath):
        if file.name.lower().endswith((".rhd")):
            files.append(file.name)
    files = set(files)
    
    return files

# Functions per channel:

# Filtering

def filter_data_per_ch(data, passband=[250,6e3], sample_freq=20e3):
    """
    Performs digital filtering of an unfiltered timeseries ephys trace.
    
    Implements the sosfiltfilt package for zero-timeshifted filtering of the signal, with
    filter_iter being a hard-coded parameter.
    
    Based on and adapted from Guosong's Matlab code (2015).
    
    Parameters:
    data (numpy array): An n_samples by n_features array of data to be filtered.
    passband (tuple): Lowerbound (element 0) and upperbound (element 1) of the filtered frequencies.
    sample_freq (int or float): The sample frequency of the unfiltered trace, in Hz or samples/sec.

    Returns:
    filtered_data (numpy array): An array of the digitally bandpass-filtered signal.
    """
    #note: can still play around with the parameters of the butter() fxn
    # as well as the sosfiltfilt fxn

    Wn = 2*np.array(passband)/sample_freq
    sos = butter(filter_order, Wn, ftype, output='sos')

    filtered_data = np.copy(data)
    for i in range(filter_iter):
        filtered_data = sosfiltfilt(sos, filtered_data, axis=0)
    return filtered_data


# Thresholding
def find_local_max(filtered_data):#indices, filtered_data):
    """
    Finds the local maximum of filtered_data within spike_time range, centered around indices.
        Note: spike_time is defined, here, by prespike+buffer -> postpike+buffer
    
    Parameters:
    filtered_data (numpy array): An n samples by n features array of bandpass filtered ephys trace.

    Returns:
    new_indices (numpy array): A numpy array with new (shifted) indices of local maxima around threshold crossing.
    """
    # dat = filtered_data
    threshold = np.nanmedian(np.abs(filtered_data)) / 0.6745 * 4
    indices = np.where(np.abs(filtered_data[int(prespike + buffer) : int(-1*(postspike + buffer + 1))]) \
                       > threshold)[0] + prespike + buffer
        
    new_indices = np.zeros((len(indices), 2))

    dat = np.zeros((len(indices), prespike+postspike+2*buffer))
    dat_abs = np.zeros((len(indices), prespike+postspike+2*buffer))

    for i in range(len(indices)):
        x = filtered_data[indices[i] - prespike - buffer: indices[i] + postspike + buffer]
        dat[i,:] = x
        dat_abs[i, :] = np.abs(x)
        y = argrelextrema(np.abs(x), np.greater, axis=0, order=prespike+postspike)[0]
        
        new_indices[i,0] = indices[i] # new_indices[i, 0] are the original indices

        if len(y) > 0:
            new_indices[i,1] = y
            
        else:
            other = np.where(dat_abs[i,:]==np.max(dat_abs[i,:]))[0]
            new_indices[i,1] = other
            
    # Accounting for overlapping indices within the spike_time window
    
    local_max_idx = [int(i + j - (prespike+buffer)) for i,j in zip(new_indices[:,0], new_indices[:,1])]
    local_max_idx = sorted(list(set(local_max_idx)))
    # Accounting for beginning and ending
    local_max_idx = [i for i in local_max_idx if i>(prespike+buffer) and (i+postspike+buffer)<len(filtered_data)]


    filtered_indices = []
    
    indices_diff = np.diff(local_max_idx)>postspike

    for i in range(len(indices_diff)):
        idx = local_max_idx[i]
        if indices_diff[i]==True:
            filtered_indices.append(idx)
        else:
            datum = filtered_data[idx - prespike - buffer: idx + postspike + buffer]
            if datum.shape[0] != 62:
                pdb.set_trace() 
            new_idx = np.where(datum == datum.max())[0][0]
            new_idx += idx - (prespike + buffer)
            filtered_indices.append(new_idx)

    if indices_diff[-1]==True:
        filtered_indices.append(local_max_idx[-1])
        filtered_indices = sorted(set(filtered_indices))
        
    return filtered_indices
    
# now it is the following:
# new_indices = find_local_max(filtered_data_ch, artifact_threshold)

# Spike collection

def return_spike_idx(local_max_idx, filtered_data):
    """
    Performs threshold-based and temporal-based spike collection. 
    Based on and adapted from Guosong's Matlab code (2015).
    

    Parameters:
    local_max_idx (list): A list with unique indices of local maxima.
    filtered_data (numpy array): An n samples by n features array of bandpass filtered ephys trace.

    Returns:
    idx_keep (list): A list with indices, to be used on filtered_indices, that pass the spike 
        collection conditions.
    idx_del (list): A list with indices that did not pass the spike collection conditions. (Record keeping)
    """
    
    idx_keep = []
    idx_del = []

    for l in range(len(local_max_idx)):
        idx = local_max_idx[l]

        dat_spike = filtered_data[idx - prespike - buffer : idx + postspike + buffer]

        if dat_spike.shape[0] == spike_time + 1:
            
            max_clearing = max(dat_spike[0], dat_spike[-1])
            main_peak = filtered_data[idx]

            max_pre = np.max(np.abs(filtered_data[idx - prespike: idx - int(prespike*0.4)]))
            max_post = np.max(np.abs(filtered_data[idx + int(postspike*0.5): idx + postspike]))

            if max_clearing < main_peak/2 and main_peak < artifact_threshold and max_post < main_peak*0.4 \
            and max_pre < main_peak*0.8:
                idx_keep.append(l)

            elif max_clearing < abs(main_peak)/2 and abs(main_peak)<artifact_threshold and \
                max_post < abs(main_peak)*0.4 and max_pre < abs(main_peak)*0.8:
                idx_keep.append(l)

            else:
                idx_del.append(l)
                
    return idx_keep, idx_del  

#idx_keep, idx_del = return_spike_idx(filtered_indices, data)

def collect_spike_array(idx_keep, filtered_indices, filtered_data):
    """
    Performs threshold-based and temporal-based spike collection. 
    Based on and adapted from Guosong's Matlab code (2015).
    

    Parameters:
    idx_keep (list): A list with indices, to be used on filtered_indices, that pass the spike 
        collection conditions.
    idx_del (list): A list with indices that did not pass the spike collection conditions. (Record keeping)
    filtered_indices (list): A list with unique indices of local maxima.
    data (numpy array): An n samples by n features array of bandpass filtered ephys trace.
    
    Returns:
    spike_time (numpy array): An n-shaped array, where n: timestamp (in #samples) of spike amplitude
    spike_array (numpy array): An n x m array, where n: spike ID, m: samples of spike waveform
    """
    
    spike_array = np.zeros((len(idx_keep), spike_time))
    spike_times = np.array(filtered_indices)[idx_keep]

    for i in range(len(idx_keep)):
        idx = filtered_indices[idx_keep[i]]
        datum = filtered_data[idx - prespike : idx + postspike + 1]
        spike_array[i,:] = datum.ravel()
        
    return spike_times, spike_array

# Clustering

def dimred_data(X, n_components=3):
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    return X_pca, explained_variance_ratio

def dimred_cluster(X, n_clusters = None):
    """
    Performs k-means clustering on PCA-transformed data with a user-defined range of number of clusters.
    
    Implements the Silhouette method to optimize for the number of clusters, based on maximizing the
    average silhouette score.

    Parameters:
    X (numpy array): An n_samples by n_features array of data to be clustered.
    n_clusters (int, optional): The number of clusters to use for k-means clustering. If not specified, the optimal number
                                 of clusters will be determined using the elbow method.

    Returns:
    labels (numpy array): An array of predicted labels for each sample in X.
    """
    
    # PCA
    X_pca, _ = dimred_data(X, n_components=3)
    
    # Clustering
    if n_clusters is None:
        # Set range of number of clusters to try
        range_n_clusters = [2, 3, 4, 5]
        # hard-coded based on a priori assumption of data structure

        silhouette_avgs = []

        for n_clusters in range_n_clusters:

            # Run k-means clustering and compute silhouette scores
            km = KMeans(n_clusters=n_clusters, n_init=1)#'auto')
            # import pdb

            # pdb.set_trace()
            cluster_labels = km.fit_predict(X_pca)
        
            silhouette_avg = silhouette_score(X_pca, cluster_labels)
            #sample_silhouette_values = silhouette_samples(X_pca, cluster_labels)
            silhouette_avgs.append(silhouette_avg)

#         print('Silhouette averages per num_clusters '+str(silhouette_avgs))

        n_clusters = np.argmax(silhouette_avgs) + 2 # to offset for n_clusters_0 = 2
#         print('The ideal number of clusters from silhouette analysis is '+str(n_clusters))

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
        cluster_labels = km.fit_predict(X)
         
        return silhouette_avgs, n_clusters, cluster_labels
    
def neuron_per_cluster(X, n_clusters, cluster_labels):
    """
    Function generating spike waveforms per cluster.

    Parameters:
    X (numpy array): An n_samples by n_features array of data to be clustered.
    n_clusters (int): The optimal number of clusters to from the dimred_cluster() function.
    cluster_labels (numpy array): An n_samples array with elements corresponding to cluster_id.
    
    Returns:
    spikecluslabel (list): A list of numpy arrays, where each element of the list corresponds to
        cluster ID, and the nested numpy arrays are the waveforms of spikes corresponding to that
        cluster.
    """
    
    spikecluslabel = []
    
    for clus in range(n_clusters):
        mask = cluster_labels==clus

        dat = X[mask]
        
        spikecluslabel.append(dat)
        
    # output will be a list of numpy arrays
    # list indices refer to the cluster ID
    # array nested within the list corresponds 
    # to the spike waveform per cluster
    return spikecluslabel


### OLD VERSION BELOW:
# # will only use below 3 in script
# import load_intan_rhd_format
# import sys
# import os


# import numpy as np
# import matplotlib.pyplot as plt

# import scipy.io as sio
# from scipy.signal import butter, filtfilt, sosfiltfilt, ellip, cheby2
# from scipy.signal import argrelextrema


# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_samples, silhouette_score
# from sklearn.decomposition import PCA



# def filter_data_per_ch(data, passband, sample_freq):
#     """
#     Performs digital filtering of an unfiltered timeseries ephys trace.
    
#     Implements the sosfiltfilt package for zero-timeshifted filtering of the signal, with
#     filter_iter being a hard-coded parameter.
    
#     Based on and adapted from Guosong's Matlab code (2015).
    
#     Parameters:
#     data (numpy array): An n_samples by n_features array of data to be filtered.
#     passband (tuple): Lowerbound (element 0) and upperbound (element 1) of the filtered frequencies.
#     sample_freq (int or float): The sample frequency of the unfiltered trace, in Hz or samples/sec.

#     Returns:
#     filtered_data (numpy array): An array of the digitally bandpass-filtered signal.
#     """
#     #note: can still play around with the parameters of the butter() fxn
#     # as well as the sosfiltfilt fxn

#     Wn = 2*np.array(passband)/sample_freq
#     sos = butter(filter_order, Wn, ftype, output='sos') # note: add filter_order and filter_iter as params

#     filtered_data = np.copy(data)
#     for i in range(filter_iter):
#         filtered_data = sosfiltfilt(sos, filtered_data, axis=0)
#     return filtered_data


# def find_local_max (indices, filtered_data):
#     """
#     Finds the local maximum of filtered_data within spike_time range, centered around indices.
    
#     Parameters:
#     indices (list): A list of indices where |filtered_data| > threshold. (hard-coded arbitrary value).
#     filtered_data (numpy array): An n samples by n features array of bandpass filtered ephys trace.

#     Returns:
#     dat (numpy array): An n x m array, where n: index of local max and m: spike values (for record keeping).
#     dat_abs (numpy array): the absolute value of dat (for record keeping).
#     new_indices (numpy array): A numpy array, where [i, 0] are original indices 
#         and [i, 1] are new (shifted) indices.
#     """
#     # note: can clean up a bit in the future and only focus on new_indices[i, 1]
    
#     new_indices = np.zeros((len(indices), 2))

#     dat = np.zeros((len(indices), prespike+postspike+2*buffer))
#     dat_abs = np.zeros((len(indices), prespike+postspike+2*buffer))


#     for i in range(len(indices)):
#         x = filtered_data[indices[i] - prespike - buffer: indices[i] + postspike + buffer]
#         dat[i,:] = x
#         dat_abs[i, :] = np.abs(x)
#         y = argrelextrema(np.abs(x), np.greater, axis=0, order=prespike+postspike)[0]
        
#         new_indices[i,0] = indices[i] # new_indices[i, 0] are the original indices

#         if len(y) > 0:
#             new_indices[i,1] = y
            
#         else:
#             other = np.where(dat_abs[i,:]==np.max(dat_abs[i,:]))[0]
#             new_indices[i,1] = other
#     return dat, dat_abs, new_indices

# #_, _, new_indices = find_local_max(indices, filtered_data-ch)

# def new_indices_true(new_indices, filtered_data):
#     """
#     Filters the indices to account for overlapping indices within the spike_time window.
    

#     Parameters:
#     new_indices (numpy array): A numpy array, where [i, 0] are original indices 
#         and [i, 1] are new (shifted) indices.
#     filtered_data (numpy array): An n samples by n features array of bandpass filtered ephys trace.

#     Returns:
#     filtered_indices (list): A list with unique indices of local maxima.

#     """
    
#     local_max_idx = [int(i + j - (prespike+buffer)) for i,j in zip(new_indices[:,0], new_indices[:,1])]
#     local_max_idx = sorted(list(set(local_max_idx)))

#     filtered_indices = []
    
#     indices_diff = np.diff(local_max_idx)>postspike

#     for i in range(len(indices_diff)):
#         idx = local_max_idx[i]
#         if indices_diff[i]==True:
#             filtered_indices.append(idx)
#         else:
#             datum = filtered_data[idx - prespike - buffer: idx + postspike + buffer]
#             new_idx = np.where(datum == datum.max())[0][0]
#             new_idx += idx - (prespike + buffer)
#             filtered_indices.append(new_idx)

#     if indices_diff[-1]==True:
#         filtered_indices.append(local_max_idx[-1])
#         filtered_indices = sorted(set(filtered_indices))
        
#     return filtered_indices

# def return_spike_idx(local_max_idx, data):
#     """
#     Performs threshold-based and temporal-based spike collection. 
#     Based on and adapted from Guosong's Matlab code (2015).
    

#     Parameters:
#     local_max_idx (list): A list with unique indices of local maxima.
#     data (numpy array): An n samples by n features array of bandpass filtered ephys trace.

#     Returns:
#     idx_keep (list): A list with indices, to be used on filtered_indices, that pass the spike 
#         collection conditions.
#     idx_del (list): A list with indices that did not pass the spike collection conditions. (Record keeping)
#     """
    
#     idx_keep = []
#     idx_del = []

#     for l in range(len(local_max_idx)):
#         idx = local_max_idx[l]

#         dat_spike = data[idx - prespike - buffer : idx + postspike + buffer]

#         if dat_spike.shape[0] == spike_time + 1:
            
#             max_clearing = max(dat_spike[0], dat_spike[-1])
#             main_peak = data[idx]

#             max_pre = np.max(np.abs(data[idx - prespike: idx - int(prespike*0.4)]))
#             max_post = np.max(np.abs(data[idx + int(postspike*0.5): idx + postspike]))

#             if max_clearing < main_peak/2 and main_peak < artifact_threshold and max_post < main_peak*0.4 \
#             and max_pre < main_peak*0.8:
#                 idx_keep.append(l)

#             elif max_clearing < abs(main_peak)/2 and abs(main_peak)<artifact_threshold and \
#                 max_post < abs(main_peak)*0.4 and max_pre < abs(main_peak)*0.8:
#                 idx_keep.append(l)

#             else:
#                 idx_del.append(l)
                
#     return idx_keep, idx_del  

# #idx_keep, idx_del = return_spike_idx(filtered_indices, data)

# def collect_spike_array(idx_keep, idx_del, filtered_indices, data):
#     """
#     Performs threshold-based and temporal-based spike collection. 
#     Based on and adapted from Guosong's Matlab code (2015).
    

#     Parameters:
#     idx_keep (list): A list with indices, to be used on filtered_indices, that pass the spike 
#         collection conditions.
#     idx_del (list): A list with indices that did not pass the spike collection conditions. (Record keeping)
#     filtered_indices (list): A list with unique indices of local maxima.
#     data (numpy array): An n samples by n features array of bandpass filtered ephys trace.
    
#     Returns:
#     spike_time (numpy array): An n-shaped array, where n: timestamp (in #samples) of spike amplitude
#     spike_array (numpy array): An n x m array, where n: spike ID, m: samples of spike waveform
#     """
    
#     spike_array = np.zeros((len(idx_keep), spike_time))
#     spike_times = np.array(filtered_indices)[idx_keep]

#     for i in range(len(idx_keep)):
#         idx = filtered_indices[idx_keep[i]]
#         datum = data[idx - prespike : idx + postspike + 1]
#         spike_array[i,:] = datum.ravel()
        
#     return spike_times, spike_array

# def dimred_cluster(X, n_clusters = None):
#     """
#     Performs k-means clustering on PCA-transformed data with a user-defined range of number of clusters.
    
#     Implements the Silhouette method to optimize for the number of clusters, based on maximizing the
#     average silhouette score.

#     Parameters:
#     X (numpy array): An n_samples by n_features array of data to be clustered.
#     n_clusters (int, optional): The number of clusters to use for k-means clustering. If not specified, the optimal number
#                                  of clusters will be determined using the elbow method.

#     Returns:
#     labels (numpy array): An array of predicted labels for each sample in X.
#     """
    
#     # Performing PCA
#     pca = PCA(n_components=3)
#     X_pca = pca.fit_transform(X)
    
#     if n_clusters is None:
#         # Set range of number of clusters to try
#         range_n_clusters = [2, 3, 4, 5]

#         # it looks like it very slightly prefers 6 over 3
#         # so in future versions, can pick out n top silhouette avg scores 
#         # and select the smallest cluster size out of them
#         # OR: if sample_silhouette_values are < 0 -> exclude from optimization schema

#         silhouette_avgs = []

#         for n_clusters in range_n_clusters:

#             # Run k-means clustering and compute silhouette scores
#             km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
#             cluster_labels = km.fit_predict(X_pca)
#             silhouette_avg = silhouette_score(X_pca, cluster_labels)
#             sample_silhouette_values = silhouette_samples(X_pca, cluster_labels)
#             silhouette_avgs.append(silhouette_avg)

#         print('Silhouette averages per num_clusters '+str(silhouette_avgs))

#         n_clusters = np.argmax(silhouette_avgs) + 2 # to offset for n_clusters_0 = 2
#         print('The ideal number of clusters from silhouette analysis is '+str(n_clusters))

#         km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
#         cluster_labels = km.fit_predict(X)
         
#         return silhouette_avgs, n_clusters, cluster_labels

# def plot_clusters(X, n_clusters, cluster_labels):
#     """
#     Plotting function for spike waveforms per cluster.
    

#     Parameters:
#     X (numpy array): An n_samples by n_features array of data to be clustered.
#     n_clusters (int): The optimal number of clusters to from the dimred_cluster() function.
#     cluster_labels (numpy array): An n_samples array with elements corresponding to cluster_id.
#     """
    
#     n_colors = n_clusters

#     cmap = plt.get_cmap('viridis')

#     colors = cmap(np.linspace(0, 1, n_colors))

#     for clus in range(n_clusters):
#         mask = cluster_labels==clus

#         dat_temp = X[mask]

#         plt.plot(dat_temp[0,:], c=colors[clus], label='Cluster '+str(clus+1))
#         for j in range(dat_temp.shape[0]):
#             plt.plot(dat_temp[j,:], c=colors[clus])

#     plt.ylabel('Voltage ($\mu$V)')
#     plt.legend()
#     plt.show()
    
# def neuron_per_cluster(X, n_clusters, cluster_labels):
#     """
#     Plotting function for spike waveforms per cluster.

#     Parameters:
#     X (numpy array): An n_samples by n_features array of data to be clustered.
#     n_clusters (int): The optimal number of clusters to from the dimred_cluster() function.
#     cluster_labels (numpy array): An n_samples array with elements corresponding to cluster_id.
    
#     Returns:
#     spikecluslabel (list): A list of numpy arrays, where each element of the list corresponds to
#         cluster ID, and the nested numpy arrays are the waveforms of spikes corresponding to that
#         cluster.
#     """
    
#     spikecluslabel = []
    
#     for clus in range(n_clusters):
#         mask = cluster_labels==clus

#         dat = X[mask]
        
#         spikecluslabel.append(dat)
        
#     # output will be a list of numpy arrays
#     # list indices refer to the cluster ID
#     # array nested within the list corresponds 
#     # to the spike waveform per cluster
#     return spikecluslabel