import argparse
import os
from datetime import datetime
import time
import pdb

import utils
import numpy as np
import pickle

from resource import getrusage, RUSAGE_SELF

from pathlib import Path

## IDEAS:
# remove the num_channels dependency and just handle at the end
def local_retrieve(mypath):
    '''
    If data are hosted on your local hard-drive, 
    just provide the path 'mypath' to the directory
    that hosts the Intan recording session.
    '''
    files = []
    for file in os.scandir(mypath):
        if file.name.lower().endswith((".rhd")):
            files.append(file.name)
    files = sorted(files) # ordering files
    
    return files

# Helper functions

def retrieve_recording_files(path):#data_location, mouse_id, date):
    '''
    Function retrieving intan recording files from directory,
    optimized for Grace Woods' longitudinal recording sets.
    
    Parameters:
    path (string): path indicating direcotry containing recording files,
        defined by params in bash script.

    Returns:
    files (list): a list of filenames of the recordings (ordered).
    '''
    mypath =  path
    files = []
    for file in os.scandir(mypath):
        if file.name.lower().endswith((".rhd")):
            files.append(file.name)
    files = sorted(files)
    
    return files

# Putting it all together:

def filtunfilt_alldata(path, num_channels=32, maxfiles=-1, batch_process=False):
    """
    Function acquiring Intan recording files either in a batched or one-shot approach 
    (refer to batch_process), utilizing filter functions defined in utils.py

    Parameters:
    path (string): A string indicating the path of the intan recording files ('.rhd' files).
    num_channels (int, optional): Number of channels to analyze. Note: currently does not accept
        a specified list of channels. If interested and in need of help, contact Grace.
    batch_process (bool, optional): Parameter controlling whether to batch process spike sort or
        one-shot process for the entirety of its recording session.
    
    Returns:
    unfiltered_data (numpy array OR list): object with unfiltered data; will be numpy array if 
        one-shot analysis ("batch_process=False") or list if batch-processed ("batch_process=True")
        where each element corresponds to a one-minute recording segment (ordered).
    filtered_data (numpy array OR list):  object with bandpass filtered data; will be numpy array if 
        one-shot analysis ("batch_process=False") or list if batch-processed ("batch_process=True")
        where each element corresponds to a one-minute recording segment (ordered).
    """
    files = retrieve_recording_files(path+'/') # NOTe: had to add '/' from the bash file

    if batch_process == False:
        unfiltered_data = np.zeros((1, num_channels))

        # get unfiltered data
        for file in files[:maxfiles]:
            intan_dat, _ = utils.load_intan_rhd_format.load_file(path+file)
            dat = intan_dat['amplifier_data'].T
            unfiltered_data = np.append(unfiltered_data, dat, axis=0)
        # now filtered data:
        filtered_data = np.zeros_like(unfiltered_data)

        for ch in range(num_channels):
            data = unfiltered_data[:, ch]
            filtered_data[:, ch] = utils.filter_data_per_ch(data, passband=[250, 6e3], sample_freq=20e3)

    elif batch_process == True:
        filtered_data = np.empty(len(files), dtype=object)

        for batch,file in enumerate(files):
            intan_dat, _ = utils.load_intan_rhd_format.load_file(path+file)
            dat = intan_dat['amplifier_data'].T
            unfiltered_data = []
            filtered_data[batch] = utils.filter_data_per_ch(dat, passband=[250, 6e3], sample_freq=20e3)

    return unfiltered_data, filtered_data


def run_spikesorting(path, maxfiles):
    num_channels = 32
    _, filtered_data = filtunfilt_alldata(path, num_channels, maxfiles, batch_process=False)
    num_channels = filtered_data.shape[1]

    noise_allchs = np.nanmedian(np.abs(filtered_data), axis=0)
    numclusters_allchs = np.zeros(num_channels)
    spiketimes_allchs, spike_allchs, spikecluster_labels, spiketimes_cluster, \
        silhouette_allchs, spikecluster_allchs, spikepca_allchs, pca_variance_allchs = \
            [np.empty(num_channels, dtype=object) for _ in range(8)]

    date_computation = datetime.now()
    time_start = time.perf_counter()

    for ch in range(num_channels):
        dat_ch = filtered_data[:, ch]

        filtered_indices = utils.find_local_max(dat_ch)
        idx_keep, _ = utils.return_spike_idx(filtered_indices, dat_ch)
        spike_times, spike_array = utils.collect_spike_array(idx_keep, filtered_indices, dat_ch)
        spike_pca, spike_varianceratio = utils.dimred_data(spike_array, n_components=3)
        silhouette_avgs, nideal_clusters, cluster_labels = utils.dimred_cluster(spike_array, n_clusters=None)
        neuron_cluster = utils.neuron_per_cluster(spike_array, nideal_clusters, cluster_labels)

        spiketimes_allchs[ch] = spike_times
        spike_allchs[ch] = spike_array
        spikecluster_labels[ch] = cluster_labels
        spiketimes_cluster[ch] = utils.neuron_per_cluster(spike_times, nideal_clusters, cluster_labels)

        spikepca_allchs[ch] = spike_pca
        pca_variance_allchs[ch] = spike_varianceratio

        silhouette_allchs[ch] = silhouette_avgs
        numclusters_allchs[ch] = int(nideal_clusters)
        spikecluster_allchs[ch] = neuron_cluster
    
    time_elapsed = (time.perf_counter() - time_start)
    print('Finished within '+str(time_elapsed)+'seconds')

    results_dict = {'Path': path, 'analysis_details': {'date of computation':date_computation, 
                                                       'time to spikesort':time_elapsed},
                                    'data': {'filtered_data':filtered_data},
                                    'noise': noise_allchs,
                                    'spikes': {'spike_time': spiketimes_allchs, 'spike_waveforms': spike_allchs, \
                                               'spikecluster_labels': spikecluster_labels,
                                                'silhouette_scores':silhouette_allchs, 'numideal_cluster': numclusters_allchs,
                                                'spikecluster_id': spikecluster_allchs},
                                    'etc_metrics':{'PCA':spike_pca, 'PCA_variance': pca_variance_allchs}}
    return results_dict

def run_spikesorting_batch(path, maxfiles):
    num_channels = 32

    files = retrieve_recording_files(path)
    num_batches = len(files)

    _, filtered_data = filtunfilt_alldata(path, num_channels, maxfiles, batch_process=True)


    noise_allchs, numclusters_allchs, spiketimes_allchs, spikecluster_labels, spiketimes_cluster, \
    spike_allchs,silhouette_allchs, spikecluster_allchs, spikepca_allchs, pca_variance_allchs \
              = [np.empty(num_batches, dtype=object) for _ in range(10)]
    
    date_computation = datetime.now()
    time_start = time.perf_counter()

    for batch in range(num_batches):
        print('Batch'+str(batch)+' Processing')
        dat_filt = filtered_data[batch]

        noise_allchs[batch] = np.nanmedian(np.abs(dat_filt), axis=0)
        numclusters_allchs[batch], spiketimes_allchs[batch], spikecluster_labels[batch], \
            spiketimes_cluster[batch], spike_allchs[batch], silhouette_allchs[batch], \
                spikecluster_allchs[batch], spikepca_allchs[batch], pca_variance_allchs[batch] \
                  = [np.empty(num_channels, dtype=object) for _ in range(9)]

        for ch in range(num_channels):
            dat_ch = dat_filt[:,ch]

            filtered_indices = utils.find_local_max(dat_ch)
            idx_keep, _ = utils.return_spike_idx(filtered_indices, dat_ch)
            spike_times, spike_array = utils.collect_spike_array(idx_keep, filtered_indices, dat_ch)
            spike_pca, spike_varianceratio = utils.dimred_data(spike_array, n_components=3)
            silhouette_avgs, nideal_clusters, cluster_labels = utils.dimred_cluster(spike_array, n_clusters=None)
            neuron_cluster = utils.neuron_per_cluster(spike_array, nideal_clusters, cluster_labels)
                        
            spiketimes_allchs[batch][ch] = spike_times
            spike_allchs[batch][ch] = spike_array
            spikecluster_labels[batch][ch] = cluster_labels
            spiketimes_cluster[batch][ch] = utils.neuron_per_cluster(spike_times, nideal_clusters, cluster_labels)
            spikepca_allchs[batch][ch] = spike_pca
            pca_variance_allchs[batch][ch] = spike_varianceratio

            silhouette_allchs[batch][ch] = silhouette_avgs
            numclusters_allchs[batch][ch] = int(nideal_clusters)
            spikecluster_allchs[batch][ch] = neuron_cluster
            

    time_elapsed = (time.perf_counter() - time_start)
    print('Finished within '+str(time_elapsed)+' seconds')
    #return noise_allchs, spiketimes_allchs, spike_allchs, silhouette_allchs, numclusters_allchs, spikecluster_allchs
    results_dict = {'Path': path, 'analysis_details': {'date of computation':date_computation, 
                                                       'time to spikesort':time_elapsed},
                                    'data': {'filtered_data_batch' : filtered_data},
                                    'noise': noise_allchs,
                                    'spikes': {'spike_time': spiketimes_allchs, 
                                               'spike_waveforms': spike_allchs,
                                               'silhouette_scores':silhouette_allchs,
                                               'numideal_cluster': numclusters_allchs,
                                               'spikecluster_labels': spikecluster_labels,
                                               'spikecluster_id': spikecluster_allchs,},
                                    'etc_metrics':{'PCA':spikepca_allchs, 'PCA_variance': pca_variance_allchs}}
    return results_dict 

def main(args):
    print('Directory: '+str(args.datadir))
    print(args.maxfiles)
    results = run_spikesorting(args.datadir, args.maxfiles)
    print('Memory used: '+str(getrusage(RUSAGE_SELF).ru_maxrss))

    datadir = Path(args.datadir)

    if args.outdir is None:
        outdir = datadir
    else:
        outdir = Path(args.outdir)
    
    animal_id, session_id = datadir.parts[-2], datadir.parts[-1]
    outfile = outdir/animal_id/(session_id+'.pickle')

    with open(outfile, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__== "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', type=str)
    parser.add_argument('--n_components', type=int, default=3) # PCA
    parser.add_argument('--maxfiles', type=int, default=-1)
    parser.add_argument('--outdir', type=str, default=None)

    args = parser.parse_args()
    main(args)

