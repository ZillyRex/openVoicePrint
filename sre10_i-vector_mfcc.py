# -*- coding: utf-8 -*-
"""
Created on Tue May 12 2015

@author: Anthony Larcher

This script run an experiment on the male NIST Speaker Recognition
Evaluation 2010.
For more details about the protocol, refer to
http://www.itl.nist.gov/iad/mig/tests/sre/2010/

The number of trials performed is
target
nontarget

"""


import sys
import numpy as np
import scipy

import os
import copy

import sidekit
import multiprocessing
import matplotlib.pyplot as mpl
import logging

logging.basicConfig(filename='log/sre10_i-vector.log',level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)


#################################################################
# Set your own parameters
#################################################################
train = True
test = True
distribNb = 512  # number of Gaussian distributions for each GMM
rank_TV = 400
audioDir = '/lium/paroleh/larcher/data/nist_tuto/mfcc/'

# Automatically set the number of parallel process to run.
# The number of threads to run is set equal to the number of cores available 
# on the machine minus one or to 1 if the machine has a single core.
#nbThread = max(multiprocessing.cpu_count()-1, 1)
nbThread = 30

#################################################################
# Load IdMap, Ndx, Key from PICKLE files and ubm_list
#################################################################
print('Load task definition')
enroll_idmap = sidekit.IdMap('task/sre10_coreX-coreX_m_trn.h5', 'hdf5')
nap_idmap = sidekit.IdMap('task/sre04050608_m_training.h5', 'hdf5')
#back_idmap = sidekit.IdMap('task/sre10_coreX-coreX_m_back.h5', 'hdf5')
test_ndx = sidekit.Ndx('task/sre10_coreX-coreX_m_ndx.h5', 'hdf5')
test_idmap = sidekit.IdMap('task/sre10_coreX-coreX_m_test.h5', 'hdf5')
keysX = []
for cond in range(9):
    keysX.append(sidekit.Key('task/sre10_coreX-coreX_det{}_key.h5'.format(cond + 1)))

with open('task/ubm_list.txt', 'r') as inputFile:
    ubmList = inputFile.read().split('\n')

ubmList = ubmList[:500]

if train:
    #%%
    #################################################################
    # Process the audio to generate MFCC
    #################################################################
    print('Create the feature server to extract MFCC features')
    fs = sidekit.FeaturesServer(input_dir=audioDir,
                 input_file_extension='.mfcc',
                 label_dir=audioDir,
                 label_file_extension='.lbl',
                 from_file='spro4',
                 config='sid_8k',
                 keep_all_features=False)


    #%%
    #################################################################
    # Train the Universal background Model (UBM)
    #################################################################
    print('Load data to Train the UBM by EM')
    ubm = sidekit.Mixture()
    llk = ubm.EM_split(fs, ubmList, distribNb, numThread=nbThread)
    ubm.save_pickle('gmm/ubm_mfcc.p')

    #%%
    #################################################################
    # Compute the sufficient statistics on the UBM
    #################################################################
    print('Compute the sufficient statistics')
    # Create a StatServer for the enrollment data and compute the statistics
    enroll_stat = sidekit.StatServer(enroll_idmap, ubm)
    enroll_stat.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(enroll_stat.segset.shape[0]), numThread=nbThread)
    enroll_stat.save('data/stat_sre10_coreX-coreX_m_enroll_mfcc.h5')

    nap_stat = sidekit.StatServer(nap_idmap, ubm)
    nap_stat.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(nap_stat.segset.shape[0]), numThread=nbThread)
    nap_stat.save('data/stat_sre04050608_m_training_mfcc.h5')


    test_stat = sidekit.StatServer(test_idmap, ubm)
    test_stat.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(test_stat.segset.shape[0]), numThread=nbThread)
    test_stat.save('data/stat_sre10_coreX-coreX_m_test_mfcc.h5')

    #%%
    #################################################################
    # Train Total Variability Matrix for i-vector extration
    #################################################################
    print('Estimate Total Variability Matrix')
    mean, TV, G, H, Sigma = nap_stat.factor_analysis(rank_TV, rank_G=0, rank_H=None, re_estimate_residual=False,
                        itNb=(10,0,0), minDiv=True, ubm=ubm, 
                        batch_size=1000, numThread=nbThread)

    sidekit.sidekit_io.write_pickle(TV, 'data/TV_sre04050608_m_mfcc.p')
    sidekit.sidekit_io.write_pickle(mean, 'data/TV_mean_sre04050608_m_mfcc.p')
    sidekit.sidekit_io.write_pickle(Sigma, 'data/TV_Sigma_sre04050608_m_mfcc.p')


    #%%
    #################################################################
    # Extract i-vectors for target models, training and test segments
    #################################################################
    print('Extraction of i-vectors') 
    enroll_iv = enroll_stat.estimate_hidden(mean, Sigma, V=TV, U=None, D=None, numThread=nbThread)[0]
    enroll_iv.save('data/iv_sre10_coreX-coreX_m_enroll_mfcc.h5')

    test_iv = test_stat.estimate_hidden(mean, Sigma, V=TV, U=None, D=None, numThread=nbThread)[0]
    test_iv.save('data/iv_sre10_coreX-coreX_m_test_mfcc.h5')

    nap_iv = nap_stat.estimate_hidden(mean, Sigma, V=TV, U=None, D=None, numThread=nbThread)[0]
    nap_iv.save('data/iv_sre04050608_m_training_mfcc.h5')


if test:

    enroll_iv = sidekit.StatServer('data/iv_sre10_coreX-coreX_m_enroll_mfcc.h5')
    nap_iv = sidekit.StatServer('data/iv_sre04050608_m_training_mfcc.h5')
    test_iv = sidekit.StatServer('data/iv_sre10_coreX-coreX_m_test_mfcc.h5')

    print('Run PLDA scoring evaluation without normalization')    

    meanSN, CovSN = nap_iv.estimate_spectral_norm_stat1(1, 'sphNorm')

    nap_iv_sn1 = copy.deepcopy(nap_iv)
    enroll_iv_sn1 = copy.deepcopy(enroll_iv)
    test_iv_sn1 = copy.deepcopy(test_iv)

    nap_iv_sn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])
    enroll_iv_sn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])
    test_iv_sn1.spectral_norm_stat1(meanSN[:1], CovSN[:1])

    print('Run PLDA rank = {}, {} iterations with Spherical Norm'.format(400, 10))
    mean1, F1, G1, H1, Sigma1 = nap_iv_sn1.factor_analysis(400, rank_G=0, rank_H=None,
                            re_estimate_residual=True,
                            itNb=(10,0,0), minDiv=True, ubm=None,
                            batch_size=1000, numThread=nbThread)
    scores_plda_sn1 = sidekit.iv_scoring.PLDA_scoring(enroll_iv_sn1, test_iv_sn1, test_ndx, mean1, F1, G1, Sigma1)
    scores_plda_sn1.save('scores/scores_plda_eigh_rank_{}_it{}_lapack_sn1_sre10_coreX-coreX_m_mfcc.h5'.format(400, 10))


    print('Plot PLDA scoring DET curves')
    prior = sidekit.effective_prior(0.001, 1, 1)
    # Initialize the DET plot to 2008 settings
    dp = sidekit.DetPlot(windowStyle='sre10', plotTitle='I-Vectors SRE 2010 male')
       
    dp.set_system_from_scores(scores_plda_sn1, keysX[4], sys_name='Cond 5')
        
    dp.set_system_from_scores(scores_2cov, keys[4], sys_name='PLDA Spherical Norm')
    dp.create_figure()
    dp.plot_rocch_det(0)
    dp.plot_DR30_both(idx=0)
    dp.plot_mindcf_point(prior, idx=0)
