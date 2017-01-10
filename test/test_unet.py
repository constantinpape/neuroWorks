# tests for the data generator methods
# TODO proper test suites and unit tests

import sys
import os
import matplotlib.pyplot as plt

# add source files to the path
sys.path.append('/home/constantin/Work/my_projects/neuroWorks/')

import neuroworks

def test_unet_training():
    datap = '/home/constantin/Work/neurodata_hdd/isbi12_data/raw/train-volume.h5'
    labelp = '/home/constantin/Work/neurodata_hdd/isbi12_data/groundtruth/train-labels.h5'

    # we use the last 4 slices for validation
    validation_samples = range(26,30)

    gen = neuroworks.PlainTrainDataGenerator(1, datap, labelp,
            first_dim_changing = False, validation_samples = validation_samples)

    # use default params for optimiser and architecture
    network = neuroworks.Unet( {}, {})
    # Train for 1e4 steps and evaluate every 100th step
    network.train(gen, './tmp.ckpt', 1e4, 100)



if __name__ == '__main__':
    test_unet_training()
