# tests for the data generator methods
# TODO proper test suites and unit tests

import sys
import os
import matplotlib.pyplot as plt

# add source files to the path
sys.path.append('/home/constantin/Work/my_projects/neuroWorks/')

import neuroworks

def test_plain_generator():
    datap = '/home/constantin/Work/neurodata_hdd/isbi12_data/raw/train-volume.h5'
    labelp = '/home/constantin/Work/neurodata_hdd/isbi12_data/groundtruth/train-labels.h5'

    gen = neuroworks.PlainTrainDataGenerator(2, datap, labelp, first_dim_changing = False)

    for it, batch_data in enumerate(gen):

        if it > 1:
            break

        raw, labels = batch_data
        labels = labels[...,0]

        fig, ax = plt.subplots(2,2)
        ax[0,0].imshow(raw[0].squeeze(), cmap = 'gray')
        ax[0,1].imshow(labels[0].squeeze(), cmap = 'gray')
        ax[1,0].imshow(raw[1].squeeze(), cmap = 'gray')
        ax[1,1].imshow(labels[1].squeeze(), cmap = 'gray')
        plt.show()
        plt.close()



if __name__ == '__main__':
    test_plain_generator()
