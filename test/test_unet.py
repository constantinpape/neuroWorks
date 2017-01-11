# tests for the data generator methods
# TODO proper test suites and unit tests

import sys
import os
import matplotlib.pyplot as plt
import h5py

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
    iterations = 500#1e4
    val_step   = 50#100
    network.train(gen, './tmp.ckpt', iterations, val_step)

def test_unet_testing():
    datap = '/home/constantin/Work/neurodata_hdd/isbi12_data/raw/test-volume.h5'
    gen   = neuroworks.PlainTestDataGenerator(1, datap, first_dim_changing = False)

    network = neuroworks.Unet( {}, {})
    prediction = network.predict('./tmp.ckpt', gen)

    print prediction.shape, prediction.max(), prediction.min()

    with h5py.File(datap) as f:
        ds = f['data']

        for i in (0,1):
            fig, ax = plt.subplots(2)
            raw = ds[:,:,i]
            pred = prediction[i]
            assert raw.shape == pred.shape
            ax[0].imshow(raw, cmap = 'gray')
            ax[0].imshow(pred, cmap = 'gray')
            plt.show()
            plt.close()



if __name__ == '__main__':
    test_unet_testing()
