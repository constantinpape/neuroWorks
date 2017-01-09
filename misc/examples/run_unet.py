from unet_mod import hdf5_gen
from unet_mod import unet

import h5py
import numpy as np

# paths on sirherny
train_path = '/home/constantin/Work/neurodata_hdd/isbi12_data/raw/train-volume.h5'
labels_path = '/home/constantin/Work/neurodata_hdd/isbi12_data/groundtruth/train-labels.h5'
test_path = '/home/constantin/Work/neurodata_hdd/isbi12_data/raw/test-volume.h5'

# mobile paths
#train_path = '/home/consti/Work/data_neuro/nature_experiments/isbi12_data/raw/train-volume.h5'
#labels_path = '/home/consti/Work/data_neuro/nature_experiments/isbi12_data/groundtruth/train-labels.h5'
#test_path = '/home/consti/Work/data_neuro/nature_experiments/isbi12_data/raw/test-volume.h5'

def debug_loader():

    import matplotlib.pyplot as plt

    train_gen = hdf5_gen.Hdf5DataProvider2D(train_path, labels_path)

    im1, label1 = train_gen._next_data()
    im2, label2 = train_gen._next_data()

    print label1.min(), label1.max()

    fig, ax = plt.subplots(2,2)

    ax[0][0].imshow(im1, cmap = 'gray')
    ax[0][1].imshow(label1, cmap = 'gray')

    ax[1][0].imshow(im2, cmap = 'gray')
    ax[1][1].imshow(label2, cmap = 'gray')

    plt.show()


# U-Net params:
n_class = 2
n_layers = 3
n_features = 16

def train_isbi():
    train_gen = hdf5_gen.Hdf5DataProvider2D(train_path, labels_path)

    net = unet.Unet(channels = train_gen.channels,
            n_class = n_class,
            layers = n_layers,
            features_root = n_features)

    trainer = unet.Trainer(net, optimizer = 'momentum', opt_kwargs = dict(momentum=.8))

    save_path = trainer.train(train_gen,
            '/home/constantin/Work/home_hdd/nnets',
            training_iters = 20,
            epochs = 100,
            #epochs = 10,
            display_step = 2)

    return save_path


def predict_isbi(path, keep_channel = 0):
    #test_gen = hdf5_gen.Hdf5DataProvider2D(test_path)
    with h5py.File(test_path) as f:
        test_data = f['data'][:]
        test_data = test_data.transpose((2,0,1))
        test_data = test_data[:,:,:,None]

    net = unet.Unet(channels = 1,
            n_class = n_class,
            layers = n_layers,
            features_root = n_features)

    pred = net.predict(path, test_data)
    pred = pred[:,:,:,keep_channel]
    # normalize
    pred -= pred.min()
    pred /= pred.max()
    return pred


def view_prediction(prediction):
    import matplotlib.pyplot as plt
    with h5py.File(test_path) as f:
        data = f['data'][:,:,0]

    pred0 = prediction[0,:,:]

    # pad the prediction
    pad = (data.shape[0] - pred0.shape[0]) / 2
    pred0 = np.pad(pred0, ((pad,pad),(pad,pad)), 'constant' )

    assert pred0.shape == data.shape

    fig, ax = plt.subplots(2)

    ax[0].imshow(data, cmap = 'gray')
    ax[1].imshow(pred0, cmap = 'gray')

    plt.show()




if __name__ == '__main__':
    #debug_loader()

    #save_path = train_isbi()
    #print "Network trained and saved to:"
    #print save_path

    save_path = "/home/constantin/Work/home_hdd/nnets/model.cpkt"
    prediction = predict_isbi(save_path)

    view_prediction(prediction)
