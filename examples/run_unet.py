from unet_mod import hdf5_gen
from unet_mod import unet

train_path = '/home/constantin/Work/neurodata_hdd/isbi12_data/raw/train-volume.h5'
labels_path = '/home/constantin/Work/neurodata_hdd/isbi12_data/groundtruth/train-labels.h5'
test_path = '/home/constantin/Work/neurodata_hdd/isbi12_data/raw/test-volume.h5'

def train_isbi():
    train_gen = hdf5_gen.Hdf5DataProvider2D(train_path, labels_path)

    n_layers = 3
    n_features = 16

    net = unet.Unet(channels = train_gen.channels,
            n_class = train_gen.n_class,
            layers = n_layers,
            features_root = n_features)

    trainer = unet.Trainer(net, optimizer = 'momentum', opt_kwargs = dict(momentum=.8))

    save_path = trainer.train(train_gen,
            '/home/constantin/Work/home_hdd/nnets',
            training_iters = 20,
            epochs = 100,
            display_step = 2)

    return save_path


def predict_isbi(path):
    pass


if __name__ == '__main__':
    train_isbi()
