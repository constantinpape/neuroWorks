# basis class for a model

import os
import tensorflow as tf

import layers

class Model(object):
    """
    Basis class for a network model.
    Implements training and prediction.
    The inheriting classes need to implement the architecture.
    """

    def __init__(self, model_params, optimize_params):
        """
        Initialize the model, specifying model and optimization paramaters.
        @param model_params:    dictionary with the parameters for the model.
        @param optimize_params: dictionary with the parameters for the optimiser.
        """
        self.model_params = model_params
        self.optimize_params = optimize_params


    def architecture(self):
        """
        Defines the architecture of the model.
        Needs to be implemented in the inheriting models.
        """
        raise AttributeError("architecture is not implemented for the basis model class.")


    def model_params_descriptions(self):
        """
        Returns a dictionary containing the expected model param names and their description.
        """
        raise AttributeError("model_params_descriptions is not implemented for the basis model class.")


    # TODO implement
    def optimize_params_descriptions(self):
        """
        Returns a dictionary containing the expected optimize param names and their description.
        """
        pass


    # TODO implement
    # TODO need to incorporate validation data properly!
    # TODO Need to specify what happens if the save_path exists
    def train(self, train_gen, save_path):
        """
        Train the model.
        @param train_gen: Generator for the training data.
        @param save_path: Path to save the final model.
        """
        pass


    # TODO implement
    def predict(self, save_path, test_data):
        """
        Predict the model from a given training checkpoint.
        @param save_path: Path to training checkpoint.
        @param test_data: Test data as numpy array.
        """
        pass
