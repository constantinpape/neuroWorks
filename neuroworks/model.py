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

        tf.reset_default_graph() # TODO check what this does and if it is necessary!

        # declare data and label variables
        self.n_channels = model_params.get("n_channels",1)
        self.x = tf.placeholder("float", shape = [None,None,None,self.n_channels])
        # for now, we hardcode 2 class
        self.y = tf.placeholder("float", shape = [None,None,None,2])
        # probability for potential dropout
        self.keep_prob = tf.placeholder("float")

        logits, self.variables = self.architecture()

        # TODO the U-Net example does some potential class weighting here, ignore this for now

        # cross entropy loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( tf.reshape(logits,[-1,2]),
            tf.reshape(self.y,[-1,2]) ) )

        # TODO What is this and why do we not include regularization here
        # to keep track of gradients, deactivate for now
        #self.gradients_node = tf.gradients(loss, self.variables)

        # L2 regularization, turned off with gamma = 0
        gamma = self.optimize_params.get("gamma",1e-3)
        regularizer = gamma * sum([tf.nn.l2_loss(var) for var in self.variables])

        self.loss = loss + regularizer

        self.predicter = layers.pixel_wise_softmax_2(logits)
        self.accuracy  = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(self.predicter,3), tf.argmax(self.y,3))
            ,tf.float32))



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


    # TODO in the U-Net example there is a lot of summary stuff I don't get yet, leave out for now!
    def _init_before_training(self, save_path, restore_path):
        """
        Init before training.
        """
        global_step = tf.Variable(0)

        # TODO what's this - gradient tracking deactivated for now
        #self.norm_gradients_node = tf.Variable(tf.constant(0.,shape=[len(self.gradients_node)]))

        save_folder = os.path.split(save_path)[0]
        debug_folder = os.path.join(save_folder, 'debug')
        if restore_path is not '' and os.path.exists(debug_folder):
            shutil.rmtree(debug_folder)

        # make folders for debug pictures
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)

        # get the optimizer TODO implement more than momentum

        optimizer_key = self.optimize_params.get('optimizer','momentum')
        if optimizer_key == 'momentum':

            # optimizer parameter
            eta = self.optimize_params.get('eta',.2) # learning rate
            decay_rate = self.optimize_params.get('decay_rate',.95)
            decay_steps = self.optimize_params.get('decay_steps',1000)
            momentum  = self.optimize_params.get('momentum',.9)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=eta,
                    global_step=global_step,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=True) # TODO What's staircase?

            # TODO what about nesterov?
            self.optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate_node,
                    momentum = momentum).minimize(self.loss, global_step=global_step)

        else:
            raise AttributeError("Only momentum optimizer implemented for now, you are trying to use " + optimizer_key)

        # TODO this needs to be changed in the new tf versions
        init = tf.initialize_all_variables()

        return init


    # TODO implement checkpoint saving
    def train(self,
        train_gen,
        save_path,
        num_iterations,
        validation_step,
        restore_path = ''):
        """
        Train the model.
        @param train_gen: Generator for the training data.
        @param save_path: Path to save the final model.
        @param num_iterations: Total number of traning iterations.
        @param validation_step: Perform validation (and save if activated) every validation_iter.
        @param restore_path: Continue training from existing model.
        @return:
        """

        if restore_path is not '':
            assert os.path.exists(restore_path)

        # get the validation data
        try:
            test_x, test_y = train_gen.get_validation_samples()
        except RuntimeError:
            raise RuntimeError("The training generator has no validation data, stopping trainng.")

        init = self._init_before_training(save_path, restore_path)

        drop_prob = self.optimize_params.get('drop_prob',.9)

        with tf.Session() as sess:
            sess.run(init)

            if restore_path is not '':
                checkpoint = tf.train.get_checkpoint_state(restore_path)
                if checkpoint and checkpoint.model_checkpoint_path:
                    self.restore(sess, checkpoint.model_checkpoint_path)

            pred_shape = self.run_validation(sess, test_x, test_y) # TODO add save name

            print "Start optimization" # TODO proper logging
            for step, batch_data in enumerate(train_gen):
                batch_x, batch_y = batch_data

                # gradient step
                _, loss, lr = sess.run([self.optimizer, self.loss, self.learning_rate_node ],
                        feed_dict = {self.x : batch_x,
                                    self.y : batch_y,
                                    self.keep_prob : drop_prob})

                if step % validation_step == 0:
                    self.run_validation(sess, test_x, test_y)
                    # TODO make checkpoint point

                # TODO log more stuff
                print step, 'done'

                if step >= num_iterations:
                    break

            # save the trained net
            self.save(sess,save_path,'trained')


    def run_validation(self,
            session,
            val_x,
            val_y,
            save_name=''):
        """
        Run validation.
        """

        acc, loss, pred = session.run([self.accuracy,self.loss,self.predicter],
                feed_dict = {self.x : val_x,
                            self.y : val_y,
                            self.keep_prob : 1.})
        shape = pred.shape

        # TODO propper logging
        print "Validation accuracy:", acc
        print "Validation loss:", loss

        # TODO save debug images

        return pred.shape



    # TODO implement
    def restore(self,session,checkpoint):
        pass


    def save(self,session,save_path,name):
        """
        Saves the current session as checkpoint.
        @params session: Current session.
        @params save_path: Save Path.
        @params name: Additional name indicator.
        @returns: Updated save_path.
        """
        # TODO naming ?!
        saver = tf.train.Saver()
        save_path = saver.save(session, save_path)
        return save_path



    # TODO implement
    def predict(self, save_path, test_gen):
        """
        Predict the model from a given training checkpoint.
        @param save_path: Path to training checkpoint.
        @param test_gen:  Generator for the test data.
        """
        pass
