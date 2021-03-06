from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import os
import sys


class MLP:
    """
    Neural network implementation (architecture: multi-layer perceptron) used for classification (distinguishing between
    Farris- and Felsenstein-type trees). If it receives the path to a YAML config file as argument (such as the ones
    specified in ./config), it initializes the network immediately starts the training process.
    """
    # initializes the multi-layer perceptron with the parameters as specified in the config file 
    def __init__(self, path):
        try:
            with open(path, 'r') as stream:
                try:
                    data = yaml.safe_load(stream)
                    try:
                        timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
                        self.dropout = data['dropout']
                        self.layers = data['layers']
                        self.learning_rate = data['learning_rate']
                        self.batch_size = data['batch_size']
                        self.epochs = data['epochs']
                        self.display_step_ep = data['display_step_ep']
                        self.display_step_it = data['display_step_it']
                        self.perc_eval = data['perc_eval']
                        self.multiple_data = data['multiple_data']
                        self.data_file = data['data_file']
                        self.data_length = data['data_length']
                        self.save_network_to = data['save_network_to'] + timestamp if 'save_network_to' in data \
                            else None
                        self.read_network_from = data['read_network_from'] if 'read_network_from' in data else None
                        self.output_file = data['save_network_to'] + timestamp + ".log" if 'save_network_to' in data \
                            else None
                        self.test_only = data['test_only']
                        self.weight_initializer = data['weight_initializer'] if 'weight_initializer' in data \
                            else 'xavier'
                        self.transfer_function = data['transfer_function'] if 'transfer_function' in data else 'relu'
                        self.activation_function = data['activation_function'] if 'activation_function' in data \
                            else 'softmax'
                        self.cost_f = data['cost_function'] if 'cost_function' in data else 'cross-entropy'
                        self.opt = data['optimizer'] if 'optimizer' in data else 'adam'
                        self.offset = data['offset']
                        self.shuf_buffer = data['shuf_buffer'] if 'shuf_buffer' in data else 1000
                        tf.logging.set_verbosity({'debug': 10, 'error': 40, 'fatal': 50, 'info': 20, 'warn': 30}
                                                 [data['log_level']] if 'log_level' in data else 20)
                        self.seed_init = data['seed_init'] if 'seed_init' in data else None
                        self.seed_shuffle = data['seed_shuffle'] if 'seed_shuffle' in data else None
                    except KeyError as e:
                        print("Key error. Please refer to config file spec for more details", e)
                        exit(1)
                    # initializes weights and biases 
                    self.weights = self.init_weights()
                    self.biases = self.init_biases()
                except yaml.scanner.ScannerError as e:
                    print("yaml config is not valid. Please follow spec and provide valid yaml.", e)
                    exit(1)
        except FileNotFoundError:
            print("Config file with path {} not found.".format(path))
            exit(1)
    
    # writes network parameters to log file (as specified in the config file) 
    def init_log_file(self):
        try:
            with open(self.output_file, 'w+') as f:
                f.write("layers: " + str(self.layers) + "\n")
                f.write("learning_rate: " + str(self.learning_rate) + "\n")
                f.write("batch_size: " + str(self.batch_size) + "\n")
                f.write("epochs: " + str(self.epochs) + "\n")
                f.write("display_step_it: " + str(self.display_step_it) + "\n")
                f.write("display_step_ep: " + str(self.display_step_ep) + "\n")
                f.write("perc_eval: " + str(self.perc_eval) + "\n")
                f.write("data_file: " + str(self.data_file) + "\n")
                f.write("data_length: " + str(self.data_length) + "\n")
                if self.save_network_to:
                    f.write("save_network_to: " + self.save_network_to + "\n")
                f.write("weight_initializer: " + str(self.weight_initializer) + "\n")
                f.write("transfer_function: " + str(self.transfer_function) + "\n")
                f.write("activation_function: " + str(self.activation_function) + "\n")
                f.write("cost_function: " + str(self.cost_f) + "\n")
                f.write("optimizer: " + str(self.opt) + "\n")
                f.write("dropout: " + str(self.dropout) + "\n")
                f.write("offset: " + str(self.offset) + "\n")
                if self.seed_init:
                    f.write("seed_init: " + str(self.seed_init) + "\n")
                if self.seed_shuffle:
                    f.write("seed_shuffle: " + str(self.seed_shuffle) + "\n")
        except FileNotFoundError:
            print("Path to output file does not exist.")
            exit(1)

    # initializes weights according to the specified method (glorot_normal or random_normal)
    def init_weights(self):
        weights = []
        initializer = tf.glorot_normal_initializer(seed=self.seed_init)
        for l in range(1, len(self.layers)):
            if self.weight_initializer == 'normal':
                weights.append(tf.Variable(tf.random_normal([self.layers[l - 1], self.layers[l]], seed=self.seed_init)))
            else:
                weights.append(tf.Variable(initializer([self.layers[l - 1], self.layers[l]])))
        return weights

    # initializes biases with zeros
    def init_biases(self):
        biases = []
        initializer = tf.zeros_initializer()
        for l in range(1, len(self.layers)):
            biases.append(tf.Variable(initializer([self.layers[l]])))
        return biases

    # performs feed-forward algorithm and returns network output
    def prediction(self, x, no_activation, training):
        intermediate_layer = x
        # performs feed-forward algorithm from input layer to second last layer
        for i in range(0, len(self.layers) - 2):
            drop_out = tf.layers.dropout(intermediate_layer, rate=self.dropout, training=training)
            # if transfer function in config file is specified as 'sigmoid' 
            if self.transfer_function == 'sigmoid':
                intermediate_layer = tf.nn.sigmoid(tf.add(tf.matmul(drop_out, self.weights[i]), self.biases[i]))
            # default for transfer function is 'relu' 
            else:
                intermediate_layer = tf.nn.relu(tf.add(tf.matmul(drop_out, self.weights[i]), self.biases[i]))
        # intermediary result of feed-forward operation (without activation in last layer)
        result = tf.add(tf.matmul(intermediate_layer, self.weights[len(self.layers) - 2]),
                        self.biases[len(self.layers) - 2])
        # if boolean function parameter 'no_activation' is set to True, the intermediate result is returned
        if no_activation:
            return tf.reshape(result, [tf.shape(result)[0]])
        # if 'no_activation' is set to False and the activation_function is 'sigmoid' (as specified in the config file),
        # the sigmoid function is applied to the intermediate result and returned
        if self.activation_function == 'sigmoid':
            return tf.nn.sigmoid(result)
        # default activation function is softmax
        return tf.nn.softmax(result)

    # returns the cost function as specified in the config file (MSE or sigmoid cross-entropy)
    def cost_function(self, x, y):
        if self.cost_f == 'MSE':
            return tf.losses.mean_squared_error(y, self.prediction(x, False, True))
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=self.prediction(x, True, True))
    
    # returns the optimization algorithm used for training the network as specified in the config file (GSD or Adam)
    def optimizer(self, cost):
        if self.opt == 'GSD':
            return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        return tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

    # set tensorflow record defaults (see https://www.tensorflow.org/api_docs/python/tf/io/decode_csv for details)
    def get_record_defaults(self):
        zeros = tf.zeros(shape=(1,), dtype=tf.float32)
        ones = tf.ones(shape=(1,), dtype=tf.float32)
        return [ones] + [zeros] * (self.layers[0] + self.offset) + [ones]

    # parses row of input file and splits data into features and label  
    def parse_row(self, tf_string):
        data = tf.decode_csv(
            tf.expand_dims(tf_string, axis=0), self.get_record_defaults())
        features = data[self.offset + 1:-1]
        features = tf.stack(features, axis=-1)
        label = data[-1]
        features = tf.squeeze(features, axis=0)
        label = tf.squeeze(label, axis=0)
        return features, label

    # defines input pipeline for train data and returns dataset containing labelled features
    def get_train_data(self):
        data = tf.data.TextLineDataset([self.data_file]).skip(1).shuffle(buffer_size=self.shuf_buffer, seed=self.seed_shuffle)
        test_size = int(self.data_length * self.perc_eval)
        training_data = data.skip(test_size).batch(self.batch_size)
        test_data = data.take(test_size).batch(self.batch_size).repeat()
        return training_data.map(self.parse_row, num_parallel_calls=8), \
            test_data.map(self.parse_row, num_parallel_calls=8)

    # loads, shuffles and prepares test data for network
    def get_test_data(self, path):
        tf.logging.info("Storing data in memory")
        try:
            # load data
            if self.multiple_data:
                df = pd.concat(pd.read_csv(path + "/" + file) for file in os.listdir(path))
            else:
                tf.logging.info("Reading from {}".format(path))
                df = pd.read_csv(path)
        except FileNotFoundError:
            print("File with labelled input data not found.")
            exit(1)
        tf.logging.info("Data is stored in memory")
        df_label = np.array(df.iloc[:, :1].values)
        df = df.iloc[:, 1+self.offset:]
        # separate data into features and labels
        features = np.array(df.iloc[:, :-self.layers[-1]].values)
        labels = np.array(df.iloc[:, -self.layers[-1]:].values)
        return features, labels, df_label

    # saves network model to path as specified in config file
    @staticmethod
    def save_network(path, sess, epoch):
        file_name = path + "_" + str(epoch)
        saver = tf.train.Saver()
        saver.save(sess, file_name)
        print("Saved network in ", file_name)

    # returns accuracies for input data that stems from Felsenstein-, Farris-type trees and both
    @staticmethod
    def compute_accuracies(pred_y, eval_y_data, mode):

        correct_fel = correct_far = sum_fel = sum_far = 0.0

        for i, j in zip(pred_y[0], eval_y_data):
            if j == 0:  # it's felsenstein
                sum_fel += 1
                if i[0] < 0.5:
                    correct_fel += 1
            if j == 1:  # it's farris
                sum_far += 1
                if i[0] >= 0.5:
                    correct_far += 1
        accuracies = {'all': (correct_fel + correct_far) / (sum_fel + sum_far) * 100 if (sum_fel + sum_far) else 0,
                      'felsenstein': correct_fel / sum_fel * 100 if sum_fel > 0 else sum_fel,
                      'farris': correct_far / sum_far * 100 if sum_far > 0 else sum_far}
        tf.logging.info(mode, accuracies)
        return accuracies

    # configures computational graph and starts tensorflow session in which network is trained
    def train(self):

        # get dataset that contains entire data
        train_dataset, test_dataset = self.get_train_data()

        # create iterator to get batches
        iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        train_init_op = iter.make_initializer(train_dataset)
        test_init_op = iter.make_initializer(test_dataset)

        # get features and labels
        X, Y = iter.get_next()

        cost = self.cost_function(X, Y)
        optimizer = self.optimizer(cost)

        saver = tf.train.Saver()

        # initialize network
        init = tf.global_variables_initializer()

        # start session
        sess = tf.InteractiveSession()

        tf.logging.info("Started session")

        sess.run(init)

        if self.read_network_from:
            saver.restore(sess, os.path.join('./', self.read_network_from))

        tf.logging.info("Starting training")

        if self.save_network_to:
            self.init_log_file()
          
            tf.logging.info("Network will be stored in %s" % self.save_network_to)
            tf.logging.info("Network spec will be stored in %s" % self.output_file)

        # train
        for epoch in range(0, self.epochs):

            try:
                # every i epochs (i specified in config file as display_step_ep),
                # the network model is saved and validation results are printed
                if self.display_step_ep:
                    if epoch % self.display_step_ep == 0:
                       
                        tf.logging.info("\nEPOCH # %d" % epoch)
 
                        if self.save_network_to:
                            self.save_network(self.save_network_to, sess, epoch)
                        sess.run(test_init_op)  # switched to test data iterator
                        
                        tf.logging.debug("X: %s", X.eval())
                        
                        # get prediction of network for validation data 
                        pred_y = tf.convert_to_tensor(sess.run([self.prediction(X, False, False)])[0])
                      
                        tf.logging.debug("Predictions: %s", sess.run(pred_y))
                        tf.logging.debug("Target: %s", sess.run(Y))
                    
                        # compute accuracies for validation data 
                        accuracies = tf.metrics.mean_per_class_accuracy(tf.round(Y), tf.round(
                            tf.reshape(pred_y, [tf.shape(pred_y)[0]])), 2)
                        sess.run(tf.local_variables_initializer())
                   
                        tf.logging.info("Test accuracies: %s", sess.run(accuracies))
                        sys.stdout.flush()

                sess.run(train_init_op)
                while True:
                    try:
                        sess.run(optimizer)
                    except tf.errors.OutOfRangeError:
                        break

                # randomize training set after each epoch
                train_dataset = train_dataset.shuffle(buffer_size=self.shuf_buffer)

            except KeyboardInterrupt:
                try:
                    with open(self.output_file, 'a+') as f:
                        f.write("Number of epochs network was trained with: " + str(epoch) + "\n")
                except FileNotFoundError:
                    print("Path to output file does not exist.")
                    exit(1)

        sess.close()

    # test single data file
    def test(self):

        # get test data
        x_data, y_data, data_labels = self.get_test_data(self.data_file)

        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)

        # get prediction of network for test data
        prediction = self.prediction(X, False, False)

        saver = tf.train.Saver()

        # initialize network
        init = tf.global_variables_initializer()

        # start session
        sess = tf.InteractiveSession()

        sess.run(init)

        saver.restore(sess, os.path.join('./', self.read_network_from))

        # compute accuracies of prediction
        accuracies = self.compute_accuracies(sess.run([prediction], feed_dict={X: x_data, Y: y_data}), y_data, "test")

        sess.close()

        return accuracies


if __name__ == '__main__':
    try:
        nn = MLP(sys.argv[1])
        nn.train()
    except IndexError as e:
        print("Please specify a path to a config file as first commandline argument")
        exit(1)


