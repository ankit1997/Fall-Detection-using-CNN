"""
@author: Ankit Bindal
"""

import config
import tensorflow as tf

def cnn_model(log=False):
    """ Builds a CNN model and returns the inputs/outputs with loss and train_op in a dictionary. """
    
    print("Building the CNN model...")

    # Placeholder for input sensor data.
    sensor_data = tf.placeholder(tf.float32, 
                                shape=(None, *config.FEATURES_SHAPE),
                                name="sensor-data")

    # Placeholder for output class label.
    label = tf.placeholder(tf.uint8, shape=(None, 2), name="label")

    training = tf.placeholder(tf.bool, name="is_training")

    if config.activation == "lrelu":
        activation = tf.nn.leaky_relu
    elif config.activation == "elu":
        activation = tf.nn.elu
    elif config.activation == "relu":
        activation = tf.nn.relu
    elif config.activation == "tanh":
        activation = tf.nn.tanh
    else:
        print("Unknown activation {}".format(config.activation))
        exit()

    with tf.variable_scope("Convolution-layers"):
        
        conv1 = tf.layers.conv1d(sensor_data, 16, 3, 
                                strides=1, 
                                activation=activation,
                                name="conv1")
        pool1 = tf.layers.max_pooling1d(conv1, 2, 2, name="pool1")

        conv2 = tf.layers.conv1d(pool1, 32, 3, 
                                strides=1, 
                                activation=activation,
                                name="conv2")
        pool2 = tf.layers.max_pooling1d(conv2, 2, 2, name="pool2")

        conv3 = tf.layers.conv1d(pool2, 64, 3, 
                                strides=1, 
                                activation=activation,
                                name="conv3")
        pool3 = tf.layers.max_pooling1d(conv3, 2, 2, name="pool3")

    flatten = tf.layers.flatten(pool3)

    with tf.variable_scope("Dense-layers"):

        dense1 = tf.layers.dense(flatten, 512, activation=activation, name="dense1")
        dropout1 = tf.layers.dropout(dense1, training=training, name="dropout1")

        dense2 = tf.layers.dense(dropout1, 64, activation=activation, name="dense2")
        dropout2 = tf.layers.dropout(dense2, training=training, name="dropout2")

        dense3 = tf.layers.dense(dropout2, 8, activation=activation, name="dense3")
        dropout3 = tf.layers.dropout(dense3, training=training, name="dropout3")

        dense4 = tf.layers.dense(dropout3, 4, activation=activation, name="dense4")
        
        with tf.variable_scope("dense5"):
            dense5 = tf.layers.dense(dense4, 2, activation=tf.nn.softmax)
            prediction = tf.argmax(dense5, axis=1)

    loss = tf.losses.softmax_cross_entropy(label, dense5)
    tf.summary.scalar("Loss", loss)
    summary = tf.summary.merge_all()

    if config.optimizer == "adam":
        train = tf.train.AdamOptimizer(config.lr).minimize(loss)
    elif config.optimizer == "adadelta":
        train = tf.train.AdadeltaOptimizer(config.lr).minimize(loss)
    else:
        print("Unknown optimizer: {}".format(config.optimizer))
        exit()

    if log:
        print("Sensor data: {}".format(sensor_data.shape))
        print("Label: {}".format(label.shape))

        print("Conv1: {}".format(conv1.shape))
        print("Pool1: {}".format(pool1.shape))
        print("Conv2: {}".format(conv2.shape))
        print("Pool2: {}".format(pool2.shape))
        print("Conv3: {}".format(conv3.shape))
        print("Pool3: {}".format(pool3.shape))

        print("Flatten: {}".format(flatten.shape))

        print("Dense1: {}".format(dense1.shape))
        print("Dense2: {}".format(dense2.shape))
        print("Dense3: {}".format(dense3.shape))
        print("Dense4: {}".format(dense4.shape))
        print("Dense5: {}".format(dense5.shape))

        # Get total number of trainable parameters
        total_params = 0
        for var in tf.trainable_variables():
            p = 1
            for d in var.shape:
                p *= d.value
            total_params += p
        print("Trainable Parameters: ", total_params)

    model = {
        'sensor_data': sensor_data,
        'label': label,
        'prediction': prediction,
        'loss': loss,
        'train': train,
        'summary': summary,
        'training': training
    }

    return model

def rnn_model(log=False):
    """ Builds a RNN model and returns the inputs/outputs with loss and train_op in a dictionary. """
    
    print("Building the RNN model...")

    # Placeholder for input sensor data.
    sensor_data = tf.placeholder(tf.float32, 
                                shape=(None, *config.FEATURES_SHAPE),
                                name="sensor-data")

    # Placeholder for output class label.
    label = tf.placeholder(tf.uint8, shape=(None, 2), name="label")

    training = tf.placeholder(tf.bool, name="is_training")

    if config.activation == "lrelu":
        activation = tf.nn.leaky_relu
    elif config.activation == "elu":
        activation = tf.nn.elu
    elif config.activation == "relu":
        activation = tf.nn.relu
    elif config.activation == "tanh":
        activation = tf.nn.tanh
    else:
        print("Unknown activation {}".format(config.activation))
        exit()

    output = None
    timesteps = config.FEATURES_SHAPE[0]

    with tf.variable_scope("lstm-layer"):
        cell = tf.nn.rnn_cell.MultiRNNCell([
                            tf.nn.rnn_cell.BasicLSTMCell(num_units=32),
                            tf.nn.rnn_cell.BasicLSTMCell(num_units=64)])
        state = cell.zero_state(config.BATCH_SIZE, tf.float32)

        for i in range(timesteps):
            output, state = cell(sensor_data[:, i, :], state)

    with tf.variable_scope("Dense-layers"):
        dense1 = tf.layers.dense(output, 128, activation=activation, name="dense1")
        dropout1 = tf.layers.dropout(dense1, training=training, name="dropout1")

        dense2 = tf.layers.dense(dropout1, 64, activation=activation, name="dense2")
        dropout2 = tf.layers.dropout(dense2, training=training, name="dropout2")

        dense3 = tf.layers.dense(dropout2, 8, activation=activation, name="dense3")
        dropout3 = tf.layers.dropout(dense3, training=training, name="dropout3")

        dense4 = tf.layers.dense(dropout3, 4, activation=activation, name="dense4")
        
        with tf.variable_scope("dense5"):
            dense5 = tf.layers.dense(dense4, 2, activation=tf.nn.softmax)
            prediction = tf.argmax(dense5, axis=1)

    loss = tf.losses.softmax_cross_entropy(label, dense5)
    tf.summary.scalar("Loss", loss)
    summary = tf.summary.merge_all()

    if config.optimizer == "adam":
        train = tf.train.AdamOptimizer(config.lr).minimize(loss)
    elif config.optimizer == "adadelta":
        train = tf.train.AdadeltaOptimizer(config.lr).minimize(loss)
    else:
        print("Unknown optimizer {}".format(config.optimizer))
        exit()

    if log:
        print("Sensor data: {}".format(sensor_data.shape))
        print("Label: {}".format(label.shape))

        print("LSTM output: {}".format(output.shape))

        print("Flatten: {}".format(flatten.shape))

        print("Dense1: {}".format(dense1.shape))
        print("Dense2: {}".format(dense2.shape))
        print("Dense3: {}".format(dense3.shape))
        print("Dense4: {}".format(dense4.shape))
        print("Dense5: {}".format(dense5.shape))

        # Get total number of trainable parameters
        total_params = 0
        for var in tf.trainable_variables():
            p = 1
            for d in var.shape:
                p *= d.value
            total_params += p
        print("Trainable Parameters: ", total_params)

    model = {
        'sensor_data': sensor_data,
        'label': label,
        'prediction': prediction,
        'loss': loss,
        'train': train,
        'summary': summary,
        'training': training
    }

    return model

if __name__ == "__main__":

    # model = cnn_model(log=True)
    rnn_model()
