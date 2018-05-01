"""
@author: Ankit Bindal
"""

import config
import tensorflow as tf

def cnn_model(log=False):
    print("Building the CNN model...")
    # Placeholder for input sensor data.
    sensor_data = tf.placeholder(tf.float32, 
                                shape=(None, *config.FEATURES_SHAPE),
                                name="sensor-data")

    # Placeholder for output class label.
    label = tf.placeholder(tf.uint8, shape=(None, ), name="label")

    with tf.variable_scope("Convolution-layers"):
        
        conv1 = tf.layers.conv1d(sensor_data, 16, 3, strides=1, name="conv1")
        pool1 = tf.layers.max_pooling1d(conv1, 2, 2, name="pool1")

        conv2 = tf.layers.conv1d(pool1, 32, 3, strides=1, name="conv2")
        pool2 = tf.layers.max_pooling1d(conv2, 2, 2, name="pool2")

        conv3 = tf.layers.conv1d(pool2, 64, 3, strides=1, name="conv3")
        pool3 = tf.layers.max_pooling1d(conv3, 2, 2, name="pool3")

    flatten = tf.layers.flatten(pool3)

    with tf.variable_scope("Dense-layers"):

        dense1 = tf.layers.dense(flatten, 512, name="dense1")
        dropout1 = tf.layers.dropout(dense1, name="dropout1")

        dense2 = tf.layers.dense(dropout1, 64, name="dense2")
        dropout2 = tf.layers.dropout(dense2, name="dropout2")

        dense3 = tf.layers.dense(dropout2, 8, name="dense3")
        dropout3 = tf.layers.dropout(dense3, name="dropout3")

        dense4 = tf.layers.dense(dropout3, 4, name="dense4")
        
        with tf.variable_scope("dense5"):
            dense5 = tf.layers.dense(dense4, 1)
            dense5 = tf.squeeze(dense5)

        prediction = tf.round(dense5, name="prediction")

    loss = tf.losses.mean_squared_error(label, dense5)
    tf.summary.scalar("Loss", loss)
    summary = tf.summary.merge_all()

    train = tf.train.AdamOptimizer(0.0001).minimize(loss)

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
        'summary': summary
    }

    return model

if __name__ == "__main__":

    model = cnn_model(log=True)

    import numpy as np

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(config.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        summ = sess.run(model['summary'], feed_dict={
                                                model['sensor_data']: np.random.random(size=(1, *config.FEATURES_SHAPE)),
                                                model['label']: np.array([1])
                                            })
        writer.add_summary(summ)
