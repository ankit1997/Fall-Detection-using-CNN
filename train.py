import os
import sys
import config
import numpy as np
import tensorflow as tf
from model import cnn_model
from validate import validation
from data_loader import DataLoader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def train(dataLoader,
    validate_after=10, 
    resume=False,
    perform_training=True):
    
    """
    Trains the fall detection model using the provided training dataset.
    Additionally, runs the validation dataset after every `validate_after` epochs.
    """
    
    model = cnn_model()

    sess = tf.Session()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(os.path.join(config.logdir, "train"), sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(config.logdir, "validation"), sess.graph)

    sess.run(tf.global_variables_initializer())

    if resume:
        try:
            prev_session = os.path.join("saved_model", "model.ckpt-90")
            saver.restore(sess, prev_session)
            print("Using previous session: {}".format(prev_session))
        except:
            print("Creating a new session.")
    
    if perform_training:
        for e in range(config.EPOCHS):
            epoch_loss = 0.0

            for sensor, label in dataLoader.next_train():
                # Run the graph.
                loss, _, tb = sess.run([model['loss'], 
                                        model['train'],
                                        model['summary']], 
                                    feed_dict={
                                        model['sensor_data']: sensor,
                                        model['label']: label
                                    })
                epoch_loss += loss

            print("Average loss for epoch {} = {}".format(e, epoch_loss/dataLoader.train_batches))

            if e%validate_after == 0:
                saver.save(sess, config.ckpt, global_step=e)
                validation(sess, model, dataLoader, valid_writer, e)
                train_writer.add_summary(tb, e)

    print("Calculating validation accuracy...")

    accuracies = []

    for sensor, label in dataLoader.next_validation():
        # Run the graph.
        pred = sess.run(model['prediction'], 
                        feed_dict={
                            model['sensor_data']: sensor,
                            model['label']: label
                        })
        
        accuracies.append(np.count_nonzero(pred == label) / pred.shape[0] * 100)

    accuracies = np.array(accuracies)
    print("Min Validation set accuracy: {} %".format(accuracies.min()))
    print("Max Validation set accuracy: {} %".format(accuracies.max()))
    print("Average Validation set accuracy: {} %".format(accuracies.mean()))

    sess.close()

if __name__ == "__main__":

    resume = '--resume' in sys.argv[1:]
    perform_training = '--validate' not in sys.argv[1:]

    # Load dataset
    dataLoader = DataLoader(sys.argv[1])

    # Train the model
    train(dataLoader, resume=resume, perform_training=perform_training)