"""
@author: Ankit Bindal
"""

import os
import sys
import config
import argparse
import numpy as np
import model as models
import tensorflow as tf
from validate import validation
from data_loader import DataLoader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def train(dataLoader,
    validate_after=5, 
    resume=False,
    perform_training=True,
    save_best=False,
    model_='cnn'):    
    """
    Perform training and validation of model.
    Args:
        dataLoader : DataLoader object
        validate_after : Number of epochs after which validation is performed.
                         The model is also saved after this.
        resume : If True, a previously saved model file is loaded.
        perform_training : If False, training step is skipped, and final testing is done.
        save_best : If True, save session for epoch with minimum validation loss.
        model_ : String denoting the neural network model to use (RNN or CNN)
    """

    model = None
    if model_ == 'cnn':
        model = models.cnn_model()
    elif model_ == 'rnn':
        model = models.rnn_model()

    sess = tf.Session()
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    if resume:
        try:
            prev_session = config.resume_ckpt
            saver.restore(sess, prev_session)
            print("Using previous session: {}".format(prev_session))
        except Exception as exp:
            print(exp)
            print("Creating a new session.")

    if save_best:
        MIN_VAL_LOSS = 100000000000
    
    if perform_training:
        config.init()
        
        train_writer = tf.summary.FileWriter(os.path.join(config.logdir, "train"), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(config.logdir, "validation"), sess.graph)
        
        for e in range(config.EPOCHS):
            epoch_loss = 0.0

            for sensor, label in dataLoader.next_train():
                # Run the graph.
                loss, _, tb = sess.run([model['loss'], 
                                        model['train'],
                                        model['summary']], 
                                    feed_dict={
                                        model['sensor_data']: sensor,
                                        model['label']: label,
                                        model['training']: True
                                    })
                epoch_loss += loss

            avg_loss = epoch_loss/dataLoader.train_batches
            print("Average loss for epoch {} = {}".format(e, avg_loss))

            if e%validate_after == 0:
                
                val_loss = validation(sess, model, dataLoader, valid_writer, e)

                if save_best:
                    if val_loss < MIN_VAL_LOSS:
                        path = saver.save(sess, config.ckpt, global_step=e)
                        print("Saved model to {}".format(path))
                        MIN_VAL_LOSS = val_loss
                else:
                    path = saver.save(sess, config.ckpt, global_step=e)
                    print("Saved model to {}".format(path))

                train_writer.add_summary(tb, e)


    print("===========================================")
    print("Calculating validation accuracy...")

    accuracies = []
    positives = negatives = 0
    true_positives = true_negatives = false_positives = false_negatives = 0

    for sensor, label in dataLoader.next_validation():
        # Run the graph.
        pred = sess.run(model['prediction'], 
                        feed_dict={
                            model['sensor_data']: sensor,
                            model['label']: label,
                            model['training']: False
                        })

        label = np.argmax(label, axis=1)

        positives += np.count_nonzero(label == 1)
        negatives += np.count_nonzero(label == 0)

        # detects the condition when the condition is present. 
        true_positives += np.count_nonzero(pred + label == 2)

        # does not detect the condition when the condition is absent.
        true_negatives += np.count_nonzero(pred + label == 0)

        # wrongly indicates that a particular condition or attribute is present.
        false_positives += np.count_nonzero(pred > label)
        
        # wrongly indicates that a particular condition or attribute is absent.
        false_negatives += np.count_nonzero(pred < label)
        
        accuracies.append(np.count_nonzero(pred == label) / pred.shape[0] * 100)

    accuracies = np.array(accuracies)
    
    # print(positives, negatives)
    # print("True positives : {}".format(true_positives))
    # print("False negatives: {}".format(false_negatives))
    # print("False positives: {}".format(false_positives))
    # print("True negatives: {}".format(true_negatives))

    print("Sensitivity: {}".format(true_positives/positives))
    print("Specificity: {}".format(true_negatives/negatives))
    print("Precision: {}".format(true_positives/(true_positives+false_positives)))

    print("Min Validation set accuracy: {} %".format(accuracies.min()))
    print("Max Validation set accuracy: {} %".format(accuracies.max()))
    print("Average Validation set accuracy: {} %".format(accuracies.mean()))

    sess.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to dataset")
    parser.add_argument("--resume", help="Resume previous training session", action="store_true")
    parser.add_argument("--validate", help="If set, validation occurs without training", action="store_true")
    parser.add_argument("--saveBest", help="If set, save session only for lowest validation loss epoch", action="store_true")
    parser.add_argument("--usePrevious", help="If set, use previously loaded dataset", action="store_true")
    parser.add_argument("--reload", help="If set, built dataset again", action="store_true")
    parser.add_argument("--cnn", help="Use CNN model", action="store_true")
    parser.add_argument("--rnn", help="Use RNN model", action="store_true")
    args = parser.parse_args()

    path = args.path
    resume = args.resume
    perform_training = not args.validate
    saveBest = args.saveBest
    use_previous = args.usePrevious
    reload = args.reload
    cnn = args.cnn
    rnn = args.rnn

    assert not(cnn and rnn), "Must select single model type."
    assert not(reload and use_previous), "Both reload and usePrevious flags cannot be set."

    if reload:
        use_previous = False

    if cnn:
        model_ = 'cnn'
    elif rnn:
        model_ = 'rnn'
    else:
        print("Unreachable code")
        exit()

    # Load dataset
    dataLoader = DataLoader(path, use_previous=use_previous)
    
    # Train the model
    train(dataLoader, 
        resume=resume, 
        perform_training=perform_training, 
        save_best=saveBest,
        model_=model_)