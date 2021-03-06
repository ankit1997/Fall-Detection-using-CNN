"""
@author: Ankit Bindal
"""

import tensorflow as tf

def validation(sess, 
    model, 
    dataLoader,
    writer,
    epoch):
    """
    Runs the neural network model on validation dataset.
    Args:
        sess : Tensorflow session.
        model : Dict containing model inputs and outputs.
        dataLoader : Object of DataLoader class.
        writer : tf.summary.FileWriter object.
        epoch : Current Epoch.
    Returns:
        Average validation error on validation dataset.
    """

    total_loss = 0.0

    for sensor, label in dataLoader.next_validation():
        # Run the graph.
        loss, tb = sess.run([model['loss'], 
                            model['summary']], 
                            feed_dict={
                                model['sensor_data']: sensor,
                                model['label']: label,
                                model['training']: False
                            })
        total_loss += loss

    avg_loss = total_loss / dataLoader.valid_batches
    print("======================================")
    print("Validation loss : {}".format(avg_loss))
    print("======================================")

    writer.add_summary(tb, global_step=epoch)

    return avg_loss