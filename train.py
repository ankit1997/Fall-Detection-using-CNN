import os
import sys
import config
import numpy as np
import tensorflow as tf
from model import cnn_model
from validate import validation
from data_loader import DataLoader

def train(dataLoader,
	validate_after=10, 
	resume=False):
	
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
			prev_session = os.path.join("saved_model", "model.ckpt")
			saver.restore(sess, prev_session)
			print("Using previous session: {}".format(prev_session))
		except:
			print("Creating a new session.")
	
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
			saver.save(sess, config.ckpt)
			validation(sess, model, dataLoader, valid_writer, e)
			train_writer.add_summary(tb, global_step=e)

	print("Calculating validation accuracy...")

	accuracy = 0.0

	for sensor, label in dataLoader.next_validation():
		# Run the graph.
		pred = sess.run(model['prediction'], 
						feed_dict={
							model['sensor_data']: sensor,
							model['label']: label
						})
		
		accuracy += np.count_nonzero(pred == label) / pred.shape[0]

	accuracy = accuracy / dataLoader.valid_batches
	print("Validation set accuracy: {} %".format(accuracy))

	sess.close()

if __name__ == "__main__":
	dataLoader = DataLoader(sys.argv[1], reload=True)
	
	if '--resume' in sys.argv[1:]:
		resume = True
	else:
		resume =  False
	
	train(dataLoader, resume=resume)