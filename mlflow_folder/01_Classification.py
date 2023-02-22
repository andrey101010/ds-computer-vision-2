# start in jupyter notebook mlflow ui before executing this script
from logging import getLogger
import pandas as pd
import mlflow.tensorflow
import mlflow.keras
import keras
import tensorflow as tf
import mlflow
from mlflow_folder.config import TRACKING_URI, EXPERIMENT_NAME

logger = getLogger(__name__)
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Create Callbacks. Terminates the run when 90 % accuracy are reached.
class Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

# Load dataset
mnist = tf.keras.datasets.fashion_mnist

def run_training():
  logger.info("Getting MNIST data")
  (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
  training_images  = training_images / 255.0
  test_images = test_images / 255.0
  
  mlflow.keras.autolog() # or mlflow.tensorflow.autolog() does not work here

  with mlflow.start_run(): # which automatically terminates the run at the end of the with block.
    logger.info(f"Creating model")
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    callback = Callback()
    model.compile(optimizer = tf.keras.optimizers.Adam(),
            loss = 'sparse_categorical_crossentropy',
            metrics=['accuracy'])


    model.fit(training_images, training_labels, epochs=1, callbacks = [callback])


    return print(model.evaluate(test_images, test_labels), model.predict(test_images)[0], test_labels[0])
    
    # logger.info(f"The run is ending")
    # mlflow.end_run() not needed because "with" statement above close each run atomatically

if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    run_training()