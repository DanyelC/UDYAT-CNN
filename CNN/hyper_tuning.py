# -*- coding: utf-8 -*-

"""the dropout rate for the three dropout layers
the number of filters for the convolutional layers
the number of units for the dense layer
its activation function """
import cv2
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from random import randrange
#from keras import backend as K
from sklearn.model_selection import train_test_split, KFold
from tensorflowonspark import TFCluster, TFNode
import argparse
import pyarrow
from pyarrow import hdfs
import imagecodecs

def main_fun(args, ctx):

    strategy = tf.distribute.MirroredStrategy()

    color_mode = "grayscale"
    number_colour_layers = 1
    image_size = (6, 6)
    image_shape = image_size + (number_colour_layers,)
    seed=randrange(10000)


    #path='caminhodataset'
    connect = hdfs.connect("master",9000)

    #======================= Normalização das imagens de treino ========================
    def standardize(img):
        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / std
        return img
    #===================================================================================


    data = []
    labels = []
    Files = ['benign', 'malicious']
    label_val = 0

    #for files in Files:
    #    cpath = os.path.join(path, files)
    #    cpath = os.path.join(cpath) #, 'images')
    #    for img in os.listdir(cpath):
    #        image_array = cv2.imread(os.path.join(cpath, img),cv2.IMREAD_GRAYSCALE)
    #        data.append(image_array)
    #        labels.append(label_val)
    #    label_val = 1

    for files in Files:
        for i in range(1000): ## botar metade do dataset
            img_file = connect.open('/caminhohdfs/datasetimageTif/'+files+'/'+str(i)+'.tiff', mode='rb')
            img_bytes = img_file.read()
            numpy_img = imagecodecs.tiff_decode(img_bytes) ##RGB
            numpy_img = np.dot(numpy_img[...,:3], [0.2989, 0.5870, 0.1140]) ###grayscale
            data.append(numpy_img)
            labels.append(label_val)
        label_val = 1


    data = np.asarray(data)
    data=data.reshape(-1,6,6,1)
    labels = np.asarray(labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)



    #==================== Normalização das imagens de treino ===========================
    standardized_images=[]

    for img in x_train:
        normalized_image = standardize(img[:,:,0])
        standardized_images.append(normalized_image)
    x_train = np.array(standardized_images)
    #print(x_train.shape)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    print(x_train.shape)
    #===================================================================================


    #=================================HYPER=================================================

    from keras_tuner import HyperModel

    class CNNHyperModel(HyperModel):
        def __init__(self, input_shape, num_classes):
            self.input_shape = input_shape
            self.num_classes = num_classes

        def build(self, hp):
            model = keras.Sequential([
                layers.Input(shape=(6, 6, 1)),

                layers.Conv2D(filters=hp.Choice('num_filters_1', values=[64,128],default=64), kernel_size=hp.Choice('kernel_size_1', values=[3],default=3),activation=hp.Choice('dense_activation_1', values=['relu', 'tanh', 'sigmoid'],default='relu'),padding='same'), # 24

                layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)), # 12

                layers.Conv2D(filters=hp.Choice('num_filters_2', values=[128, 256],default=256), kernel_size=hp.Choice('kernel_size_2', values=[3],default=3), activation=hp.Choice('dense_activation_2', values=['relu', 'tanh', 'sigmoid'],default='relu'),padding='same'), # 10

                layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)), # 12

                layers.Conv2D(filters=hp.Choice('num_filters_3', values=[384,512],default=384), kernel_size=hp.Choice('kernel_size_3', values=[3],default=3), activation=hp.Choice('dense_activation_3', values=['relu', 'tanh', 'sigmoid'],default='relu'),padding='same'), # 8

                layers.Conv2D(filters=hp.Choice('num_filters_4', values=[384,512],default=384), kernel_size=hp.Choice('kernel_size_4', values=[3],default=3), activation=hp.Choice('dense_activation_4', values=['relu', 'tanh', 'sigmoid'],default='relu'),padding='same'), # 6

                layers.Conv2D(filters=hp.Choice('num_filters_5', values=[256,512],default=256), kernel_size=hp.Choice('kernel_size_5', values=[3],default=3), activation=hp.Choice('dense_activation_5', values=['relu', 'tanh', 'sigmoid'],default='relu'),padding='same'), # 4

                layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)), # 12
                
                layers.Flatten(),

                layers.Dense(units=hp.Choice('dense_units1', values=[1024,2046],default=1024), activation=hp.Choice('dense_activation1', values=['relu', 'tanh', 'sigmoid','softmax'],default='softmax')),
                
                layers.Dropout(rate=0.3),

                layers.Dense(units=hp.Choice('dense_units2', values=[1024,2046],default=1024), activation=hp.Choice('dense_activation2', values=['relu', 'tanh', 'sigmoid','softmax'],default='softmax')),
                
                layers.Dropout(rate=0.3),
                
                layers.Dense(1, activation='sigmoid')])

            METRICS = [keras.metrics.TruePositives(name='tp'), keras.metrics.FalsePositives(name='fp'),keras.metrics.TrueNegatives(name='tn'), keras.metrics.FalseNegatives(name='fn'),keras.metrics.BinaryAccuracy(name='accuracy'),keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]

            model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', min_value=0.00001, max_value=0.01, sampling='LOG',default=0.0001)),loss = 'binary_crossentropy',metrics=METRICS)
            model.build()
            model.summary()
            return model


    hypermodel = CNNHyperModel(input_shape=(6,6,1), num_classes=2)


    HYPERBAND_MAX_EPOCHS = 150 #30, 30 e 3 #BOM BOTAR LA P 200 e usar stopping
    MAX_TRIALS = 5
    EXECUTION_PER_TRIAL = 3


    from keras_tuner.tuners import Hyperband
    #factor: Integer, the reduction factor for the number of epochs and number of models for each bracket. Defaults to 3.
    #hyperband_iterations: Integer, at least 1, the number of times to iterate over the full Hyperband algorithm. One iteration will run approximately 
    # max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs across all trials. It is recommended to set this to as high a value as is within 
    # your resource budget. Defaults to 1.
    tuner = Hyperband(
        hypermodel,
        max_epochs=HYPERBAND_MAX_EPOCHS,
        objective='val_accuracy',
        seed=seed,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='/home/gta/hyperband',
        project_name='CNN-IDS'
    )

    tuner.search_space_summary()

    #N_EPOCH_SEARCH = 30 # 30


    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)]

    #import sklearn
    #tava 150 epocas
    tuner.search(x_train, y_train, epochs=150, validation_data=(x_test,y_test), callbacks=callbacks)


    #=========================================================================================

    print("Summary: \n")
    # Show a summary of the search
    tuner.results_summary()

    print("The best model: \n")
    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]

    from contextlib import redirect_stdout
    with  open("caminhoparahyper/hiperparametros.txt",'a') as f:
        with redirect_stdout(f):
            print(tuner.results_summary())

    #tava 25 epocas
    train=best_model.fit(x_train, y_train, epochs=150,callbacks=callbacks)

    print("Evaluate the best model: \n")
    # Evaluate the best model.
    metrics = best_model.evaluate(x_test,y_test)

    print("loss, tp, fp, tn, fn, accuracy, precision, recall, auc")
    print("====================== \n")
    print(metrics)
    print("====================== \n")



###########################


from pyspark.context import SparkContext
from pyspark.conf import SparkConf


if __name__ == '__main__':

    # Starts the Spark Context and set the number of executors
    conf = SparkConf().set("spark.cores.max", "8")
    conf.set("spark.executor.memory", "20g")
    sc = SparkContext(conf=conf.setAppName("tensorflow_teste"))
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1


    # Parses the arguments from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=64)
    parser.add_argument("--buffer_size", help="size of shuffle buffer", type=int, default=1000)
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=3)
    parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
    parser.add_argument("--dataset_path", help="path to the dataset")
    parser.add_argument("--nn_type", help="type of neural network model to build", default="lstm")
    args = parser.parse_args()


    # Starts TensorFlowOnSpark
    cluster = TFCluster.run(sc, main_fun,args,num_executors=num_executors, num_ps=0,input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief')

    # Stops TensorFlowOnSpark after the processing is done
    cluster.shutdown()