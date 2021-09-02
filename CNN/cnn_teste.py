import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflowonspark import TFCluster, TFNode
import argparse



def main_fun(args, ctx):

    strategy = tf.distribute.MirroredStrategy()

    path='/home/gta/SBEG/datasetimage/'



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

    for files in Files:
        cpath = os.path.join(path, files)
        cpath = os.path.join(cpath) #, 'images')
        for img in os.listdir(cpath):
            image_array = cv2.imread(os.path.join(cpath, img),cv2.IMREAD_GRAYSCALE)
#            print(image_array)
#            print(type(image_array))
#            print(image_array.shape)
            data.append(image_array)
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
    #===================================================================================


    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    kfolds=[]
    for train, test in kfold.split(data, labels):
        x_train, x_test = data[train], data[test]
        y_train, y_test = labels[train], labels[test]
        
        model = keras.Sequential([
        layers.Input(shape=(6, 6, 1)),
    	layers.Conv2D(64, 3, activation='relu',padding='same'), # 128

    	layers.Conv2D(128,3, activation='tanh',padding='same'), #1024

    	layers.MaxPooling2D(pool_size=(2, 2)), # 12

    	layers.Conv2D(128, 2, activation= 'relu',padding='same'), # 10

    	layers.Dropout(rate=0.3),

    	layers.Conv2D(256,2, activation='relu',padding='same'), # 1024

    	layers.Conv2D(256,2, activation='relu',padding='same'), # 

    	layers.Conv2D(256,2, activation='tanh',padding='same'), # 4

    	layers.Dropout(rate=0.3),

    	layers.Conv2D(64,2, activation='tanh',padding='same'), # 256

    	layers.Dense(units=64, activation='relu'),

    	layers.Flatten(),
    	layers.Dense(1, activation='sigmoid')])

        METRICS = [keras.metrics.TruePositives(name='tp'), keras.metrics.FalsePositives(name='fp'),keras.metrics.TrueNegatives(name='tn'), keras.metrics.FalseNegatives(name='fn'),keras.metrics.BinaryAccuracy(name='accuracy'),keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0004),loss = 'binary_crossentropy',metrics=METRICS)
        model.build()
        model.summary()
    
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=50, restore_best_weights=True)] #EarlyStopping
    
        # Later, train it on the data for x epochs
        history = model.fit(x_train, y_train, epochs=100,batch_size=256,callbacks=callbacks)

        print("Evaluate the best model: \n")
        # Evaluate the best model.
        metrics = model.evaluate(x_test,y_test)
        print("[loss, tp, fp, tn, fn, accuracy, precision, recall, auc, f1]: " +str(metrics))
        kfolds.append(metrics[5])
        with open("/home/gta/SBEG/results/results.txt",'a') as file:
            file.write(str(metrics))
            file.write("\n")

    print("%f (+/- %f)" % (np.mean(kfolds), np.std(kfolds)))

    tf.keras.backend.clear_session()

###########################


from pyspark.context import SparkContext
from pyspark.conf import SparkConf


if __name__ == '__main__':

    # Starts the Spark Context and set the number of executors
    conf = SparkConf().set("spark.cores.max", "8")
    conf.set("spark.executor.memory", "20g")
    sc = SparkContext(conf=conf.setAppName("adaptação-rede-antiga"))
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1
    #conf.set("spark.executor.instances", "4")

    #conf.set("spark.executor.cores", "5")


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

#    x_train, x_test, y_train, y_test = load_data(args.dataset_path)

    # Starts TensorFlowOnSpark
    cluster = TFCluster.run(sc, main_fun,args,num_executors=num_executors, num_ps=0,input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief')

    # Stops TensorFlowOnSpark after the processing is done
    cluster.shutdown()


