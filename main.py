import cv2
import numpy as np
import os
from keras import callbacks
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
import keras
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.model_selection import train_test_split
import splitfolders
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator , array_to_img,img_to_array,load_img
from tensorflow.keras.regularizers import l2


imgSize = 128
modelName = 'Sports-CNN'
def createLabel(img):
    word_label = img.split('_')
    if word_label[0] == 'Basketball':
        return np.array([1, 0, 0, 0, 0, 0])
    elif word_label[0] == 'Football':
        return np.array([0, 1, 0, 0, 0, 0])
    elif word_label[0] == 'Rowing':
        return np.array([0, 0, 1, 0, 0, 0])
    elif word_label[0] == 'Swimming':
        return np.array([0, 0, 0, 1, 0, 0])
    elif word_label[0] == 'Tennis':
        return np.array([0, 0, 0, 0, 1, 0])
    elif word_label[0] == 'Yoga':
        return np.array([0, 0, 0, 0, 0, 1])


#get train-validation ratio

#TRAIN_DIR = "nn23-sports-image-classification"
#splitfolders.ratio(TRAIN_DIR, output="output", ratio=(.8, .2))

#data agumentation
"""
IMG_SIZE= 128
train_datagen = ImageDataGenerator(
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    height_shift_range=0.2,
                                    fill_mode='reflect'
)

folder_dir = "Train/basketball"
i=0
for images in os.listdir(folder_dir):
    img= load_img(rf"Train/basketball/{images}")
    x=img_to_array(img)
    x=x.reshape((1,)+x.shape)
    i=0
    for batch in train_datagen.flow(x,batch_size=1,save_to_dir='augmented images/Abasketball',save_prefix='Basketball',save_format='JPG'):
        i+=1
        if i>11:
            break
"""

trainRoot ="Train"
def createTrainData():
    training_data = []
    for img in tqdm(os.listdir(trainRoot)):
        path = os.path.join(trainRoot, img)
        img_data = cv2.imread(path,1)
        img_data = cv2.resize(img_data, (imgSize, imgSize))
        training_data.append([np.array(img_data), createLabel(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

testRoot = 'Test'
def create_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(testRoot)):
        path = os.path.join(testRoot, img)
        img_data = cv2.imread(path,1)
        img_data = cv2.resize(img_data, (imgSize, imgSize))
        testing_data.append([np.array(img_data),img])
    np.save('test_data.npy', testing_data)
    return testing_data

validationRoot = "output/val"
def create_validation_data():
    validation_data = []
    for img in tqdm(os.listdir(validationRoot)):
        path = os.path.join(validationRoot,img)
        img_data = cv2.imread(path,1)
        img_data = cv2.resize(img_data, (imgSize, imgSize))
        validation_data.append([np.array(img_data), createLabel(img)])
    shuffle(validation_data)
    np.save('validation_data.npy', validation_data)
    return validation_data

if (os.path.exists('train_data.npy')):
    train_data =np.load('train_data.npy',allow_pickle=True)
else:
    train_data = createTrainData()

if (os.path.exists('test_data.npy')):
    test_data =np.load('test_data.npy',allow_pickle=True)
else:
    test_data = create_test_data()

if (os.path.exists('validation_data.npy')):
    validation_data =np.load('validation_data.npy',allow_pickle=True)
else:
    validation_data = create_validation_data()

train = train_data
test = test_data
validation = validation_data
X_train = np.array([i[0] for i in train]).reshape(-1, imgSize, imgSize, 3)
y_train = [i[1] for i in train]


X_validation = np.array([i[0] for i in validation]).reshape(-1, imgSize, imgSize, 3)
y_validation = [i[1] for i in validation]


X_test = np.array([i[0] for i in test]).reshape(-1, imgSize, imgSize, 3)
y_test = [i[1] for i in test]

tf.reset_default_graph()

#ZFNet
"""inputLayer = input_data([None, imgSize, imgSize, 3], name="input")
convLayer1 = conv_2d(inputLayer, 96, 7,regularizer='L2',activation="relu",strides=2)
norm1 = tflearn.layers.normalization.batch_normalization (convLayer1, stddev=0.002, trainable=True, restore=True)
poolLayer1 = max_pool_2d(norm1, 3)

convLayer2 = conv_2d(poolLayer1,256,5,regularizer='L2',strides=1,activation='relu')
norm2 = tflearn.layers.normalization.batch_normalization (convLayer2,stddev=0.002, trainable=True, restore=True)
poolLayer2 = max_pool_2d(norm2,3,strides=2)

convLayer3 = conv_2d(poolLayer2,384,3,regularizer='L2',strides=1,activation='relu')
norm3 = tflearn.layers.normalization.batch_normalization (convLayer3, stddev=0.002, trainable=True, restore=True)
convLayer4 = conv_2d(norm3,384,3,regularizer='L2',strides=1,activation='relu')
norm4 = tflearn.layers.normalization.batch_normalization (convLayer4, stddev=0.002, trainable=True, restore=True)
convLayer5 = conv_2d(norm4,256,3,regularizer='L2',strides=1,activation='relu')
norm5 = tflearn.layers.normalization.batch_normalization (convLayer5, stddev=0.002, trainable=True, restore=True)

poolLayer3 = max_pool_2d(norm5, 3)

dropoutLayer1 = dropout(poolLayer3,0.5)
fullyLayer1 = fully_connected(dropoutLayer1,4069,activation='relu')

dropoutLayer2 = dropout(fullyLayer1, 0.5)
fullyLayer2 = fully_connected(dropoutLayer2,4069,activation='relu')
outputLayer = fully_connected(dropoutLayer2,6,activation='softmax')

cnn_layers = regression(outputLayer, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)"""


#12 layers deep CNN model
"""conv_input = input_data(shape=[None, imgSize, imgSize, 3], name='input')
conv1 = conv_2d(conv_input, 32, 5,regularizer='L2',activation='relu')
norm1 = tflearn.layers.normalization.batch_normalization (conv1, stddev=0.002, trainable=True, restore=True)
pool1 = max_pool_2d(norm1, 5)

conv2 = conv_2d(pool1, 64, 5,regularizer='L2', activation='relu')
norm2 = tflearn.layers.normalization.batch_normalization (conv2,stddev=0.002, trainable=True, restore=True)
pool2 = max_pool_2d(norm2, 5)

conv3 = conv_2d(pool2, 128, 5,regularizer='L2', activation='relu')
norm3 = tflearn.layers.normalization.batch_normalization (conv3,stddev=0.002, trainable=True, restore=True)
pool3 = max_pool_2d(norm3, 5)

conv4 = conv_2d(pool3, 64, 5, regularizer='L2',activation='relu')
norm4 = tflearn.layers.normalization.batch_normalization (conv4,stddev=0.002, trainable=True, restore=True)
pool4 = max_pool_2d(norm4, 5)

conv5 = conv_2d(pool4, 32, 5,regularizer='L2', activation='relu')
norm5 = tflearn.layers.normalization.batch_normalization (conv5, stddev=0.002, trainable=True, restore=True)
pool5 = max_pool_2d(norm5, 5)

fully_layer1 = fully_connected(pool5, 1024, activation='relu')
fully_layer1 = dropout(fully_layer1, 0.5)

cnn_layers = fully_connected(fully_layer1, 6, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets',metric="accuracy")
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)"""


#LEtNet 5
conv_input = input_data(shape=[None, imgSize, imgSize, 3], name='input')
conv1 = conv_2d(conv_input, 6, 5, activation='tanh')
norm1 = tflearn.layers.normalization.batch_normalization(conv1, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9,
                                                         stddev=0.002, trainable=True, restore=True, reuse=False,
                                                         scope=None, name='BatchNormalization')
pool1 = max_pool_2d(norm1, 2, strides=2)

conv2 = conv_2d(pool1, 16, 5, activation='tanh')
norm2 = tflearn.layers.normalization.batch_normalization(conv2, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9,
                                                         stddev=0.002, trainable=True, restore=True, reuse=False,
                                                         scope=None, name='BatchNormalization')
pool2 = max_pool_2d(norm2, 2, strides=2)

conv3 = conv_2d(pool2, 120, 5, activation='tanh')
norm3 = tflearn.layers.normalization.batch_normalization(conv3, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9,
                                                         stddev=0.002, trainable=True, restore=True, reuse=False,
                                                         scope=None, name='BatchNormalization')
fully_layer = fully_connected(norm3, 84, activation='tanh')
fully_layer = dropout(fully_layer, 0.4)
cnn_layers = fully_connected(fully_layer, 6, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy',
                        name='targets', metric="accuracy")
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)





if (os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
              validation_set=({'input': X_validation}, {'targets': y_validation}), snapshot_step= 500,
              show_metric=True, run_id=modelName)
    model.save('model.tfl')



#make test CSV file
import csv
def checkOutput(output):
    max_num = max(output)
    index = np.where(output == max_num)
    index = int(index[0])
    return index


file = open('test.csv', 'w',newline='')
writer = csv.writer(file)
row = ["image_name","label"]
writer.writerow(row)

predicted_values = model.predict(X_test)

for i in range(len(predicted_values)):
    output = checkOutput(predicted_values[i])
    row=[y_test[i],output]
    writer.writerow(row)

