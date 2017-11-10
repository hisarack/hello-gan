
import numpy as np
import boto3

from keras.layers import Input
from keras.layers.core import Activation
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

# Discriminator
D = Sequential()
D.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), input_shape=(28, 28, 1), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.02)))
D.add(LeakyReLU(0.2))
D.add(Dropout(0.3))
D.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), input_shape=(28, 28, 1), padding='same'))
D.add(LeakyReLU(0.2))
D.add(Dropout(0.3))
D.add(Flatten())
D.add(Dense(1))
D.add(Activation('sigmoid'))

# compile
optimizer = Adam(lr=0.0002, beta_1=0.5)
D.compile(
    loss='binary_crossentropy', 
    optimizer=optimizer
)

# Generator
G = Sequential()
G.add(Dense(input_dim=100, output_dim=128*7*7, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
G.add(LeakyReLU(0.2))
G.add(Reshape((7, 7, 128)))
G.add(UpSampling2D(size=(2, 2)))
G.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
G.add(LeakyReLU(0.2))
G.add(UpSampling2D(size=(2, 2)))
G.add(Conv2D(1, (5, 5), padding='same'))
G.add(Activation('tanh'))

# compile
G.compile(
    loss='binary_crossentropy', 
    optimizer=optimizer,
    metrics=['accuracy']
)

# compile GAN
D.trainable = False
GAN = Sequential()
GAN.add(G)
GAN.add(D)
GAN.compile(
    loss='binary_crossentropy', 
    optimizer=optimizer
)
D.trainable = True

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, :, :, None]
X_test = X_test[:, :, :, None]

EPOCH = 5
BATCH_SIZE = 256

for epoch in range(EPOCH):
    print("Epoch is", epoch)
    print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
    for index in range(int(X_train.shape[0]/BATCH_SIZE)):
        # Get a random set of input noise and images
        noise = np.random.normal(0, 1, size=[BATCH_SIZE, 100])
        imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)]
        # Generate fake MNIST images
        generatedImages = G.predict(noise)
        X = np.concatenate([imageBatch, generatedImages])
        # Labels for generated and real data
        yDis = np.zeros(2*BATCH_SIZE)
        # One-sided label smoothing from improved WGAN
        yDis[:BATCH_SIZE] = 0.9
        # Train discriminator
        D.trainable = True
        d_loss = D.train_on_batch(X, yDis)
        # Train generator
        noise = np.random.normal(0, 1, size=[BATCH_SIZE, 100])
        yGen = np.ones(BATCH_SIZE)
        D.trainable = False
        g_loss = GAN.train_on_batch(noise, yGen)


# Save weight
D_LOCALPATH = '../model/mnist/D.h5'
G_LOCALPATH = '../model/mnist/G.h5'
GAN_LOCALPATH = '../model/mnist/GAN.h5'
S3_BUCKET = 'hisarack-gan'
D_S3PATH = 'mnist/D.h5'
G_S3PATH = 'mnist/G.h5'
GAN_S3PATH = 'mnist/GAN.h5'
D.save(D_LOCALPATH)
G.save(G_LOCALPATH)
GAN.save(GAN_LOCALPATH)
s3 = boto3.resource('s3')
s3.meta.client.upload_file(D_LOCALPATH, S3_BUCKET, D_S3PATH)
s3.meta.client.upload_file(G_LOCALPATH, S3_BUCKET, G_S3PATH)
s3.meta.client.upload_file(GAN_LOCALPATH, S3_BUCKET, GAN_S3PATH)

