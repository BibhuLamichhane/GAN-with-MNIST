import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization as bn
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam as adam, SGD
from tensorflow.keras.datasets import mnist

data = mnist.load_data()

(x_train, y_train), (x_test, y_test) = data

x_train = x_train / 255 * 2 - 1
x_test = x_test / 255 * 2 - 1

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

latent_dim = 200


def training_test_data(n):
    new_x_train = []
    for i in range(len(y_train)):
        if y_train[i] == n:
            new_x_train.append(x_train[i])

    new_x_train = np.array(new_x_train)
    new_y_train = np.zeros(len(new_x_train))
    new_y_train[:] = n

    new_x_test = []
    for i in range(len(y_test)):
        if y_test[i] == n:
            new_x_test.append(x_test[i])

    new_x_test = np.array(new_x_test)
    new_y_test = np.zeros(len(new_x_test))
    new_y_test[:] = n

    return new_x_train, new_y_train, new_x_test, new_y_test


def gen(latent_dim):
    i = Input(shape=latent_dim)
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
    x = bn(momentum=0.8)(x)
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
    x = bn(momentum=0.8)(x)
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
    x = bn(momentum=0.8)(x)
    x = Dense(784, activation='tanh')(x)

    model = Model(i, x)
    return model


def discriminator(size):
    i = Input(shape=(size,))
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(i, x)

    return model


Discriminator = discriminator(784)

Discriminator.compile(
    loss='binary_crossentropy',
    optimizer=adam(0.0002, 0.5),
    metrics=['accuracy']
)

generator = gen(latent_dim)

x = Input(shape=(latent_dim,))
img = generator(x)
Discriminator.trainable = False
fake_img = Discriminator(img)

main_model = Model(x, fake_img)
main_model.compile(
    loss='binary_crossentropy',
    optimizer=adam(0.0002, 0.5),
)

batch = 32
epochs = 5000
sample_period = 100

ones = np.ones(batch)
zeros = np.zeros(batch)

discriminator_loss = []
generator_loss = []

if not os.path.exists('gan_images'):
    os.makedirs('gan_images')


def sample_images(epoch, n):
    rows, cols = 5, 5
    noise = np.random.randn(rows * cols, latent_dim)
    imgs = generator.predict(noise)

    imgs = 0.5 * imgs + 0.5

    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(imgs[idx].reshape(28, 28), cmap='gray')
            axs[i, j].axis('off')
            idx += 1
    fig.savefig(f"gan_images/{n}--{epoch}.png")
    plt.close()


n = 0
for n in range(10):
    new_x_train, new_y_train, new_x_test, new_y_test = training_test_data(n)
    for epoch in range(epochs):

        idx = np.random.randint(0, new_x_train.shape[0], batch)
        real_imgs = new_x_train[idx]

        noise = np.random.randn(batch, latent_dim)
        fake_imgs = generator.predict(noise)

        d_loss_real, d_acc_real = Discriminator.train_on_batch(real_imgs, ones)
        d_loss_fake, d_acc_fake = Discriminator.train_on_batch(fake_imgs, zeros)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        noise = np.random.randn(batch, latent_dim)
        g_loss = main_model.train_on_batch(noise, ones)

        noise = np.random.randn(batch, latent_dim)
        g_loss = main_model.train_on_batch(noise, ones)

        discriminator_loss.append(d_loss)
        generator_loss.append(g_loss)

        if epoch % 100 == 0:
            print(f"epoch: {epoch + 1}/{epochs}, d_loss: {d_loss:.2f}, \
          d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")

        if epoch % sample_period == 0:
            sample_images(epoch, n)
