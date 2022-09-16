from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

class CGAN:

    def __init__(self, batch, channels, buffer, classes,imsize,latent, x_train, y_train, x_test, y_test, gen_loss, dis_loss, dis_optimizer, gen_optimizer):
        self.batch_size = batch
        self.num_channels = channels
        self.buffer_size = buffer
        self.num_classes = classes
        self.image_size = imsize
        self.latent_dim = latent
        self.xtrain = x_train
        self.ytrain = y_train
        self.xtest = x_test
        self.ytest = y_test
        self.generator_loss = gen_loss
        self.discriminator_loss = dis_loss
        self.dataset = None
        self.dis_optimizer = dis_optimizer
        self.gen_optimizer = gen_optimizer

    def build_dataset(self, d1, d2, c, n):
        x_values = np.concatenate([self.xtrain, self.xtest])
        y_labels = np.concatenate([self.ytrain, self.ytest])
        x_values = x_values.astype("float32") / 255.0
        x_values = np.reshape(x_values, (-1, d1, d2, c))
        y_labels = keras.utils.to_categorical(y_labels, n)
        dataset = tf.data.Dataset.from_tensor_slices((x_values, y_labels))
        dataset = dataset.shuffle(buffer_size=self.buffer_size).batch(self.batch_size)
        self.dataset = dataset
        return dataset
    
    def build_generator(self):
        gen_channels = self.latent_dim + self.num_classes
        generator = keras.Sequential([
        keras.layers.InputLayer((gen_channels,)),
        layers.Dense(7 * 7 * gen_channels),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, gen_channels)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        ])
        return generator
        
    
    def build_discriminator(self, d1, d2):
        disc_channels = self.num_channels + self.num_classes
        discriminator = keras.Sequential([
        keras.layers.InputLayer((d1, d2, disc_channels)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
        ])
        return discriminator
    
    
    def train(self, dataset):
        x_data, y_labels = dataset
        labels = y_labels[:, :, None, None]
        labels = tf.repeat(labels, repeats=[self.image_size * self.image_size])
        labels = tf.reshape(labels, (-1, self.image_size, self.image_size, self.num_classes))
        batch = tf.shape(x_data)[0]
        latent_vectors = tf.random.normal(shape=(batch, self.latent_dim))
        vector_labels = tf.concat([latent_vectors, labels], axis=1)
        generator, discriminator = self.build_generator(), self.build_discriminator(self.image_size, self.image_size)
        generated_images = generator(vector_labels)
        image_and_labels = tf.concat([generated_images, labels], -1)
        real_image_and_labels = tf.concat([x_data, labels], -1)
        combined_images = tf.concat([image_and_labels, real_image_and_labels], axis=0)
        labels = tf.concat([tf.ones((batch, 1)), tf.zeros((batch, 1))], axis=0)
        with tf.GradientTape() as tape:
            predictions = discriminator(combined_images)
            disc_loss = self.discriminator_loss(labels, predictions)
        gradients = tape.gradient(disc_loss, discriminator.trainable_weights)
        self.dis_optimizer.apply_gradients(zip(gradients, discriminator.trainable_weights))
        random_vectors = tf.random.normal(shape=(batch, self.latent_dim))
        random_vector_labels = tf.concat([random_vectors, vector_labels], axis=1)
        misleading_labels = tf.zeros((batch, 1))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.generator_loss(misleading_labels, predictions)
        grads = tape.gradient(g_loss, generator.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
        return
    
    def test(self):
        # TODO: add tests
    
    def plot_progress(self):
        #TODO: add progress tracking
