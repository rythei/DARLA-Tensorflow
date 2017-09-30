import tensorflow as tf
from sklearn.utils import shuffle
from utils import *

class DAE():
    def __init__(self):
        self.save_path = 'checkpoints/dae'

        self.latent_dim = 100
        self.batch_size = 64
        self.epochs = 10
        self.learning_rate = 1e-3
        self.from_scratch = True

        self.noise = tf.random_normal([self.batch_size, 64, 64, 1], 0, .1, tf.float32)

        self.x = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 1])
        self.noisy_x = tf.add(self.x, self.noise)

        self.z = self.encoder(self.noisy_x, 'encoder', None)
        self.x_hat = self.decoder(self.z, 'decoder', None)

        self.cost = tf.reduce_mean(tf.pow(self.x - self.x_hat, 2))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def encoder(self, x, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.reshape(x, [-1, 64, 64, 1])

            conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=4, strides=2, padding='same',
                                     activation=tf.nn.relu)

            # output_size = output_after_conv(64, kernel_size=4, strides=2, padding='SAME') ## 32

            ## size: [batchsize, 32, 32, 32]
            conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=4, strides=2, padding='same',
                                     activation=tf.nn.relu)

            ## size: [batchsize, 16, 16, 32]
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=4, strides=2, padding='same',
                                     activation=tf.nn.relu)

            ## size: [batchsize, 8, 8, 64]
            conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=4, strides=2, padding='same',
                                     activation=tf.nn.relu)

            ## size: [batchsize, 4, 4, 64]
            flat = tf.reshape(conv4, [-1, 4 * 4 * 64])

            dense = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)

            z = tf.layers.dense(inputs=dense, units=self.latent_dim)

        return z

    def decoder(self, z, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            z = tf.reshape(z, [-1, self.latent_dim])

            dense = tf.layers.dense(inputs=z, units=256, activation=tf.nn.relu)

            dense = tf.reshape(dense, [-1, 1, 1, 256])

            deconv1 = tf.layers.conv2d_transpose(inputs=dense, filters=64, kernel_size=4, strides=2, padding='valid',
                                                 activation=tf.nn.relu)

            deconv2 = tf.layers.conv2d_transpose(inputs=deconv1, filters=64, kernel_size=4, strides=2, padding='same',
                                                 activation=tf.nn.relu)

            deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, filters=32, kernel_size=4, strides=2, padding='same',
                                                 activation=tf.nn.relu)

            deconv4 = tf.layers.conv2d_transpose(inputs=deconv3, filters=32, kernel_size=4, strides=2, padding='same',
                                                 activation=tf.nn.relu)

            output_layer = tf.layers.conv2d_transpose(inputs=deconv4, filters=1, kernel_size=4, strides=2, padding='same')

        return output_layer

    def j_func(self, x):
        x = np.reshape(x, (-1, 64, 64, 1))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path=self.save_path)
            z = sess.run([self.z], feed_dict={self.x: x})

        return z

    def get_shape(self):
        return self.x_hat.get_shape(), self.noisy_x.get_shape()

    def train(self, x):
        N = len(x)
        best_loss = 1e8
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            if not self.from_scratch:
                saver.restore(sess, save_path=self.save_path)

            for epoch in range(self.epochs):
                training_batch = zip(range(0, N, self.batch_size),
                                     range(self.batch_size, N + 1, self.batch_size))

                x = shuffle(x)

                print('Epoch: ', epoch)
                batch_best_loss = best_loss
                iter = 0
                for start, end in training_batch:
                    _, loss_val = sess.run([self.optimizer, self.cost], feed_dict={self.x: x[start:end]})

                    if loss_val < batch_best_loss:
                        batch_best_loss = loss_val

                    if iter % 10 == 0:
                        print("Iter: ", iter, "Loss: ", loss_val)

                    iter += 1
                if batch_best_loss < best_loss:
                    best_loss = batch_best_loss
                    saver.save(sess, save_path=self.save_path)


class BetaVAE():
    def __init__(self):

        self.save_path = 'checkpoints/beta_vae'

        self.latent_dim = 100
        self.batch_size = 64
        self.BETA = 1
        self.learning_rate = 1e-4

        self.x = tf.placeholder(tf.float32, [None, 64, 64, 1])

        self.mu_z, self.log_sigma_z = self.encoder(self.x, 'encoder', None)
        self.samples = tf.random_normal([self.batch_size, self.latent_dim], 0, 1, dtype=tf.float32)
        self.sample_z = self.mu_z + tf.exp(self.log_sigma_z)*self.samples
        self.x_hat = self.decoder(self.sample_z, 'decoder', None)

        m = DAE()
        m.batch_size = self.batch_size

        self.jx = m.j_func(self.x)
        self.jx_hat = m.j_func(self.x_hat)

        self.latent_loss = 0.5 * tf.reduce_sum(
           tf.square(self.mu_z) + tf.square(tf.exp(self.log_sigma_z)) - 2*self.log_sigma_z - 1, 1)

        self.reconstruction_loss = tf.pow(self.jx-self.jx_hat, 2)

        self.cost = tf.reduce_mean(self.reconstruction_loss + self.BETA*self.latent_loss)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)


    def encoder(self, x, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.reshape(x, [-1, 64, 64, 1])

            conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=4, strides=2, padding='same',
                                     activation=tf.nn.relu)

            #output_size = output_after_conv(64, kernel_size=4, strides=2, padding='SAME') ## 32

            ## size: [batchsize, 32, 32, 32]
            conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=4, strides=2, padding='same',
                                     activation=tf.nn.relu)

            ## size: [batchsize, 16, 16, 32]
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=4, strides=2, padding='same',
                                     activation=tf.nn.relu)

            ## size: [batchsize, 8, 8, 64]
            conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=4, strides=2, padding='same',
                                     activation=tf.nn.relu)

            ## size: [batchsize, 4, 4, 64]
            flat = tf.reshape(conv4, [-1, 4*4*64])

            dense = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)

            log_sigma = tf.layers.dense(inputs=dense, units=self.latent_dim)
            mu = tf.layers.dense(inputs=dense, units=self.latent_dim)

        return mu, log_sigma

    def decoder(self, sample_z, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            sample_z = tf.reshape(sample_z, [-1, self.latent_dim])

            dense = tf.layers.dense(inputs=sample_z, units=256, activation=tf.nn.relu)

            dense = tf.reshape(dense, [-1, 1, 1, 256])

            deconv1 = tf.layers.conv2d_transpose(inputs=dense, filters=64, kernel_size=4, strides=2, padding='valid',
                                                 activation=tf.nn.relu)

            deconv2 = tf.layers.conv2d_transpose(inputs=deconv1, filters=64, kernel_size=4, strides=2, padding='same',
                                                 activation=tf.nn.relu)

            deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, filters=32, kernel_size=4, strides=2, padding='same',
                                                 activation=tf.nn.relu)

            deconv4 = tf.layers.conv2d_transpose(inputs=deconv3, filters=32, kernel_size=4, strides=2, padding='same',
                                                 activation=tf.nn.relu)

            output_layer = tf.layers.conv2d_transpose(inputs=deconv4, filters=1, kernel_size=4, strides=2, padding='same')

        return output_layer

    def get_shape(self):
        return self.x_hat.get_shape()


if __name__ == '__main__':
    model = BetaVAE()
    print(model.get_shape())
