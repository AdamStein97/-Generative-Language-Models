import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy as sc
import scipy.stats
from utils.loss_functions import bessel_function
import math
from utils.distributions import Gaussian, vMF, Cauchy
from utils.utils import unstacked_op

class InverseNeuralEditor():
    def __init__(self, norm_max, epsilson, kappa_init, edit_dim, dist_name):
        """
        Encodes inserted and deleted words into a vector
        :param norm_max: Max value for f_norm - 10 in paper
        :param epsilson: Range for uniform distribution over f_norm
        :param kappa_init: vMF parameter
        :param edit_dim: The size of the generated edit vector
        """
        self.norm_max = norm_max
        self.eps = epsilson
        self.gaussian = "N"
        self.vMF = "vMF"
        self.vMFc = "vMF Constant"
        self.cauchy = "Cauchy"
        self.none = "None"
        self.edit_dim = edit_dim


        self.dist_name = dist_name
        if self.dist_name == self.vMFc:
            self.k = tf.constant(kappa_init)
        if self.dist_name == self.vMF:
            self.dist = vMF()
            self.param_layer = tf.layers.Dense(self.edit_dim / 2 , name="q_model_dist_param")
        if self.dist_name == self.gaussian:
            self.dist = Gaussian()
            self.param_layer = tf.layers.Dense(self.edit_dim / 2, name="q_model_dist_param")
        if self.dist_name == self.cauchy:
            self.dist = Cauchy()
            self.param_layer = tf.layers.Dense(self.edit_dim / 2, name="q_model_dist_param")


        #used the learn transform from the embedding steps to create an edit vector
        self.transform = tf.layers.Dense(self.edit_dim / 2 , name="q_model_f_transform")
    def gen_edit_vec(self, insert_words, delete_words):
        """
        Generates an edit vector z based on inserted and deleted words
        :param insert_words: Tensor of set of insert words
        :param delete_words: Tensor of set of deleted words
        :return: An edit vector tensor of size [1, batch_size, edit_dim] which specifies the type of edit we want to perform
        """

        insert_transformed = tf.reduce_sum(self.transform.apply(insert_words), 1)
        delete_transformed = tf.reduce_sum(self.transform.apply(delete_words),1)
        f = tf.concat([insert_transformed, delete_transformed], -1)

        f_norm = tf.norm(f, axis=1)
        f_norm = tf.minimum(f_norm, tf.subtract(self.norm_max, self.eps))
        z_norm = tfp.distributions.Uniform(f_norm, tf.add(f_norm, self.eps))
        z_norm_sample = tf.squeeze(z_norm.sample([1]))

        f_dir = unstacked_op(f, f_norm, tf.divide)

        if self.dist_name == self.vMFc:
            z_dir_sample = self.sample_vMF(f, self.k)
        elif self.dist_name == self.none:
            return f
        else:
            dist_param = tf.layers.dense(f, self.edit_dim)

            d_norm = tf.norm(dist_param, axis=1)
            dist_param = unstacked_op(dist_param, d_norm, tf.divide)
            if self.dist_name == self.gaussian:
                params = [f_dir, dist_param]
                self.dist.init(params, tf.shape(f))
                z = self.dist.sample()
                return tf.squeeze(z)
            else:
                params = [f_dir, dist_param]
                self.dist.init(params, tf.shape(f_dir))
                z_dir_sample = self.dist.sample()
                z_dir_sample = tf.squeeze(z_dir_sample)

        z = unstacked_op(z_norm_sample, z_dir_sample, tf.multiply)
        return z


    def _add_norm_noise(self, munorm, eps):
        """
        Clip the value of munorm to norm_max - epsilon and add some noise
        :param munorm: Normalised mu parameter taken from the generated edit vector
        :param eps: Range for uniform distribution over f_norm
        :return: Clipped mu_norm with small random noise added

        E.g:
        norm_max = 10
        eps = 0.1
        Input munorm: tensor filled with [8,11,6,12]
        r_val = 0.5
        Output Clip: [8,9.9, 6, 9.9]
        Output: [8.5, 10.4, 6.5, 10.4]
        """
        r_val = tf.random.uniform([1]) #samples 1 item from [0,1)
        random_val = tf.broadcast_to(r_val, [self.edit_dim]) #expands random value to be size of edit vector
        random_val_scaled = tf.multiply(random_val, eps)
        #Clip munorm to be less than norm_max - eps
        clip = tf.clip_by_value(munorm, clip_value_min=0, clip_value_max=tf.subtract(self.norm_max, self.eps))
        #Add the noise vector to clipped munorm
        return tf.add(clip, random_val_scaled)


    def sample_vMF(self, f, kappa):
        """
        Samples vMF distribution to generate edit_vectors with noise
        :param f: Our edit vectors generated using inserted and deletd words [batch_size, edit_dim]
        :param kappa: Hyperparam of the distribution - fixed rather than learnt to manage KL term
        :return: Edit vector batch with noise applied [batch_size, edit_dim]
        """
        batch_size, _ = f.get_shape().as_list()
        result_list = []
        list_f = tf.unstack(f)
        for f in list_f:
            w_32 = self._sample_weight(kappa, self.edit_dim)
            w_array = tf.broadcast_to(w_32, [self.edit_dim])
            f_norm = tf.broadcast_to(tf.norm(f) , [self.edit_dim])
            munoise = self._add_norm_noise(f_norm, self.eps)
            # sample a point v on the unit sphere that's orthogonal to mu
            v = self._sample_orthonormal_to(tf.divide(f, f_norm), self.edit_dim)

            # compute new point
            scale_factor = tf.sqrt(tf.subtract(tf.ones(self.edit_dim),  tf.pow(w_array, 2)))
            orth_term = tf.multiply(v, scale_factor)
            muscale = tf.divide(tf.multiply(f , w_array), f_norm)
            sampled_vec = tf.multiply((tf.add(orth_term, muscale)),munoise)

            result_list.append(sampled_vec)

        return tf.stack(result_list)

    def _sample_loop(self, dim, kappa, b, x, c):
        def cond(w, kappa, dim, x, c, u):
            bool = tf.less(kappa * w + dim * tf.log(tf.constant(1.0) - x * w) - c, tf.log(u))
            return tf.reshape(bool, [])

        def body(w, kappa, dim, x, c, u):
            beta = tfp.distributions.Beta(dim / tf.constant(2.0), dim / tf.constant(2.0))
            z = beta.sample([1])
            w = (tf.constant(1.0) - (tf.constant(1.0) + b) * z) / (tf.constant(1.0) - (tf.constant(1.0) - b) * z)
            u = tfp.distributions.Uniform(0, 1).sample([1])
            return [w, kappa, dim, x, c, u]

        beta = tfp.distributions.Beta(dim / tf.constant(2.0), dim / tf.constant(2.0))
        z = beta.sample([1])
        w = (tf.constant(1.0) - (tf.constant(1.0) + b) * z) / (tf.constant(1.0) - (tf.constant(1.0) - b) * z)
        u = tfp.distributions.Uniform(0, 1).sample([1])
        return_values = tf.while_loop(cond, body, [w, kappa, dim, x, c, u], back_prop=False)
        return return_values[0]

    def _sample_weight(self, kappa, dim):
        dim = tf.constant(dim - 1.0, tf.float32)  # since S^{n-1}
        b = dim / (tf.sqrt(tf.constant(4.0)  * tf.pow(kappa, tf.constant(2.0)) + tf.pow(dim, tf.constant(2.0))) + tf.constant(2.0) * kappa)  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
        x = (tf.constant(1.0) - b) / (tf.constant(1.0) + b)
        c = kappa * x + dim * tf.log(tf.constant(1.0)- tf.pow(x, tf.constant(2.0)))  # dim * (kdiv *x + np.log(1-x**2))
        w = self._sample_loop(dim, kappa, b, x, c)
        return w

    def _sample_orthonormal_to(self, mu, dim):

        v = tf.random.normal([dim])
        dot_prod = tf.tensordot(mu, v, axes=1)
        rescale_value = tf.divide(dot_prod, tf.norm(mu))
        proj_mu_v = tf.multiply(mu, tf.broadcast_to(rescale_value,[dim]))
        ortho = tf.subtract(v, proj_mu_v)
        ortho_norm = tf.norm(ortho)
        return tf.divide(ortho, tf.broadcast_to(ortho_norm, [dim]))

    def calc_kl(self):
        if self.dist_name == self.vMFc or self.dist_name == self.none:
            return 0,0
        else:
            kl, var = self.dist.kl_cost()
            return kl, var



