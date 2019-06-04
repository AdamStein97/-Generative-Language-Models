import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.custom_gradient import custom_gradient
import math
from scipy import special

class Gaussian():
    def init(self, params, z_shape):
        self.mu = params[0]
        self.var = tf.abs(params[1]) + tf.constant(0.00000001)
        self.z_shape = tf.shape(self.mu)

    def sample(self):
        epsilon = tfp.distributions.Normal(0, 1)
        epsilon_samples = epsilon.sample([self.z_shape[-2], self.z_shape[-1]])
        # reparam trick
        z_sample = self.mu + tf.pow(self.var, tf.constant(0.5)) * epsilon_samples
        return z_sample

    def kl_cost(self):
        """
        Calculates the KL divergence for each sentence in the batch if our prior is N(0,1)
        :return: The KL divergence for each batch element
        """
        kl = -1 - tf.log(self.var) + self.var + tf.pow(self.mu, 2)
        return 0.5 * tf.reduce_sum(kl, axis=-1), self.var


class Cauchy():
    def init(self, params, z_shape):
        self.x0 = params[0]
        self.gamma = tf.abs(params[1])
        self.z_shape = z_shape

    def sample(self):
        uniform =  tfp.distributions.Normal(0, 1)
        samples = uniform.sample([self.z_shape[-2], self.z_shape[-1]])
        z_sample = self.x0 + self.gamma * tf.tan(tf.constant(math.pi) * (samples - tf.constant(0.5)))
        return z_sample

    def kl_cost(self):
        v = tf.log((tf.sqrt(tf.abs(tf.constant(1.0) + tf.squeeze(self.x0))) + tf.squeeze(self.gamma)) / tf.squeeze(self.gamma))
        kl = tf.reduce_sum(v, -1)

        return kl, self.gamma

class vMF():
    def __init__(self):
        self.l = tf.layers.Dense(1, name="q_model_kappa_layer")

    def init(self, params, z_shape):
        self.f = tf.squeeze(params[0])
        self.m = tf.cast(z_shape[-1], tf.float32)
        self.k = tf.abs(self.l.apply(tf.squeeze(params[1]))) * self.m + self.m/tf.constant(8.0)
        self.m_int = z_shape[-1]

    def sample(self):
        k_list = tf.unstack(self.k)
        f_list = tf.unstack(self.f)
        self.batch_size = len(f_list)
        results = []
        for i in range(self.batch_size):
            mu = f_list[i] / k_list[i]
            w = self._sample_weight(k_list[i])
            v = tfp.distributions.Normal(0,1).sample(self.m_int)
            v = v / tf.norm(v)
            r = tf.sqrt(tf.constant(1.0) - tf.pow(w,2)) * v +w*mu
            results.append(r)

        return tf.expand_dims(tf.stack(results),0)


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

    def _sample_weight(self, kappa):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = self.m - tf.constant(1.0)  # since S^{n-1}
        b = dim / (tf.sqrt(tf.constant(4.0)  * tf.pow(kappa, tf.constant(2.0)) + tf.pow(dim, tf.constant(2.0))) + tf.constant(2.0) * kappa)  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
        x = (tf.constant(1.0) - b) / (tf.constant(1.0) + b)
        c = kappa * x + dim * tf.log(tf.constant(1.0)- tf.pow(x, tf.constant(2.0)))  # dim * (kdiv *x + np.log(1-x**2))
        w = self._sample_loop(dim, kappa, b, x, c)
        return w

    def kl_cost(self):
        return kl_vMF(self.m, self.k), self.k

def summing_loop(max):
    def cond(x, i):
        b = tf.less_equal(i, max)
        return b

    def body(x, i):
        x = x + tf.log(i)
        i = i + tf.constant(1.0, tf.float64)
        return [x, i]

    i = tf.constant(1.0, tf.float64)
    x = tf.constant(0.0, tf.float64)
    return_values = tf.while_loop(cond, body, [x, i])
    return return_values[0]

@custom_gradient
def kl_vMF(m, k):
    m = tf.cast(m, tf.float64)
    k = tf.cast(k, tf.float64)
    half_m = m / tf.constant(2.0, tf.float64)
    half_m_minus_one = half_m - tf.constant(1.0, tf.float64)
    half_m_plus_one = half_m + tf.constant(1.0, tf.float64)
    term1 = k * bessel_function(half_m, k) / bessel_function(half_m_minus_one, k)
    term2 = (m/tf.constant(2.0, tf.float64) - tf.constant(1.0, tf.float64)) * tf.log(k) - m/tf.constant(2.0, tf.float64) * tf.log(tf.constant(2*math.pi, tf.float64)) - tf.log(bessel_function(half_m_minus_one, k))
    term3 = m/tf.constant(2.0, tf.float64) * tf.log(tf.constant(math.pi, tf.float64)) + tf.log(tf.constant(2.0, tf.float64))
    term4 = summing_loop(half_m)

    kl = term1 + term2 - tf.pow(term3 - term4, tf.constant(-1.0, tf.float64))

    def grad(dy):
        k_squeeze = tf.squeeze(k)
        grad_term1 = bessel_function(half_m_plus_one, k_squeeze) / bessel_function(half_m_minus_one, k_squeeze)
        k_unstack = tf.unstack(k_squeeze)
        r = []
        for k_singular in k_unstack:
            x = (bessel_function(half_m, k_singular) * (bessel_function(half_m - tf.constant(2.0, tf.float64), k_singular)  + bessel_function(half_m,k_singular)))
            y = tf.pow(bessel_function(half_m_minus_one, k_singular), tf.constant(2.0, tf.float64))
            val = tf.cond(y < tf.constant(1e-300, tf.float64),lambda: tf.constant(1.0, tf.float64), lambda: tf.divide(x, y))
            r.append(val)
        grad_term2 = tf.stack(r)

        g = tf.cast(tf.squeeze(dy), tf.float64) * (tf.constant(0.5, tf.float64) * k_squeeze * (grad_term1 - grad_term2 + tf.constant(1.0, tf.float64)))

        return None, tf.cast(tf.expand_dims(g,1), tf.float32)

    return tf.cast(kl, tf.float32), grad


def bessel_function(v, z):
    """Exponentially scaled modified Bessel function of the first kind."""
    output = array_ops.reshape(script_ops.py_func(
        lambda v, z: special.ive(v, z, dtype=z.dtype), [v, z], z.dtype),
        ops.convert_to_tensor(array_ops.shape(z), dtype=dtypes.int32))


    return output
