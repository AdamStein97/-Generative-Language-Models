import tensorflow as tf
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.custom_gradient import custom_gradient
import scipy

def log_probability_sentence_loss_mask(vocab_probabilities, target_sentence_indexes, loss_mask):
    """
    Calculates the sum of the negative log probability ignoring the pad of the target
    :param vocab_probabilities: Tensor of [batch_size, sequence, vocab_size]
    :param target_sentence_indexes: One hot vector of word [batch_size, sequence, vocab_size]
    :param loss_mask: Used to disregard the pad when calculating loss
    :return: The average loss per sentence
    """
    mask_list = tf.unstack(loss_mask)
    probs = tf.unstack(tf.nn.softmax(vocab_probabilities))
    target_sentence_indexs = tf.unstack(target_sentence_indexes)
    probabilities_list = []
    for i in range(len(probs)):
        mask = mask_list[i]
        p = probs[i]
        idx = target_sentence_indexs[i]
        idx_flattened = tf.range(0, p.shape[0]) * p.shape[1] + tf.dtypes.cast(idx, tf.int32)
        #calcualtes the probability at each time step that the correct word is chosen
        word_probability = tf.gather(tf.reshape(p, [-1]),  # flatten input
                                     idx_flattened)  # use flattened indices

        #masked_probabilities = tf.multiply(word_probability,mask)
        # log probability of each word with a small constant to prevent infinity
        word_log_probability = -tf.log(word_probability + tf.constant(0.000000000001))
        word_log_probability = tf.multiply(word_log_probability, mask)
        probabilities_list.append(word_log_probability)

    probabilities_list = tf.stack(probabilities_list)
    #sum to together to to find the log probability of the sentence
    loss_per_instance = tf.reduce_sum(probabilities_list, axis=1)
    return loss_per_instance


def sigmoid_anneal_schedule(step, final_step_value, scaling_val):
    final_step = tf.constant(float(final_step_value))
    scaling = tf.constant(float(scaling_val)) / final_step
    weight = -final_step / (tf.constant(1.0) + tf.exp(scaling * (step - (final_step / tf.constant(2.0))))) + final_step
    return weight / final_step

@custom_gradient
def bessel_function(v, z):
    """Exponentially scaled modified Bessel function of the first kind."""
    output = array_ops.reshape(script_ops.py_func(
        lambda v, z: scipy.special.ive(v, z, dtype=z.dtype), [v, z], z.dtype),
        ops.convert_to_tensor(array_ops.shape(z), dtype=dtypes.int32))

    def grad(dy):
        return None, dy * (bessel_function(v - 1, z) - bessel_function(v, z) * (v + z) / z)

    return output, grad
