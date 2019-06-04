from utils.models.auto_encoder import AutoEncoder
import tensorflow as tf
import tensorflow_probability as tfp
from utils.loss_functions import log_probability_sentence_loss_mask, sigmoid_anneal_schedule

class Bowman():
    def __init__(self, token_embedder, encoder_layers, encoder_dim, encoder_bidirectional, decoder_layers, attention_decoder, attention_dim, decoder_dropout, augment, dist_name):
        """
        Bowman architecture - VAE for sentences
        :param token_embedder: See Token Embedder class
        :param encoder_layers: Number of layers for the encoder architecture
        :param encoder_dim: Hidden dimension of the encoder
        :param encoder_bidirectional: Boolean stating whether enocder should be bidirectional
        :param decoder_layers: Number of layers for decoder architecture
        :param attention_decoder: Boolean stating whether architecture should have attention
        :param attention_dim: Dimension of any added attention vector
        :param decoder_dropout: decoder_dropout: Percentage of time that the decoder input at a given time step is set to 0 (historyless decoder)
        :param augment: Boolean to specify whether to concat z at each time step of decoder
        """
        self.token_embedder = token_embedder
        self.architecture = AutoEncoder(encoder_layers, encoder_dim, encoder_bidirectional, decoder_layers,encoder_dim * (encoder_bidirectional +1) ,token_embedder.max_sentence_length, tf.float32, attention_decoder, attention_dim, decoder_dropout)

        self.dist_name = dist_name
        self.variational = False
        if self.dist_name in ["vMF", "N", "cauchy"]:
            self.variational = True
            self.architecture.add_variational(self.dist_name)
        self.augment = augment
        self.hold_state = False

    def decode(self, output_values, z):
        output = self.architecture.decode(output_values, z, None)

        vocab_layer = tf.layers.dense(output, self.token_embedder.number_of_words)

        return vocab_layer

    def forward(self, source_embedded, seq_length):
        """
        Forward pass of the network to convert sentence to latent varaible z and then decode back into a vocab
        layer which will  be softmaxed to re-generate sentence
        :param source_embedded: Batch of source sentences that have been embedded by token embedder [batch_size, max_sentence_length, word_dim]
        :return vocab_layer: Output to be softmaxed to generate sentences [batch_size, max_sentence_length, vocab_size]
        :return z: Vector that prototype has been encoded into - useful for debugging
        """

        output_values, z, state = self.architecture.encode(source_embedded, seq_length)


        if self.augment:
            self.architecture.add_augment_decode(z)

        vocab_layer = self.decode(output_values, z)

        return vocab_layer, z

    def forward_encode(self, source_embedded, seq_length):
        output_values, z, state = self.architecture.encode(source_embedded, seq_length)
        if self.hold_state:
            self.state = state
        return tf.stack(z)

    def forward_decode(self, z):

        if self.augment:
            self.architecture.add_augment_decode(z)

        vocab_layer = self.decode(None, z)

        return vocab_layer

    def loss(self, vocab_layer, target_sentence_ohv, step, target_mask):
        """
        Calculates the average loss over the batch
        :param vocab_layer: Output to be softmaxed to generate sentences [batch_size, max_sentence_length, vocab_size]
        :param target_sentence_ohv: One hot vectors labelling which word should be chosen [batch_size, max_sentence_lenth, vocab_size]
        :param step: Float tensor of current training step
        :param target_mask: Float tensor of list of 1s followed by 0s to define which of the target output is pad (0 for pad)
        :return: The average loss across the batch
        """

        kl_weight = sigmoid_anneal_schedule(step, 5000, 10)
        if self.variational:
            kl_value, var_param = self.architecture.kl_cost()
        else:
            kl_value = 0
            var_param = 0
        log_p = log_probability_sentence_loss_mask(vocab_layer, target_sentence_ohv, target_mask)
        loss =  log_p + kl_weight * kl_value
        return tf.reduce_mean(loss), tf.reduce_mean(kl_value), kl_weight, tf.reduce_mean(var_param), tf.reduce_mean(log_p)

