from pe_model.inverse_neural_editor import InverseNeuralEditor
from pe_model.agenda import AgendaMaker
import tensorflow as tf
from utils.models.auto_encoder import AutoEncoder
from utils.loss_functions import log_probability_sentence_loss_mask, sigmoid_anneal_schedule

class Editor():
    def __init__(self, encoder_hidden_dim, norm_eps, norm_max, kappa, agenda_dim, batch_size, token_embedder, encoder_layers, decoder_layers, edit_dim, extra_augment, attention_dim, bidirectional, attention_decoder, decoder_dropout, dist, hold_state):
        """
         The combined prototype and edit architecture model
        :param encoder_hidden_dim: hidden dimension for encoder
        :param norm_eps: hyperparam epsilon for inverse neural editor
        :param norm_max: max value for z_norm - set to 10 in paper
        :param kappa: hyperparam for vMF
        :param decoder_dim: hidden dimension for decoder
        :param agenda_dim: size of agenda vector - see AgendaMaker
        :param batch_size:
        :param token_embedder: See TokenEmbedder for more details
        :param encoder_layers: layers for encoder
        :param decoder_layers: layers for decoder
        :param edit_dim: Size of the edit vector
        :param extra_augment: Boolean of whether to augment decoder with agenda at each timestep with tensor
        :param attention_dim: Size of attention vector
        :param bidirectional: Boolean to whether encoder is bidirectional
        """

        self.edit_encoder = InverseNeuralEditor(tf.constant(norm_max), tf.constant(norm_eps),
                                                kappa, edit_dim, dist)

        self.agenda_maker = AgendaMaker(agenda_dim)
        decode_dim = encoder_hidden_dim * (bidirectional + 1)
        self.autoencoder = AutoEncoder(encoder_layers, encoder_hidden_dim, bidirectional, decoder_layers,decode_dim,
                                       token_embedder.max_sentence_length, tf.float32, attention_decoder, attention_dim,
                                       decoder_dropout)

        self.token_embedder = token_embedder
        self.batch_size = batch_size
        self.agenda_augment = extra_augment
        self.hold_state = hold_state


    def foward_train(self, prototype_embedded, insert_words, delete_words, seq_lengths):
        """
        Completes a forward pass pass of the network to generate a set of probabilities for sentences
        Different from predict() because edit_vector is generated from the target rather than being randomly sampled
        :param prototype_embedded: A tensor of sentences that have been embedded by token embedder[batch_size, max_sentence_length, word_dim]
        :param insert_words: Tensor of list of words inserted  [batch_size, max_sentence_length, word_dim]
        :param delete_words: Tensor of list of words deleted  [batch_size, max_sentence_length, word_dim]
        :return: Output from the decoder architecture before softmax [batch_size, max_sentence_length, vocab_size]
                will later be softmaxed to find the probability that a generated sentence is the target sentence
        """
        edit_vector = self.edit_encoder.gen_edit_vec(insert_words, delete_words)
        vocab_pre_softmax = self.forward(prototype_embedded, edit_vector, seq_lengths)
        return vocab_pre_softmax

    def forward(self, prototype_embedded, edit_vector, seq_lengths):
        """
        Completes a forward pass pass of the network to generate a set of probabilities for sentences
        :param prototype_embedded: A tensor of sentences that have been embedded by token embedder[batch_size, max_sentence_length, word_dim]
        :param edit_vector: A tensor of shape [1, batch_size, edit_dim] which will affect the style of the edit
        :return: Output from the decoder architecture before softmax [batch_size, max_sentence_length, vocab_size]
                will later be softmaxed to find the probability that a generated sentence is the target sentence
        """

        x_output, encoded_prototype, state = self.autoencoder.encode(prototype_embedded, seq_lengths)
        agenda = self.agenda_maker.create_agenda(encoded_prototype, edit_vector)
        if not self.hold_state:
            state = None
        decoder_outputs = self.autoencoder.decode(x_output, encoded_prototype, state)
        vocab_layer = tf.layers.dense(decoder_outputs, self.token_embedder.number_of_words)

        return vocab_layer


    def loss(self, vocab_probs, target_ohv, mask, step):
        log_p = log_probability_sentence_loss_mask(vocab_probs, target_ohv, mask)
        kl_weight = sigmoid_anneal_schedule(step, 5000,10)
        kl_value, var = self.edit_encoder.calc_kl()
        loss = log_p + kl_weight * kl_value
        return tf.reduce_mean(loss), tf.reduce_mean(kl_value), kl_weight, tf.reduce_mean(var), tf.reduce_mean(log_p)



