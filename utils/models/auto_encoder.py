from utils.models.encoder import Encoder
from utils.models.decoder import Decoder
import tensorflow as tf
from utils.distributions import Gaussian, vMF, Cauchy
class AutoEncoder():
    """
    A generic class for an LSTM autoencoder architecture
    Takes a sentence input, encodes it into a latent varaible z and then decodes it into a new sentence
    """
    def __init__(self, encoder_layers, hidden_dim, encoder_bidirectional, decoder_layers, decoder_dim,decode_steps, dtype, attention_decoder, attention_dim, decoder_dropout):
        """
        :param encoder_layers: LSTM layers for the encoder
        :param hidden_dim: The hidden dimension for the encoder and decoder
        :param encoder_bidirectional: Boolean to make encoder a bidirectional LSTM
        :param decoder_layers: LSTM layers for the decoder
        :param decode_steps: Number of words produced by decoder
        :param dtype: Datatype of input
        :param attention_decoder: Boolean to add an attention mechanism to decoder
        :param attention_dim: The size of the attention vector
        :param decoder_dropout: Percentage of time that the decoder input at a given time step is set to 0 (historyless decoder)
        """
        self.encoder = Encoder(encoder_layers, hidden_dim, encoder_bidirectional, dtype)
        #Not variational by default
        self.variational = False
        self.attention_decoder = attention_decoder
        self.decoder_dim = decoder_dim

        self.decoder = Decoder(decoder_layers, self.decoder_dim, decode_steps, dtype, decoder_dropout)

        if attention_decoder:
            self.decoder.set_up_attention(attention_dim)

    def encode(self, input, seq_lengths):
        """
        Encodes an input into a latent variable z
        :param input: Input tensor of dimension [batch, sequence, ..]
        :param seq_lengths: The length of each sentence in the batch without padding - prevents the padding being used to train the network
        :return: output_values: The output at every time step of the LSTM, used for attention
        :return: z: Encoded latent variable of size [1, batch, encoder_output_dim] - unstacked into a list for decoder
        :return: state: final state of architecture to be passed to decoder
        """
        dims = len(tf.unstack(tf.shape(input)))
        perm = [1, 0] + list(range(2, dims))
        #LSTM requires [sequence, batch,..] so first 2 dimensions are swapped
        input_reshape = tf.transpose(input, perm)
        input_unstack = tf.unstack(input_reshape)
        output_values, z, state = self.encoder.forward(input_unstack, seq_lengths)
        z = tf.stack(z)
        z = tf.expand_dims(z,0)

        return output_values, z, state

    def decode(self, output_values, z, state):
        """
        :param output_values: output during the encoding step for attention
        :param z: Latent varaible to decode
        :param: state: final state of encoder architecture
        :return: The output from the decode [batch_size, sequence, decoder_dim]
        """
        if self.variational:
            z = self.sample(z)

        if self.attention_decoder:
            self.decoder.forward_attention(output_values)
        z = tf.unstack(z)
        output = self.decoder.forward(z, state)
        return output

    def add_augment_decode(self, augment_tensor):
        """
        Add augmentation to the docder such that a tensor is concatonated with the input at each time step
        :param augment_tensor: [batch_size, n]
        """
        self.decoder.set_augment_each_time_step(augment_tensor)

    def forward(self, input, seq_lengths):
        """
        Foward pass of the model
        :param input: Input tensor of dimension [batch, sequence, ..]
        :param: The length of each sentence in the batch without padding
        :return: decoded output as well as the latent variable z that the input was encoded into
        """
        output_values, z, state = self.encode(input, seq_lengths)
        output = self.decode(output_values, z, state)
        return output, z


    def sample(self, z):
        self.outputs = []
        z = tf.stack(z)
        for param in self.param_layers:
            # run Z through a linear layer to create a vector for each paramter of the distribution
            self.outputs.append(param(z))

        self.dist.init(self.outputs, tf.shape(z))
        z_sample = self.dist.sample()
        return z_sample

    def add_variational(self, distribution_function_name):
        """
        Turns an autoencoder into a variational autoencodrr
        :param distribution_function: A distribution function that takes in a set of paramters and can be sampled
        :param learned_params_size: The number of paramters are function takes and therefore needs to learn
        :param prior: Our prior distribution which will be used as a regularizer
        """

        self.variational = True
        self.distribution_function = distribution_function_name
        learned_params_size = 2
        if distribution_function_name == "N":
            self.dist = Gaussian()
        elif distribution_function_name == "vMF":
            self.dist = vMF()
        elif distribution_function_name == "cauchy":
            self.dist = Cauchy()
        self.learned_params_size = learned_params_size
        self.param_layers = []

        for i in range(self.learned_params_size):
            layer_name = "distribution_param_" + str(i)
            #linear layer for each paramter we need to learn
            params = tf.layers.Dense(self.decoder_dim, name=layer_name)
            self.param_layers.append(params)

    def kl_cost(self):
        cost, var_param = self.dist.kl_cost()
        return cost, var_param




