import tensorflow as tf
from utils.models.lstm import LSTM
class Encoder():
    def __init__(self, layers, hidden_dim,  bidirectional, dtype):
        """
        Encodes sequence data into a vector
        :param layers: Number of layers for the encoder
        :param hidden_dim: Hidden dimension of the encoder
        :param bidirectional: Boolean of whether encoder is bidirectional LSTM
        :param dtype: Datatype of the LSTM
        """
        self.bidirectional = bidirectional
        self.model = LSTM(layers, hidden_dim, False, 0,  bidirectional, dtype, "enc")

    def forward(self, unstacked_input, seq_lengths):
        """
        Forward pass through the encoder
        :param unstacked_input: List of input tensors of size [batch_size, ..]
        :param seq_lengths: List of lengths of each sentence without the pad to prevent network for training with pad tokens
        :return outputs: Outputs from each encoding time step
        :return z: final state after the encoding
        :return state: final state of architecture to be passed to decoder
        """
        start_state = self.model.start_state(tf.shape(unstacked_input[0])[0])
        if self.bidirectional:
            state_fw = start_state
            state_bw = start_state
            (outputs, [output_state_fw, output_state_bw]) = self.model.forward(unstacked_input, [state_fw, state_bw], seq_lengths)
            fw_final_c, fw_final_h = output_state_fw[-1]
            bw_final_c, bw_final_h = output_state_bw[-1]
            h = tf.concat([fw_final_h, bw_final_h], -1)
            z = tf.unstack(h)
            state = []
            for i in range(len(output_state_fw)):
                layer_state = []
                for j in range(len(output_state_fw[0])):
                    val = tf.concat([output_state_fw[i][j], output_state_bw[i][j]],-1)
                    layer_state.append(val)
                state.append(layer_state)
        else:
            (outputs, state) = self.model.forward(unstacked_input, start_state, seq_lengths)
            final_c, final_h = state[-1]
            z = tf.unstack(final_h)
        return outputs, z, state
