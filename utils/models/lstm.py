import tensorflow as tf
class LSTM():
    def __init__(self, layers, hidden_dim, attention, attention_dim,  bidirectional, dtype, name):
        """
        LSTM architecture
        :param layers: Number of layers for the LSTM
        :param hidden_dim: Hidden dimension of LSTM
        :param bidirectional: Boolean stating whether bidirectional LSTM
        :param dtype: Datatype for the LSTM
        :param name: Name for the cells in the tensorflow scope
        """
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.dtype = dtype

        self.cell_fw = tf.nn.rnn_cell.MultiRNNCell([self._lstm_cell(name + str(i)) for i in range(layers)],
                                                   state_is_tuple=True)

        if bidirectional:
            self.cell_bw = tf.nn.rnn_cell.MultiRNNCell([self._lstm_cell(name + str(i)) for i in range(layers)],
                                                       state_is_tuple=True)

    def _lstm_cell(self, name):
        #TODO: Unsure on which activation here
        lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim, name=name)
        return lstm_cell

    def start_state(self, batch_size):
        return self.cell_fw.zero_state(batch_size, dtype=self.dtype)

    def forward(self, unstacked_input, cell_state, batch_seq_length):
        """
        Forward pass through LSTM
        :param unstacked_input: An unstacked sequence [sequence, batch_size, ...]
        :param batch_seq_length: A list of sentences lengths without pad for the batch to prevent padding token being used to train weights
        :param cell_state: The starting state of the LSTM
        :return output: The output state at each time step
        :return state: The final output state
        """
        if self.bidirectional:
            state_fw, state_bw = cell_state
            (output, output_state_fw, output_state_bw) = tf.nn.static_bidirectional_rnn(cell_fw=self.cell_fw, cell_bw=self.cell_bw,
                                                                                        inputs=unstacked_input, initial_state_fw=state_fw, initial_state_bw=state_bw,
                                                                                        sequence_length=batch_seq_length, dtype=tf.float32
                                                                                        )
            state = [output_state_fw, output_state_bw]

        else:
            (output, state) = tf.contrib.rnn.static_rnn(cell=self.cell_fw, inputs=unstacked_input, initial_state=cell_state, sequence_length=batch_seq_length, dtype=tf.float32)

        return (output, state)