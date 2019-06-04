import tensorflow as tf
class AgendaMaker():
    def __init__(self, agenda_dim):
        self.agenda_dim = agenda_dim


    def create_agenda(self, encoded_prototype, edit_embedding):
        """
        An agenda vector to combine the source emebedding and edit embedding
        :param encoded_prototype: Encoding of the prototype sentence (output from the encoder) [1, batch_size, encoder_output]
        :param edit_embedding: The edit produced by the edit encoder [1, batch_size, edit_embedding]
        :return: tensor of the agenda [1, batch_size, agenda_dim]
        """
        encoded_prototype = tf.unstack(encoded_prototype)[0]
        input = tf.concat([encoded_prototype, edit_embedding], -1)

        agenda = tf.layers.dense(input, self.agenda_dim)

        return tf.expand_dims(agenda,0)
