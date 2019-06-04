import numpy as np
import tensorflow as tf

class TokenEmbedder():
    """
    Class for managing vocab embeddings
    """
    def __init__(self, directory, max_sentence_length):
        self._load_embeddings(directory)
        self.unk_key = "unk"
        self.eos_key = ""
        self.number_of_words = len(self.words)
        self.max_sentence_length = max_sentence_length
        #TODO: Experiment with these embeddings
        unk_embedding = np.array([1.0 for i in range(self.word_dim)])
        #This fucks with models if wrong value (had 0 before) - need to find out why
        eos_embedding = np.array([0.0 for i in range(self.word_dim)])
        self._add_extra_tokens(self.unk_key, unk_embedding)
        self._add_extra_tokens(self.eos_key, eos_embedding)

        self.vocab_embedding_array = np.array(self.embeddings) #array of all the embeddings

    def _add_extra_tokens(self, word, embedding):
        """
        Add an extra token and embedding to TokenEmbedder e.g. an unknown token or an eos token
        :param word: Word token to be added
        :param embedding: Embedding associated with word
        """
        self.words_to_vocab_index[word] = self.last_word_index
        self.word_to_embedding[word] = embedding
        self.words.append(word)
        self.last_word_index += 1
        self.embeddings.append(embedding)
        self.number_of_words += 1


    def _load_embeddings(self, directory):
        """
        Load embeddings from a file
        :param directory: String directory of where to load embeddings from
        """
        print("Loading Embeddings")
        f = open(directory, 'r', encoding="utf8")
        self.word_to_embedding = {}
        self.embeddings = []
        self.words = []
        self.words_to_vocab_index = {}
        self.last_word_index = 0
        for line in f:
            splitLine = line.split()
            word = splitLine[0].lower()
            self.words.append(word)
            e = np.array([float(val) for val in splitLine[1:]])
            self.embeddings.append(e)
            self.word_to_embedding[word] = e
            self.word_dim = len(e)
            self.words_to_vocab_index[word] = self.last_word_index
            self.last_word_index += 1
        f.close()

    def _replace_unkown_words(self, sentence):
        """
        Replaces any words in a sentence not in our dictionary with the unknown token
        :param sentence: Input sentence as list of words
        :return: Output sentence with words replaces as list of words

        E.g
        Input: ["I", "like", "Hawaiian","pizza"]
        Output: ["I", "like", "unk","pizza"]
        """
        new_sentence = []
        for word in sentence:
            if word in self.word_to_embedding.keys():
                new_sentence.append(word)
            else:
                new_sentence.append(self.unk_key)
        return new_sentence

    def _pad_sentence(self, sentence):
        """
        Pads a sentence out using the eos token so that they are the max_sentence length
        :param sentence: Input sentence as list of words
        :return: Output padded sentence

        E.g.
        max_sentence_length = 8
        Input: ["I", "love", "ice", "cream"]
        Output:["I", "love", "ice", "cream", "eos", "eos", "eos", "eos"]
        """
        pad_size = self.max_sentence_length - len(sentence)
        for i in range(pad_size):
            sentence.append(self.eos_key)

        return sentence

    def embed_sentence(self, sentence):
        """
        Creates an embedded sentence of size [max_sentence_length, word_dim]from a list of words
        Replaces unknown words and pads sentences before finding embeddings from dictionary
        :param sentence: List of string words
        :return: List of length max_sentence_length containing embeddings vectors of size word_dim
        """
        sentence = self._replace_unkown_words(sentence)
        sentence = self._pad_sentence(sentence)
        embed_sentence = [self.word_to_embedding[word] for word in sentence]
        return embed_sentence

    def make_one_hot_vector(self, sentence):
        """
        Creates a one hot vector over the vocabulary for a given sentence, sentences are padded
        with unkown words removed
        :param sentence: List of string words
        :return: List of length max_sentence_length containing one hot vector of size number_of_words

        E.g.
        vocabulary = ["like","I", "pizza", "pasta", "eos", "unk"]
        max_sentence_length = 5
        Input = ["I", "like", "pasta", "sauce"]
        Output = [[0,1,0,0,0,0], [1,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,1], [0,0,0,0,1,0]]
        """
        sentence = self._replace_unkown_words(sentence)
        sentence = self._pad_sentence(sentence)
        word_indexes = [self.words_to_vocab_index[word] for word in sentence]
        one_hot_vectors = np.zeros((self.max_sentence_length, self.number_of_words))
        one_hot_vectors[np.arange(self.max_sentence_length), word_indexes] = 1
        return one_hot_vectors

    def gen_indexes_from_sentence(self, sentence):
        sentence = self._replace_unkown_words(sentence)
        sentence = self._pad_sentence(sentence)
        word_indexes = [self.words_to_vocab_index[word] for word in sentence]
        return word_indexes

    def add_unk_plaintext(self, sentence):
        for i, word in enumerate(sentence):
            if word not in self.word_to_embedding.keys():
                sentence[i] = "unk"
        return sentence

    def embed_word(self, word):
        #find embedding for a given word
        return self.word_to_embedding[word]

    def get_broadcasted_vocab_embeddings_tensor(self, batch_size):
        #creates a tensor of word embeddings - used in the decoder
        tensor_embedding = tf.constant(self.vocab_embedding_array.transpose(), dtype= tf.float32)
        return tf.broadcast_to(tensor_embedding, [batch_size, self.word_dim, self.number_of_words])

    def gen_sentence_from_indexes(self, indexes):
        """
        Generate a list of words based on the indexes - useful for decoding into sentence from tf.argmax
        :param indexes: Numerical indexes of word in the vocabulary
        :return: A list of words

        E.g.
        vocabulary = ["like","I", "pizza", "pasta", "eos", "unk"]
        max_sentence_length = 5
        Input = [1,0,2,4,4]
        Output = ["I", "like", "pizza", "eos", "eos"]
        """
        return [self.words[i] for i in indexes]

    def gen_sentence_from_indexes_no_repeat(self, indexes):
        """
        Generate a list of words based on the indexes - useful for decoding into sentence from tf.argmax
        :param indexes: Numerical indexes of word in the vocabulary
        :return: A list of words

        E.g.
        vocabulary = ["like","I", "pizza", "pasta", "eos", "unk"]
        max_sentence_length = 5
        Input = [1,0,2,4,4]
        Output = ["I", "like", "pizza", "eos", "eos"]
        """

        sent = []
        last_index = -1
        for i in indexes:
            if i != last_index:
                last_index = i
                sent.append(self.words[i])
        return sent
