import tensorflow as tf
import random

def greedy_decode(batch_size, y_prime):
    """
    Greed decodes a batch of softmax outputs into indexes from the vocabulary
    :param batch_size: Size of the batch
    :param y_prime: Output tensor from architecture of size [batch_size, max_sentence_length, vocab_size]
    :return: A list of indexes from the vocab
    """
    vocab_probabilities = tf.nn.softmax(y_prime)
    sentence_indexes = []
    for i in range(batch_size):
        indexes = tf.argmax(vocab_probabilities[i], axis=1)
        sentence_indexes.append(indexes)

    return sentence_indexes

def calculate_accracy(target_sentences, predicted_sentences):
    """
    Calculates the accuracy of a predicted sentence based on correct words in correct places
    :param target_sentences: List of list of target words
    :param predicted_sentences: List of list of decoded words
    :return: Accuracy of predicted words

    E.g.
    Input target_sentences: [["I, "love", "pasta", "sauce"], ["I, "like", "pizza"]]
    Input predicted_sentences: [["I, "love", "pasta"], ["I, "love","pizza"]]
    Output: 71.4 (3 s.f.)
    """
    total_acc = 0
    for i in range(len(target_sentences)):
        target_words = target_sentences[i].split(" ")
        predicted_words = predicted_sentences[i]
        correct = 0
        incorrect = 0
        min_size = min(len(target_words), len(predicted_words))
        for j in range(min_size):
            if target_words[j] == predicted_words[j]:
                correct += 1
            else:
                incorrect += 1
        incorrect += len(target_words) - min_size
        total_acc += correct /(correct + incorrect)

    total_acc = total_acc / len(target_sentences) * 100
    return total_acc

def word_droput(dropout_perc, source_sentences, unknown_embedding, eos_embedding):
    """
    Replaces a percentage of words in a sentence with the unknown token to work as a regularizer
    :param dropout_perc: Float between 0 and 1 percentage of words to remove
    :param source_sentences: Batch of source sentences that have been embedded
    :param unknown_embedding: The embedding for the unknown token
    :param eos_embedding: The embedding for the eos token (what is used to pad the sentence)
    :return: A new batch of embedded sentences who have had the uknown word added

    E.g.
    batch_size=1
    Input dropout_perc: 0.25
    Input source_sentences: [[2,4,2,1,6], [1,2,5,2,1], [3,2,5,3,1], [2,4,2,1,6], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
    Input unknown_embedding: [1,1,1,1,1]
    Input eos_embedding: [0,0,0,0,0]

    Output: [[2,4,2,1,6], [1,1,1,1,1], [3,2,5,3,1], [2,4,2,1,6], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
    """
    new_sentences = []
    for sentence in source_sentences:
        i = 0
        while (sentence[i] != eos_embedding).all():
            i += 1
        sentence_length = i
        word_to_change = int(sentence_length * dropout_perc)
        indexes = random.sample(range(0, sentence_length), word_to_change)
        for index in indexes:
            sentence[index] = unknown_embedding
        new_sentences.append(sentence)

    return new_sentences

def unstacked_op(x, y, op):
    # TODO: Figure out how to do this properly
    a = tf.unstack(x)
    b = tf.unstack(y)
    r = []
    for i in range(len(a)):
        r.append(op(a[i],b[i]))
    ret = tf.stack(r)
    return ret

