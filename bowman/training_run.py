import tensorflow as tf
import random
import numpy as np
from utils.token_embedder import TokenEmbedder
import bowman.utils_bowman as utils_bowman
from bowman.bowman_autoencoder import Bowman
from utils.utils import greedy_decode
import time
from utils.training_utils import interpolate_sentences, write_all_sentences, print_and_write_training_info,calc_accuracy, calc_bleu, display_sentences, write_scaler_summaries, display_interact_sentences, levenshtein, jaccard

class TrainingRun():
    def __init__(self, config, debug, file_name):
        self.config = config
        self.out_file_name = file_name
        self.output_file = open(self.config["log_dir"] + self.out_file_name, "w")

        if debug:
            tf.set_random_seed(1)
            random.seed(1)
            np.random.seed(1)

        self.token_embedder = TokenEmbedder(self.config["word_embedding_directory"], self.config['max_sentence_length'])
        self.train_set = utils_bowman.read_sentences("train.tsv", self.config["dataset_directroy"], self.token_embedder)
        self.validation_set = utils_bowman.read_sentences("valid.tsv", self.config["dataset_directroy"], self.token_embedder)
        self.batch_size = config["batch_size"]

        self._init_input_tensors()
        self._build_architecture()
        self._init_post_run_tensors()
        self._init_optimizers()
        self._init_tensorboard_summaries()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def _init_input_tensors(self):
        self.source = tf.placeholder(tf.float32, [self.config["batch_size"], self.config['max_sentence_length'],
                                                  self.config["word_dim"]], name="source")
        self.target_ohv = tf.placeholder(tf.int32, [self.config["batch_size"], self.config['max_sentence_length']], name="target")
        self.step = tf.placeholder(tf.float32, name="step")
        self.seq_lengths = tf.placeholder(tf.int32, [self.config["batch_size"]], name="seq_l")
        self.target_masks = tf.placeholder(tf.float32, [self.config["batch_size"], self.config['max_sentence_length']], name="mask")
        self.z = tf.placeholder(tf.float32, [1, self.config["batch_size"], self.config["hidden_dim"] * (
                self.config["encoder_bidirectional"] + 1)], name="z")


    def _build_architecture(self):
        self.bowman = Bowman(self.token_embedder, self.config["encoder_layers"],self.config["hidden_dim"],
                                            self.config["encoder_bidirectional"],self.config["decoder_layers"],
                                            self.config["attention"],self.config["attention_dim"], self.config["decoder_dropout_perc"],
                             self.config["augment"], self.config["dist_name"])


    def _init_optimizers(self):
        self.optimiser = tf.train.AdamOptimizer(self.config["learning_rate"]).minimize(self.loss)


    def _init_post_run_tensors(self):
        self.z_encoding = self.bowman.forward_encode(self.source, self.seq_lengths)
        self.y_prime = self.bowman.forward_decode(self.z)
        self.greedy_decode_sentence = greedy_decode(self.batch_size, self.y_prime)
        self.loss, self.kl_val, self.kl_weight, self.var_param, self.logp = self.bowman.loss(self.y_prime, self.target_ohv, self.step, self.target_masks)
        #self.decoded_random_sentence = greedy_decode(self.batch_size,self.bowman.gen_sentence_z(self.random_z))

    def _init_tensorboard_summaries(self):
        kl_value_summary = tf.summary.scalar('KL_Value', self.kl_val)
        kl_weight_summary = tf.summary.scalar('KL_Weight', self.kl_weight)
        loss_summary = tf.summary.scalar('Loss', self.loss)
        var_param = tf.summary.scalar('Var', self.var_param)
        reconstruction_loss = tf.summary.scalar('Reconstruction loss', self.logp)
        self.run_summary = tf.summary.merge([loss_summary, kl_value_summary, kl_weight_summary, var_param, reconstruction_loss])

    def test_model(self, num_batches, file_name):
        model_checkpoint = self.config["model_save_directory"] + self.config["model_restore_name"]
        self.output_file = open(self.config["log_dir"] + file_name, "w")
        with tf.Session() as sess:
            self.saver.restore(sess, model_checkpoint)
            test_loss_total = 0
            test_reconstruct_total = 0
            test_bleu_score_total = 0
            test_word_acc_total = 0
            test_jaccard_total = 0
            test_levenshtein_total = 0
            eval_time = time.time()
            for i in range(num_batches):

                test_batch = utils_bowman.create_batch(self.validation_set, self.config["batch_size"],
                                                       self.token_embedder, self.config["dropout_perc"],  i)

                z_sample = sess.run(self.z_encoding, feed_dict={self.source: test_batch["sentence_embed"],
                                                                self.seq_lengths: test_batch["seq_lengths"],
                                                                })
                test_loss, test_reconstruction_loss, sentence_test = sess.run(
                    [self.loss, self.logp, self.greedy_decode_sentence],
                    feed_dict={
                        self.target_ohv: test_batch[
                            'target_ohv'],
                        self.step: float(10000),
                        self.z: z_sample,
                        self.target_masks: test_batch["mask"]})

                word_acc_test = calc_accuracy(test_batch, sentence_test, self.token_embedder, "sentence")
                write_all_sentences(test_batch, sentence_test, self.token_embedder, self.output_file)
                bleu_test = calc_bleu(test_batch, sentence_test, self.token_embedder, "sentence")

                jac_d = jaccard(test_batch["sentence"], sentence_test, self.token_embedder)
                l_d = levenshtein(test_batch["sentence"], sentence_test, self.token_embedder)

                test_loss_total += test_loss
                test_reconstruct_total += test_reconstruction_loss
                test_word_acc_total += word_acc_test
                test_bleu_score_total += bleu_test
                test_jaccard_total += jac_d
                test_levenshtein_total += l_d

            print("Evaluation Time: " + str(time.time() - eval_time))
            average_test_loss = test_loss_total / num_batches
            average_reconstruction_loss = test_reconstruct_total / num_batches
            test_perplexity = 2 ** (average_reconstruction_loss / np.log(2))
            average_acc_test = test_word_acc_total / num_batches
            average_bleu_test = test_bleu_score_total / num_batches
            average_jac = test_jaccard_total / num_batches
            average_lev = test_levenshtein_total / num_batches

            print("Testing Loss: " + str(average_test_loss))
            print("Reconstruction Test Loss: " + str(average_reconstruction_loss))
            print("Testing Perplexity: " + str(test_perplexity))
            print("Testing Word Accuracy: " + str(average_acc_test))
            print("Testing BLEU Score: " + str(average_bleu_test))
            print("Jac Dis: " + str(average_jac))
            print("Lev Dist: " + str(average_lev))

            self.output_file.write("Evaluation Time: " + str(time.time() - eval_time) + "\n")
            self.output_file.write("Testing Loss: " + str(average_test_loss) + "\n")
            self.output_file.write("Reconstruction Test Loss: " + str(average_reconstruction_loss) + "\n")
            self.output_file.write("Testing Perplexity: " + str(test_perplexity) + "\n")
            self.output_file.write("Testing Word Accuracy: " + str(average_acc_test) + "\n")
            self.output_file.write("Testing BLEU Score: " + str(average_bleu_test) + "\n")
            self.output_file.write("Jac Dis: " + str(average_jac) + "\n")
            self.output_file.write("Lev Dist: " + str(average_lev) + "\n")

            self.output_file.close()

    def interpolate_sentences(self, num_batches, file_name):
        model_checkpoint = self.config["model_save_directory"] + self.config["model_restore_name"]
        self.output_file = open(self.config["log_dir"] + file_name, "w")
        with tf.Session() as sess:
            self.saver.restore(sess, model_checkpoint)

            for i in range(num_batches):
                test_batch1 = utils_bowman.create_batch(self.validation_set, self.config["batch_size"],
                                                       self.token_embedder, self.config["dropout_perc"], i)

                z_sample = sess.run(self.z_encoding, feed_dict={self.source: test_batch1["sentence_embed"],
                                                                self.seq_lengths: test_batch1["seq_lengths"],
                                                                })
                test_batch2 = utils_bowman.create_batch(self.validation_set, self.config["batch_size"],
                                                        self.token_embedder, self.config["dropout_perc"], i + 1)

                z_sample2 = sess.run(self.z_encoding, feed_dict={self.source: test_batch2["sentence_embed"],
                                                                self.seq_lengths: test_batch2["seq_lengths"],
                                                                })

                z = (z_sample + z_sample2) / 2

                gen_sentences = sess.run(
                    self.greedy_decode_sentence,
                    feed_dict={self.z: z})
                interpolate_sentences(test_batch1, test_batch2, gen_sentences, self.token_embedder, self.output_file)

    def gen_sentences_noise(self, num_batches, file_name, noise_var):
        model_checkpoint = self.config["model_save_directory"] + self.config["model_restore_name"]
        self.output_file = open(self.config["log_dir"] + file_name, "w")
        with tf.Session() as sess:
            self.saver.restore(sess, model_checkpoint)

            for i in range(num_batches):
                test_batch1 = utils_bowman.create_batch(self.validation_set, self.config["batch_size"],
                                                        self.token_embedder, self.config["dropout_perc"], i)

                z_sample = sess.run(self.z_encoding, feed_dict={self.source: test_batch1["sentence_embed"],
                                                                self.seq_lengths: test_batch1["seq_lengths"],
                                                                })

                z = z_sample + np.random.normal(0, noise_var, (np.shape(z_sample)))

                gen_sentences = sess.run(
                    self.greedy_decode_sentence,
                    feed_dict={self.z: z})
                write_all_sentences(test_batch1, gen_sentences, self.token_embedder, self.output_file)

    def interact(self, sentences, out, noise_var):
        model_checkpoint = self.config["model_save_directory"] + self.config["model_restore_name"]
        batch_interact = utils_bowman.prepare_interact_batch(self.token_embedder, self.config["batch_size"], sentences, 0.0)
        sentence_len = len(sentences)

        with tf.Session() as sess:
            self.saver.restore(sess, model_checkpoint)

            z_sample = sess.run(self.z_encoding, feed_dict={self.source: batch_interact["sentence_embed"],
                                                            self.seq_lengths: batch_interact["seq_lengths"]})

            original = sess.run(
                self.greedy_decode_sentence,
                feed_dict={self.z: z_sample
                           })

            z_noise = z_sample + np.random.normal(0, noise_var, (np.shape(z_sample)))
            new = sess.run(
                self.greedy_decode_sentence,
                feed_dict={self.z: z_noise
                           })



            display_interact_sentences(batch_interact, original, sentence_len, self.token_embedder, out)
            display_interact_sentences(batch_interact, new, sentence_len, self.token_embedder, out)


    def generate_random_sentences(self, mu, sigma, out_file):
        model_checkpoint = self.config["model_save_directory"] + self.config["model_restore_name"]
        z_sample = np.random.normal(mu, sigma, (1, self.config["batch_size"], self.config["hidden_dim"] * (self.config["encoder_bidirectional"] + 1)))
        with tf.Session() as sess:
            self.saver.restore(sess, model_checkpoint)
            gen_sentences = sess.run(
                self.greedy_decode_sentence,
                feed_dict={self.z: z_sample})
            for i in range(self.config["batch_size"]):
                sentence = self.token_embedder.gen_sentence_from_indexes_no_repeat(gen_sentences[i])
                print(" ".join(sentence))

    def train(self, restore_state):
        with tf.Session() as sess:
            if self.config["log"] == 1:
                summary_writer_train = tf.summary.FileWriter(self.config["log_dir"] + 'train_log', sess.graph, flush_secs=5)
                summary_writer_validation = tf.summary.FileWriter(self.config["log_dir"] + 'validation_log', sess.graph, flush_secs=5)

            sess.run(tf.global_variables_initializer())


            if restore_state:
                model_checkpoint = self.config["model_save_directory"] + self.config["model_restore_name"]
                self.saver.restore(sess, model_checkpoint)

            start_time = time.time()

            for step in range(self.config["training_steps"]):
                total_time = time.time() - start_time
                if(total_time/60 < self.config["max_train_mins"]):
                    batch_data = utils_bowman.create_batch(self.train_set, self.config["batch_size"],
                                                           self.token_embedder, self.config["dropout_perc"], step)
                    t0 = time.time()
                    z_sample= sess.run(self.z_encoding, feed_dict={self.source: batch_data["sentence_embed"],
                                                                                   self.seq_lengths: batch_data["seq_lengths"],
                                                                                   })

                    y, loss_val, _, kl_loss, kl_weight, sentence, summary_str, var, logp = sess.run(
                        [self.y_prime, self.loss,
                        self.optimiser, self.kl_val, self.kl_weight, self.greedy_decode_sentence, self.run_summary, self.var_param, self.logp],
                        feed_dict={self.z: z_sample,
                                   self.step: float(step),
                                   self.target_ohv: batch_data["target_ohv"],
                                   self.target_masks: batch_data["mask"]})
                    t1 = time.time()

                    if self.config["log"] == 1:
                        summary_writer_train.add_summary(summary_str, step)

                    if (step + 1) % self.config["big_evaluation"] == 0:
                        print("---------BIG EVALUATION---------")
                        print("Step: " + str(step))
                        self.output_file.write("---------BIG EVALUATION---------\n")
                        self.output_file.write("Step: " + str(step) + "\n")
                        self.output_file.write("Batches Over: " + str(self.config["batches_big_evaluation"]) + "\n")
                        train_loss_total = 0
                        test_loss_total = 0
                        test_reconstruct_total = 0
                        total_kl = 0
                        train_bleu_score_total = 0
                        test_bleu_score_total = 0
                        train_word_acc_total = 0
                        test_word_acc_total = 0
                        test_jaccard_total = 0
                        test_levenshtein_total = 0
                        eval_time = time.time()
                        for i in range(self.config["batches_big_evaluation"]):
                            batch_data = utils_bowman.create_batch(self.train_set, self.config["batch_size"],
                                                            self.token_embedder, self.config["dropout_perc"], step + i)
                            test_batch = utils_bowman.create_batch(self.validation_set, self.config["batch_size"],
                                                            self.token_embedder, self.config["dropout_perc"], step + i)

                            z_sample= sess.run(self.z_encoding, feed_dict={self.source: batch_data["sentence_embed"],
                                                                            self.seq_lengths: batch_data["seq_lengths"],
                                                                            })
                            train_loss, kl, sentence_train = sess.run(
                                [self.loss, self.kl_val, self.greedy_decode_sentence],
                                feed_dict={
                                           self.target_ohv: batch_data[
                                               'target_ohv'],
                                           self.step: float(step),
                                           self.z: z_sample,
                                           self.target_masks: batch_data["mask"]})

                            word_acc_train = calc_accuracy(batch_data, sentence_train, self.token_embedder, "sentence")
                            bleu_train = calc_bleu(batch_data, sentence_train, self.token_embedder, "sentence")

                            z_sample = sess.run(self.z_encoding, feed_dict={self.source: test_batch["sentence_embed"],
                                                                            self.seq_lengths: test_batch["seq_lengths"],
                                                                            })
                            test_loss, test_reconstruction_loss, sentence_test = sess.run(
                                [self.loss,self.logp, self.greedy_decode_sentence],
                                feed_dict={
                                           self.target_ohv: test_batch[
                                               'target_ohv'],
                                           self.step: float(step),
                                           self.z: z_sample,
                                           self.target_masks: test_batch["mask"]})

                            word_acc_test = calc_accuracy(test_batch, sentence_test, self.token_embedder, "sentence")
                            bleu_test = calc_bleu(test_batch, sentence_test, self.token_embedder, "sentence")

                            jac_d = jaccard(test_batch["sentence"], sentence_test, self.token_embedder)
                            l_d = levenshtein(test_batch["sentence"], sentence_test, self.token_embedder)

                            train_loss_total += train_loss
                            test_loss_total += test_loss
                            test_reconstruct_total += test_reconstruction_loss
                            total_kl += kl
                            train_word_acc_total += word_acc_train
                            test_word_acc_total += word_acc_test
                            train_bleu_score_total += bleu_train
                            test_bleu_score_total += bleu_test
                            test_jaccard_total += jac_d
                            test_levenshtein_total += l_d


                        print("Evaluation Time: " + str(time.time() - eval_time))
                        average_train_loss = train_loss_total / self.config["batches_big_evaluation"]
                        average_test_loss = test_loss_total / self.config["batches_big_evaluation"]
                        average_reconstruction_loss = test_reconstruct_total / self.config["batches_big_evaluation"]
                        average_kl = total_kl / self.config["batches_big_evaluation"]
                        test_perplexity = 2 ** (average_reconstruction_loss / np.log(2))
                        average_acc_train = train_word_acc_total / self.config["batches_big_evaluation"]
                        average_acc_test = test_word_acc_total / self.config["batches_big_evaluation"]
                        average_bleu_train = train_bleu_score_total / self.config["batches_big_evaluation"]
                        average_bleu_test = test_bleu_score_total / self.config["batches_big_evaluation"]
                        average_jac = test_jaccard_total / self.config["batches_big_evaluation"]
                        average_lev = test_levenshtein_total  / self.config["batches_big_evaluation"]


                        print("Training Loss: " + str(average_train_loss))
                        print("Testing Loss: " + str(average_test_loss))
                        print("Reconstruction Test Loss: " + str(average_reconstruction_loss))
                        print("Testing Perplexity: " + str(test_perplexity))
                        print("KL Divergence: " + str(average_kl))
                        print("Training Word Accuracy: " + str(average_acc_train))
                        print("Testing Word Accuracy: " + str(average_acc_test))
                        print("Training BLEU Score: " + str(average_bleu_train))
                        print("Testing BLEU Score: " + str(average_bleu_test))

                        self.output_file.write("Evaluation Time: " + str(time.time() - eval_time) + "\n")
                        self.output_file.write("Training Loss: " + str(average_train_loss) + "\n")
                        self.output_file.write("Testing Loss: " + str(average_test_loss) + "\n")
                        self.output_file.write("Reconstruction Test Loss: " + str(average_reconstruction_loss) + "\n")
                        self.output_file.write("Testing Perplexity: " + str(test_perplexity) + "\n")
                        self.output_file.write("KL Divergence: " + str(average_kl) + "\n")
                        self.output_file.write("Training Word Accuracy: " + str(average_acc_train) + "\n")
                        self.output_file.write("Testing Word Accuracy: " + str(average_acc_test) + "\n")
                        self.output_file.write("Training BLEU Score: " + str(average_bleu_train) + "\n")
                        self.output_file.write("Testing BLEU Score: " + str(average_bleu_test) + "\n")

                        if self.config["log"] == 1:
                            big_train_loss_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="Big Train Loss", simple_value=average_train_loss)])
                            big_test_loss_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="Big Test Loss", simple_value=average_test_loss)])
                            reconstruction_loss_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="Big Reconstruction Loss", simple_value=average_reconstruction_loss)])
                            perplex_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="Perplexity", simple_value=test_perplexity)])
                            kl_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="Big KL Divergence", simple_value=average_kl)])
                            train_acc_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="Big Word Accuracy Train", simple_value=average_acc_train)])
                            test_acc_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="Big Word Accuracy Test", simple_value=average_acc_test)])
                            train_bleu_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="Big Bleu Train", simple_value=average_bleu_train)])
                            test_bleu_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="Big Bleu Test", simple_value=average_bleu_test)])
                            jac_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="Big Jaccard Test", simple_value=average_jac)])
                            lev_summary = tf.Summary(
                                value=[tf.Summary.Value(tag="Big Levenshtein Test", simple_value=average_lev)])


                            summary_writer_train.add_summary(big_train_loss_summary, step)
                            summary_writer_train.add_summary(kl_summary, step)
                            summary_writer_train.add_summary(train_acc_summary, step)
                            summary_writer_train.add_summary(train_bleu_summary, step)
                            summary_writer_validation.add_summary(big_test_loss_summary, step)
                            summary_writer_validation.add_summary(reconstruction_loss_summary, step)
                            summary_writer_validation.add_summary(perplex_summary, step)
                            summary_writer_validation.add_summary(test_acc_summary, step)
                            summary_writer_validation.add_summary(test_bleu_summary, step)
                            summary_writer_validation.add_summary(jac_summary, step)
                            summary_writer_validation.add_summary(lev_summary, step)

                    if step % self.config["eval_frequency"] == 0:
                        print("---------EVALUATION---------")
                        print("Step: " + str(step))
                        print("Iteration Time: " + str(t1 - t0))
                        print("Total Minutes Training: " + str(total_time/60))
                        print("KL Loss: " + str(kl_loss))
                        print("KL Weight: " + str(kl_weight))
                        print("Var Param: " + str(var))
                        print("Reconstruction Loss: " + str(logp))

                        display_sentences(batch_data, sentence, 4, self.token_embedder, self.output_file)
                        word_acc = calc_accuracy(batch_data, sentence, self.token_embedder, "sentence")
                        bleu = calc_bleu(batch_data, sentence, self.token_embedder, "sentence")

                        self.output_file.write("---------EVALUATION---------\n")
                        self.output_file.write("Step: " + str(step) + "\n")
                        self.output_file.write("Total Minutes Training: " + str(total_time/60) + "\n")
                        self.output_file.write("KL Loss: " + str(kl_loss)+ "\n")
                        self.output_file.write("KL Weight: " + str(kl_weight)+ "\n")
                        self.output_file.write("Var Param: " + str(var) +"\n")
                        self.output_file.write("Reconstruction Loss: " + str(logp) + "\n")
                        print_and_write_training_info(self.output_file, loss_val, word_acc, bleu)


                        test_batch = utils_bowman.create_batch(self.validation_set, self.config["batch_size"],
                                                               self.token_embedder, 0.0, step)

                        z_sample = sess.run(self.z_encoding, feed_dict={self.source: test_batch["sentence_embed"],
                                                                        self.seq_lengths: test_batch["seq_lengths"],
                                                                        })
                        loss_test, sentence_test, summary_val, recontruction_test = sess.run(
                            [self.loss, self.greedy_decode_sentence, self.run_summary, self.logp],
                            feed_dict={
                                       self.target_ohv: test_batch["target_ohv"],
                                       self.step: float(step),
                                       self.z : z_sample,
                                       self.target_masks: test_batch["mask"]
                                       })


                        print("---------TEST EVALUATION---------")
                        self.output_file.write("---------TEST EVALUATION---------\n")
                        test_word_acc = calc_accuracy(test_batch, sentence_test, self.token_embedder, "sentence")
                        test_bleu = calc_bleu(test_batch, sentence_test, self.token_embedder, "sentence")
                        print_and_write_training_info(self.output_file, loss_test, test_word_acc, test_bleu)
                        print("Reconstruction Loss: " + str(recontruction_test))
                        self.output_file.write("Reconstruction Loss: " + str(recontruction_test) + "\n")
                        display_sentences(test_batch, sentence_test, 4, self.token_embedder, self.output_file)



                        if self.config["log"] == 1:
                            summary_writer_validation.add_summary(summary_val, step)
                            write_scaler_summaries(bleu, test_bleu, word_acc, test_word_acc, summary_writer_train, summary_writer_validation, step)

                        self.output_file.close()
                        self.output_file = open(self.config["log_dir"] + self.out_file_name, "a")

                    if (step + 1) % self.config["save_frequency"] == 0:
                        save_name = self.out_file_name.split(".")[0]
                        checkpoint_path = self.config["model_save_directory"] + save_name + '.ckpt'
                        self.saver.save(sess, checkpoint_path, global_step=step)

