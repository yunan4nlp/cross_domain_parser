import sys
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Config import *
from driver.Model import *
from driver.Parser import *
from data.Dataloader import *
from driver.ClassifierModel import ClassifierModel
from driver.Classifier import *
from data.CharVocab import *
from driver.CharEmbModel import *
from driver.CharEmb import *
import pickle

def train(data, dev_data, test_data, parser, classifier, charEmbedding, vocab, char_vocab, config):
    optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.model.parameters()), config)

    optimizer_ch = Optimizer(filter(lambda ch: ch.requires_grad, charEmbedding.model.parameters()), config)

    c_config = ClassifierConfig(config)
    optimizer_classifer = Optimizer(filter(lambda c: c.requires_grad, classifier.model.parameters()), c_config)

    global_step = 0
    best_UAS = 0
    best_LAS = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    adversarial = False
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        if iter >= config.start_adversarial:
            adversarial = True

        if adversarial:
            print("Start Adversarial")
        batch_iter = 0

        overall_arc_correct, overall_label_correct, overall_total_arcs = 0, 0, 0

        overall_correct, overall_total = 0, 0
        for onebatch in data_iter(data, config.train_batch_size, True):
            words, extwords, chars, tags, heads, rels, lengths, masks, char_masks, scores, domain_labels = \
                batch_data_variable(onebatch, vocab, char_vocab)

            parser.model.train()
            classifier.model.train()
            charEmbedding.model.train()

            charEmbedding.forward(chars, char_masks)
            parser.forward(words, extwords, tags, charEmbedding.outputs, masks)
            classifier.forward(parser.lstm_hidden, masks)
            domain_loss = classifier.compute_loss(domain_labels)
            domain_loss_value = domain_loss.data.cpu().numpy()

            parser_loss = parser.compute_loss(heads, rels, lengths, scores, config.threshold)
            paresr_loss_value = parser_loss.data.cpu().numpy()
            if adversarial:
                loss = parser_loss + domain_loss
            else:
                loss = parser_loss
            loss = loss / config.update_every
            loss.backward()

            arc_correct, label_correct, total_arcs = parser.compute_accuracy(heads, rels)
            correct, total = classifier.compute_accuray(domain_labels)

            overall_correct += correct
            overall_total += total

            overall_arc_correct += arc_correct
            overall_label_correct += label_correct
            overall_total_arcs += total_arcs
            uas = overall_arc_correct.item() * 100.0 / overall_total_arcs
            las = overall_label_correct.item() * 100.0 / overall_total_arcs
            acc = overall_correct / overall_total * 100.0
            during_time = float(time.time() - start_time)
            print("Step:%d, ARC:%.2f, REL:%.2f, Iter:%d, batch:%d, length:%d,time:%.2f, PaserLoss:%.2f, "
                  "DomainLoss:%.2f, DomainACC:%.2f" \
                %(global_step, uas, las, iter, batch_iter, overall_total_arcs, during_time, paresr_loss_value,
                  domain_loss_value, acc))

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, parser.model.parameters()), \
                                        max_norm=config.clip)

                nn.utils.clip_grad_norm_(filter(lambda c: c.requires_grad, classifier.model.parameters()), \
                                         max_norm=config.clip)

                nn.utils.clip_grad_norm_(filter(lambda ch: ch.requires_grad, charEmbedding.model.parameters()), \
                                         max_norm=config.clip)

                optimizer.step()
                optimizer_classifer.step()
                optimizer_ch.step()

                optimizer.zero_grad()
                optimizer_classifer.zero_grad()
                optimizer_ch.zero_grad()

                parser.model.zero_grad()
                classifier.model.zero_grad()
                charEmbedding.model.zero_grad()

                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                arc_correct, rel_correct, arc_total, dev_uas, dev_las = \
                    evaluate(dev_data, parser, charEmbedding, vocab, char_vocab, config.dev_file + '.' + str(global_step))
                print("Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, dev_uas, rel_correct, arc_total, dev_las))
                arc_correct, rel_correct, arc_total, test_uas, test_las = \
                    evaluate(test_data, parser, charEmbedding, vocab, char_vocab, config.test_file + '.' + str(global_step))
                print("Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, test_uas, rel_correct, arc_total, test_las))
                if dev_las > best_LAS:
                    print("Exceed best las: history = %.2f, current = %.2f" %(best_LAS, dev_las))
                    best_LAS = dev_las
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(parser.model.state_dict(), config.save_model_path)
                        torch.save(charEmbedding.model.state_dict(), config.save_char_model_path)
                        torch.save(classifier.model.state_dict(), config.save_classifier_model_path)


def evaluate(data, parser, charEmbedding, vocab, char_vocab, outputFile):
    start = time.time()

    parser.model.eval()
    charEmbedding.model.eval()

    output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, chars, tags, heads, rels, lengths, masks, char_masks, scores, _ = \
            batch_data_variable(onebatch, vocab, char_vocab, ignoreTree=True)
        count = 0
        charEmbedding.forward(chars, char_masks)
        arcs_batch, rels_batch = parser.parse(words, extwords, tags, charEmbedding.outputs, lengths, masks)
        for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab):
            printDepTree(output, tree)
            # arc_total, arc_correct, rel_total, rel_correct = evalDepTree(tree, onebatch[count])
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(onebatch[count][0], tree)
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1

    output.close()

    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test


    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    return arc_correct_test, rel_correct_test, arc_total_test, uas, las


class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='ctb.parser.cfg.debug')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--use_pretrain', action='store_true', default=False)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    # vocab = creatVocab(config.train_file, config.min_occur_count)

    if config.unlabelled_file != "":
        char_vocab = createCharVocab([config.train_file, config.train_target_file, config.unlabelled_file],
                                     config.min_occur_count)
    else:
        char_vocab = createCharVocab([config.train_file, config.train_target_file],
                                    config.min_occur_count)

    if args.use_pretrain:
        vocab = pickle.load(open(config.load_vocab_path, 'rb'))
        vec = vocab.create_pretrained_embs(config.pretrained_embeddings_file)
    else:
        if config.unlabelled_file != "":
            vocab = creatVocab([config.train_file, config.train_target_file, config.unlabelled_file]
                               , config.min_occur_count)
        else:
            vocab = creatVocab([config.train_file, config.train_target_file]
                               , config.min_occur_count)
        vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)

    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))
    pickle.dump(char_vocab, open(config.save_char_vocab_path, 'wb'))

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    torch.set_num_threads(args.thread)
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(config.use_cuda)

    model = ParserModel(vocab, config, vec)
    if args.use_pretrain:
        model.load_state_dict(torch.load(config.load_model_path))
        print("###Load pretrain parser ok.###")

    classifier_model = ClassifierModel(config)
    char_emb_model = CharEmbModel(char_vocab, config)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()
        classifier_model = classifier_model.cuda()
        char_emb_model = char_emb_model.cuda()
    print(model)
    print(classifier_model)
    print(char_emb_model)

    parser = BiaffineParser(model, vocab.ROOT)
    classifier = DomainClassifier(classifier_model)
    charEmbedding = CharEmb(char_emb_model)

    data = read_corpus(config.train_file, vocab)
    data_target = read_corpus(config.train_target_file, vocab)

    source_data = domain_labeling(data, True)
    target_data = domain_labeling(data_target, False)
    source_data.extend(target_data)

    #data.extend(data_target)
    dev_data = read_corpus(config.dev_file, vocab)
    test_data = read_corpus(config.test_file, vocab)

    dev_data = domain_labeling(dev_data, False)
    test_data = domain_labeling(test_data, False)

    train(source_data, dev_data, test_data, parser, classifier, charEmbedding, vocab, char_vocab, config)
