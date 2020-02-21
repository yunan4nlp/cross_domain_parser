import sys
sys.path.extend(["../../","../","./"])
import time
import numpy as np
import pickle
import argparse
from driver.Config import *
from driver.Model import *
from driver.Parser import *
from data.Dataloader import *
import pickle
import random
from driver.CharEmbModel import *
from driver.CharEmb import *

def predict(data, parser, charEmbedding, vocab, char_vocab, outputFile, unlabeled=True):
    start = time.time()
    parser.model.eval()
    charEmbedding.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    #all_batch = len(data) // config.test_batch_size + 1

    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, chars, tags, heads, rels, lengths, masks, char_masks, scores, _ = \
            batch_data_variable(onebatch, vocab, char_vocab, ignoreTree=True)
        charEmbedding.forward(chars, char_masks)
        arcs_batch, rels_batch, arc_values = parser.parse(words, extwords, tags, charEmbedding.outputs, lengths, masks, predict=True)
        for id, tree in enumerate(batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab)):
            printDepTree(output, tree, arc_values=arc_values[id])

            if not unlabeled:
                arc_total, arc_correct, rel_total, rel_correct = evalDepTree(onebatch[id][0], tree)
                arc_total_test += arc_total
                arc_correct_test += arc_correct
                rel_total_test += rel_total
                rel_correct_test += rel_correct
    output.close()


    end = time.time()
    during_time = float(end - start)
    print("\nsentence num: %d,  parser predict time = %.2f " % (len(data), during_time))

    if not unlabeled:
        uas = arc_correct_test * 100.0 / arc_total_test
        las = rel_correct_test * 100.0 / rel_total_test
        return arc_correct_test, rel_correct_test, arc_total_test, uas, las


if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../ctb.parser.cfg.debug')
    argparser.add_argument('--unlabled_file', default='')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    config.unlabled_file = args.unlabled_file

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    vec = vocab.create_pretrained_embs(config.pretrained_embeddings_file)

    char_vocab = pickle.load(open(config.load_char_vocab_path, 'rb'))

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(config.use_cuda)

    model = ParserModel(vocab, config, vec)
    model.load_state_dict(torch.load(config.load_model_path))

    char_emb_model = CharEmbModel(char_vocab, config)
    char_emb_model.load_state_dict(torch.load(config.load_char_model_path))

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()
        char_emb_model = char_emb_model.cuda()

    parser = BiaffineParser(model, vocab.ROOT)
    charEmbedding = CharEmb(char_emb_model)
    # test_data = read_corpus(config.test_file, vocab)

    if config.unlabled_file is not "":
        unlabled_data = read_corpus(config.unlabled_file, vocab)
        unlabled_data = domain_labeling(unlabled_data, False)
        predict(unlabled_data, parser, charEmbedding, vocab, char_vocab, config.unlabled_file + '.out')
    else:
        test_data = read_corpus(config.test_file, vocab)
        test_data = domain_labeling(test_data, False)
        arc_correct, rel_correct, arc_total, test_uas, test_las = \
            predict(test_data, parser, charEmbedding, vocab, char_vocab, config.test_file + '.out', unlabeled=False)
        print("Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
              (arc_correct, arc_total, test_uas, rel_correct, arc_total, test_las))

    print("Predict finished.")


