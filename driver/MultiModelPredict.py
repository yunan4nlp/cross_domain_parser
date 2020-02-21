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
from data.CharVocab import *
from driver.CharEmbModel import *
from driver.CharEmb import *
import pickle
import random

def arc_pred(avg_arc_probs, lengths, predict):
    arcs_batch, arc_values = [], []
    for arc_probs, length in zip(avg_arc_probs, lengths):
        if predict:
            arc_pred, arc_value = arc_argmax(arc_probs, length, ensure_tree=True, predict=True)
            arcs_batch.append(arc_pred)
            arc_values.append(arc_value)
        else:
            arc_pred = arc_argmax(arc_probs, length)
            arcs_batch.append(arc_pred)
    if predict:
        return arcs_batch, arc_values
    else:
        return arcs_batch

def rel_pred(avg_rel_probs, lengths, ROOT):
    rels_batch = []
    for rel_probs, length in zip(avg_rel_probs, lengths):
        rel_pred = rel_argmax(rel_probs, length, ROOT)
        rels_batch.append(rel_pred)
    return rels_batch

def multi_models_predict(data, parser_list, charEmbedding_list, vocab, char_vocab, outputFile, unlabeled=True):
    start = time.time()
    for parser in parser_list:
        parser.model.eval()
    for charEmbedding in charEmbedding_list:
        charEmbedding.model.eval()

    output = open(outputFile, 'w', encoding='utf-8')
    #all_batch = len(data) // config.test_batch_size + 1

    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, chars, tags, heads, rels, lengths, masks, char_masks, scores, _ = \
            batch_data_variable(onebatch, vocab, char_vocab, ignoreTree=True)

        arc_probs_list = []
        for (parser, charEmbedding) in zip(parser_list, charEmbedding_list):
            charEmbedding.forward(chars, char_masks)
            parser.parser_forward(words, extwords, tags, charEmbedding.outputs, masks)
            arc_probs = parser.parser_arc_prob(lengths)
            arc_probs_list.append(arc_probs)

        avg_arc_probs = np.sum(arc_probs_list, 0) / model_num
        arcs_batch, arc_values = arc_pred(avg_arc_probs, lengths, predict=True)

        rel_probs_list = []
        for parser in parser_list:
            rel_probs = parser.parser_rel_prob(arcs_batch, lengths)
            rel_probs_list.append(rel_probs)
        avg_rel_probs = np.sum(rel_probs_list, 0) / model_num
        rels_batch = rel_pred(avg_rel_probs, lengths, parser.root)

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

def check_list(l1, l2):
    max_len = len(l1)
    assert max_len == len(l2)
    for idx in range(max_len):
        assert l1[idx] == l2[idx]


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
    argparser.add_argument('--config_list', nargs = '+', help = '<Required> Set config List', required = True)

    argparser.add_argument('--unlabled_file', default='')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()

    use_cuda = False
    if gpu and args.use_cuda: use_cuda = True
    print("\nGPU using status: ", use_cuda)

    config_list = []
    parser_list = []
    vocab_list = []
    char_vocab_list = []
    charEmbedding_list = []
    for config_file in args.config_list:
        print("################################")
        print(config_file)
        config = Configurable(config_file, extra_args)
        vocab = pickle.load(open(config.load_vocab_path, 'rb'))
        char_vocab = pickle.load(open(config.load_char_vocab_path, 'rb'))
        char_emb_model = CharEmbModel(char_vocab, config)
        char_emb_model.load_state_dict(torch.load(config.load_char_model_path))

        vec = vocab.create_pretrained_embs(config.pretrained_embeddings_file)

        model = ParserModel(vocab, config, vec)
        model.load_state_dict(torch.load(config.load_model_path))

        if use_cuda:
            torch.backends.cudnn.enabled = True
            model = model.cuda()
            char_emb_model = char_emb_model.cuda()

        charEmbedding = CharEmb(char_emb_model)
        parser = BiaffineParser(model, vocab.ROOT)
        parser_list.append(parser)
        vocab_list.append(vocab)
        char_vocab_list.append(char_vocab)
        charEmbedding_list.append(charEmbedding)
    model_num = len(parser_list)

    ref_v = vocab_list[0]
    ch_ref_v = char_vocab_list[0]
    for ch_v in char_vocab_list[1:]:
        check_list(ch_v._id2char, ch_ref_v._id2char)

    for vocab in vocab_list[1:]:
        assert ref_v.PAD == vocab.PAD
        assert ref_v.ROOT == vocab.ROOT
        assert ref_v.UNK == vocab.UNK
        check_list(ref_v._id2extword, vocab._id2extword)
        check_list(ref_v._id2word, vocab._id2word)
        check_list(ref_v._id2rel, vocab._id2rel)
        check_list(ref_v._id2tag, vocab._id2tag)
        check_list(ref_v._wordid2freq, vocab._wordid2freq)

    #test_data = read_corpus(config.test_file, vocab)

    if args.unlabled_file is not "":
        unlabled_data = read_corpus(args.unlabled_file, ref_v)
        unlabled_data = domain_labeling(unlabled_data, False)
        multi_models_predict(unlabled_data, parser_list, charEmbedding_list, ref_v, ch_ref_v, args.unlabled_file + '.out', unlabeled=True)
    else:
        test_data = read_corpus(config.test_file, ref_v)
        test_data = domain_labeling(test_data, False)
        arc_correct, rel_correct, arc_total, test_uas, test_las = \
        multi_models_predict(test_data, parser_list, charEmbedding_list, ref_v, ch_ref_v, config.test_file + '.out', unlabeled=False)
        print("Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
              (arc_correct, arc_total, test_uas, rel_correct, arc_total, test_las))
    print("Predict finished.")


