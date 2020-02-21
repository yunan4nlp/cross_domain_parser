import torch.nn.functional as F
from driver.MST import *
import torch.optim.lr_scheduler
from driver.Layer import *
import numpy as np


def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)


class BiaffineParser(object):
    def __init__(self, model, root_id):
        self.model = model
        self.root = root_id
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def forward(self, words, extwords, tags, char_embs, masks):
        if self.use_cuda:
            words, extwords = words.cuda(self.device), extwords.cuda(self.device),
            tags = tags.cuda(self.device)
            masks = masks.cuda(self.device)

        arc_logits, rel_logits = self.model.forward(words, extwords, tags, char_embs, masks)
        # cache
        self.arc_logits = arc_logits
        self.rel_logits = rel_logits
        self.lstm_hidden = self.model.lstm_hidden


    def compute_loss(self, true_arcs, true_rels, lengths, scores, threshold):
        b, l1, l2 = self.arc_logits.size()
        index_true_arcs = _model_var(
            self.model,
            pad_sequence(true_arcs, length=l1, padding=0, dtype=np.int64))
        true_arcs = _model_var(
            self.model,
            pad_sequence(true_arcs, length=l1, padding=-1, dtype=np.int64))
        lower_threshold = scores < threshold
        true_arcs[lower_threshold] = -1
        masks = []
        for length in lengths:
            mask = torch.FloatTensor([0] * length + [-10000] * (l2 - length))
            mask = _model_var(self.model, mask)
            mask = torch.unsqueeze(mask, dim=1).expand(-1, l1)
            masks.append(mask.transpose(0, 1))
        length_mask = torch.stack(masks, 0)
        arc_logits = self.arc_logits + length_mask

        ###
        arc_loss = F.cross_entropy(arc_logits.view(b * l1, l2), true_arcs.view(b * l1), ignore_index=-1)

        size = self.rel_logits.size()
        output_logits = _model_var(self.model, torch.zeros(size[0], size[1], size[3]))

        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, index_true_arcs)):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][int(arcs[i])])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        b, l1, d = output_logits.size()
        true_rels = _model_var(self.model, pad_sequence(true_rels, padding=-1, dtype=np.int64))
        true_rels[lower_threshold] = -1

        ###
        rel_loss = F.cross_entropy(output_logits.view(b * l1, d), true_rels.view(b * l1), ignore_index=-1)

        loss = arc_loss + rel_loss

        return loss

    def compute_accuracy(self, true_arcs, true_rels):
        b, l1, l2 = self.arc_logits.size()
        pred_arcs = self.arc_logits.data.max(2)[1].cpu()
        index_true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        arc_correct = pred_arcs.eq(true_arcs).cpu().sum()


        size = self.rel_logits.size()
        output_logits = _model_var(self.model, torch.zeros(size[0], size[1], size[3]))

        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, index_true_arcs)):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][arcs[i]])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        pred_rels = output_logits.data.max(2)[1].cpu()
        true_rels = pad_sequence(true_rels, padding=-1, dtype=np.int64)
        label_correct = pred_rels.eq(true_rels).cpu().sum()

        total_arcs = b * l1 - np.sum(true_arcs.cpu().numpy() == -1)

        return arc_correct, label_correct, total_arcs

    def parser_forward(self, words, extwords, tags, char_embs, masks):
        if words is not None:
            self.forward(words, extwords, tags, char_embs, masks)

    def parser_arc_prob(self, lengths):
        arc_probs = []
        arc_logits = self.arc_logits.data.cpu().numpy()
        for arc_logit, length in zip(arc_logits, lengths):
            arc_prob = softmax2d(arc_logit, length, length)
            arc_probs.append(arc_prob)
        return np.array(arc_probs)

    def parser_rel_prob(self, arc_preds, lengths):
        rel_logits = self.rel_logits.data.cpu().numpy()
        rel_probs = []
        for arc_pred, rel_logit, length in zip(arc_preds, rel_logits, lengths):
            rel_logit = rel_logit[np.arange(len(arc_pred)), arc_pred]
            size = rel_logit.shape
            rel_prob = softmax2d(rel_logit, size[0], size[1])
            rel_probs.append(rel_prob)
        return np.array(rel_probs)

    def parse(self, words, extwords, tags, char_embs, lengths, masks, predict=False):
        if words is not None:
            self.forward(words, extwords, tags, char_embs, masks)
        ROOT = self.root
        arcs_batch, arc_values, rels_batch = [], [], []
        arc_logits = self.arc_logits.data.cpu().numpy()
        rel_logits = self.rel_logits.data.cpu().numpy()
        arc_value = None

        for arc_logit, rel_logit, length in zip(arc_logits, rel_logits, lengths):
            arc_probs = softmax2d(arc_logit, length, length)
            if predict:
                arc_pred, arc_value = arc_argmax(arc_probs, length, predict=True)
            else:
                arc_pred = arc_argmax(arc_probs, length)
            
            rel_probs = rel_logit[np.arange(len(arc_pred)), arc_pred]
            rel_pred = rel_argmax(rel_probs, length, ROOT)

            arcs_batch.append(arc_pred)
            arc_values.append(arc_value)
            rels_batch.append(rel_pred)
        if predict:
            return arcs_batch, rels_batch, arc_values
        else:
            return arcs_batch, rels_batch
