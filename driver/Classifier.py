import torch.nn.functional as F
from driver.MST import *
import torch.optim.lr_scheduler
from driver.Layer import *
import numpy as np

class ClassifierConfig():
    def __init__(self, config):
        self.learning_rate = config.classifier_learning_rate
        self.decay= config.decay
        self.decay_steps = config.decay_steps
        self.beta_1 = config.beta_1
        self.beta_2 = config.beta_2
        self.epsilon = config.epsilon

class DomainClassifier(object):
    def __init__(self, model):
        self.model = model
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def forward(self, lstm_hidden, masks):
        if self.use_cuda:
            masks = masks.cuda(self.device)
        score = self.model.forward(lstm_hidden, masks)
        self.score = score

    def compute_loss(self, true_labels):
        if self.use_cuda:
            true_labels = true_labels.cuda(self.device)
        loss = F.cross_entropy(self.score, true_labels)
        return loss

    def compute_accuray(self, true_labels):
        total = true_labels.size()[0]
        pred_labels = self.score.data.max(1)[1].cpu()
        correct = pred_labels.eq(true_labels).cpu().sum().item()
        return correct, total
