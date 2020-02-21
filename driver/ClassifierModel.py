from driver.Layer import *
from torch.autograd import Function
from driver.Model import drop_sequence_sharedmask



class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        reverse_grad_output = grad_output.neg() * ctx.alpha
        return reverse_grad_output, None

class ClassifierModel(nn.Module):
    def __init__(self, config):
        super(ClassifierModel, self).__init__()
        self.config = config
        self.Linear = nn.Linear(config.lstm_hiddens * 2, config.lstm_hiddens, True)
        self.MLP = NonLinear(
            input_size = config.lstm_hiddens,
            hidden_size = config.lstm_hiddens,
            activation = nn.LeakyReLU(0.1))
        self.output = nn.Linear(config.lstm_hiddens, 2, False)

    def forward(self, lstm_hidden, masks):
        hidden = avg_pooling(lstm_hidden, masks)
        hidden = self.Linear.forward(hidden)
        hidden = ReverseLayerF.apply(hidden, self.config.alpha)
        mlp_hidden = self.MLP(hidden)
        score = self.output(mlp_hidden)
        return score






