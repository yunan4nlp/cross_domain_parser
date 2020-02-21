class CharEmb(object):
    def __init__(self, model):
        self.model = model
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def forward(self, chars, char_masks):
        if self.use_cuda:
            chars = chars.cuda(self.device)
            char_masks = char_masks.cuda(self.device)
        self.outputs = self.model(chars, char_masks)
