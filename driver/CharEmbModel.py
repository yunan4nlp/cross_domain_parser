from driver.Layer import *
from driver.Model import drop_sequence_sharedmask


class CharEmbModel(nn.Module):
    def __init__(self, char_vocab, config):
        super(CharEmbModel, self).__init__()
        self.config = config
        self.char_embed = nn.Embedding(char_vocab.char_size, config.char_dims, padding_idx=0)
        char_init = np.zeros((char_vocab.char_size, config.char_dims), dtype=np.float32)
        self.char_embed.weight.data.copy_(torch.from_numpy(char_init))
        '''
        self.lstm = MyLSTM(
            input_size=config.char_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in = config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )
        self.linear = nn.Linear(config.char_dims, config.word_dims, True)
        '''

    def forward(self, chars, char_masks):
        char_emb = self.char_embed(chars)
        b, w_lengths, c_lengths, h = char_emb.size()
        char_emb = char_emb.view(b * w_lengths, c_lengths, h)
        char_masks = char_masks.view(b * w_lengths, c_lengths)

        '''
        outputs, _ = self.lstm(char_emb, char_masks, None)
        outputs = outputs.transpose(1, 0)
        if self.training:
            outputs = drop_sequence_sharedmask(outputs, self.config.dropout_mlp)
        '''

        outputs = avg_pooling(char_emb, char_masks)
        outputs = outputs.view(b, w_lengths, -1)

        return outputs

