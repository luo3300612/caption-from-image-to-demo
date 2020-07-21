import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

label_keys = ['0', '1', '2', '3', '4', '5', '6', '7']
spetial_w2i = {'<pad>': 0, '<sos>': 1, '<eos>': 2}


class MultiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, feat_size, hidden_size,
                 schedule_sample_prob=0, schedule_sample_method='greedy'):
        super(MultiLSTM, self).__init__()
        self.feat_size = feat_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.schedule_sample_prob = schedule_sample_prob
        self.schedule_sample_method = schedule_sample_method

        self.sos_idx = spetial_w2i['<sos>']
        self.eos_idx = spetial_w2i['<eos>']
        self.padding_idx = spetial_w2i['<pad>']

        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(p=0.5)

        self.feat2inputs = nn.ModuleDict({
            key: nn.Linear(self.feat_size, self.embedding_dim) for key in label_keys
        })
        self.feat_dropout = nn.Dropout(p=0.5)

        self.lstms = nn.ModuleDict({
            key: nn.LSTMCell(embedding_dim, hidden_size) for key in label_keys
        })

        self.h_dropout = nn.Dropout(0.5)
        self.logit = nn.Linear(self.hidden_size, self.vocab_size)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_state(self, x):
        bs = x.shape[0]
        return (torch.zeros((bs, self.hidden_size), device=x.device),
                torch.zeros((bs, self.hidden_size), device=x.device))

    def forward(self, feats, seqs):
        bs = feats.shape[0]
        reses = {}
        for key in label_keys:
            seq = seqs[key]
            # print('seq')
            # print(seq)
            # print(seq.shape)
            # print(seq.shape[1])
            outputs = []
            state = self.init_state(feats)
            for i in range(-1, seq.shape[1]):
                rand = np.random.uniform(0, 1, (bs,))
                if i!=-1 and (seq[:, i] == self.eos_idx).sum() + (seq[:, i] == self.padding_idx).sum() == bs:
                    break
                if i == -1:  # start token
                    word_embedding = self.feat2inputs[key](feats)
                elif self.schedule_sample_prob != 0 and (
                        rand < self.schedule_sample_prob).any():  # schedula sample
                    xt = seq[:, i].data.clone()
                    index = rand < self.schedule_sample_prob
                    last_output = outputs[-1].detach()
                    if self.schedule_sample_method == 'greedy':
                        words = last_output.argmax(-1)
                    elif self.schedule_sample_method == 'multinomial':
                        distribution = torch.exp(last_output)
                        words = torch.multinomial(distribution, 1).squeeze(-1)
                    else:
                        raise NotImplementedError
                    xt[index] = words[index]
                    word_embedding = self.embedding_dropout(self.word_embed(xt))
                else:  # Teacher Forcings
                    word_embedding = self.embedding_dropout(self.word_embed(seq[:, i]))
                state = self.lstms[key](word_embedding, state)
                h = state[0]
                logit = self.logit(self.h_dropout(h))
                # print('logit')
                # print(logit)
                outputs.append(logit)
            # print('outputs')
            # print(outputs)
            res = torch.stack(outputs, dim=1)
            res = F.log_softmax(res, dim=-1)
            reses[key] = res
        return reses

    def sample(self, feats, maxlen_dict, mode='greedy'):
        bs = feats.shape[0]
        reses = {}
        for key in label_keys:
            state = self.init_state(feats)
            outputs = []
            is_finished = torch.zeros(bs)
            for i in range(maxlen_dict[key] + 1):
                if i is 0:  # use visual feats at first step
                    word_embedding = self.feat2inputs[key](feats)
                elif mode == 'greedy':
                    last_output = outputs[-1]
                    last_token = last_output.argmax(-1)
                    is_finished[last_token == self.eos_idx] = 1
                    if is_finished.sum() == bs:  # all finished sample
                        break
                    word_embedding = self.embedding_dropout(self.word_embed(last_token))
                else:
                    raise NotImplementedError

                state = self.lstms[key](word_embedding, state)
                h = state[0]
                logit = self.logit(self.h_dropout(h))
                outputs.append(logit)
            res = torch.stack(outputs, dim=1)
            res = F.log_softmax(res, dim=-1)
            reses[key] = res
        return reses


if __name__ == '__main__':
    model = SemanticLSTM(10, 20, 120, 100)
    feats = torch.randn(16, 50)
    tags = torch.randn(16, 50)
    seq = torch.randint(0, 10, (16, 10))
    out = model(feats, tags, seq)
    print(out.shape)
    sample_out = model.sample(feats, tags, 20)
    print(sample_out.shape)
