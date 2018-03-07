import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from corpus import Corpus
import numpy as np

"""
Is pytorch's native nn.Embedding still not support max_norm?
How well is fp16 arithematic supported?
How would using multidimensional tensors speed up or slow down training?

:ivar Caveats
-need to call .half() on every module involved so their weights
   can be converted to type HalfTensor
"""


# are utterances by the user processed differently form EVE?
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers, corpus=None, rnn_type="LSTM", embedding=None,
                       embedding_dim=-1, train_embedding=False, bidirectional=True):
        super(EncoderRNN, self).__init__()
        assert not (embedding is None and train_embedding==False)

        assert corpus is not None

        self.corpus = corpus
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.train_embedding = train_embedding

        self.embedding_dim = embedding.embedding_dim
        self.corpus_size = embedding.num_embeddings
        self.embedding = [embedding]

        self.rnn = getattr(nn, rnn_type)(self.embedding_dim, self.hidden_size, self.n_layers, bidirectional=bidirectional, batch_first=False)

    def forward(self, utterances_idc):
        """
        :param utterance: a LongTensor of dimension 1
                            the eos token must be present at least
                            one time. Most of the time in production
                            it will be twice in the sequence.

        :return: the context vector (batch, hidden_size)
        """
        # If you don't specify the initial hidden state
        #  it is assumed to be uniformly 0.
        # LSTM hidden state: (h0, c0) both of shape (

        eos = self.corpus.eos
        assert type(utterances_idc.data) in [torch.LongTensor, torch.cuda.LongTensor], "EncoderRNN.forward parameter utterance must be on the gpu and a torch.HalfTensor"

        where_eos = sorted(np.where((utterances_idc == eos).data.cpu().numpy())[0])
        num_utterances = len(where_eos)
        assert num_utterances > 0
        #assert where_eos[0] != 0, str(utterances_idc) + str(where_eos)

        where_eos = [-1] + where_eos

        results = []
        for i, j in zip(where_eos[:-1], where_eos[1:]):
            utterance_indices = utterances_idc[i+1:j+1]
            embedded = self.embedding[0](utterance_indices.cpu()).cuda()
            encoding = self.rnn(embedded)[0][-1:]
            results.append(encoding)

        result = torch.cat(results, 0)
        return result

if __name__ == "__main__":
    half = True


    emb = nn.Embedding(5000, 768)
    params = {"hidden_size":512, "n_layers":3, \
              "embedding":emb, "corpus_size":-1, \
              "embedding_dim":-1, "train_embedding":False, \
              "bidirectional":False}
    enc = EncoderRNN(**params)

    enc.cuda()
    seq = Variable(torch.LongTensor(np.random.randint(0, 5000, (1,30)))).cuda()
    if half:
        enc = enc.half()

    res = enc(seq)
    print(res)
    import time
    time.sleep(10)

