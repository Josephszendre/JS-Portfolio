import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

"""
ContextRNN
-used by a manager class which keeps track of its previous states

Implements the model in: https://arxiv.org/pdf/1507.04808.pdf
"""

class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, rnn_type="GRU"):
        super(ContextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type

        self.rnn = getattr(nn, rnn_type)(self.input_size, self.hidden_size, self.n_layers)

    def forward(self, encoded_utterance, prev_hidden=None):
        """
        Uses a standard seq2seq gru encoder to encode a sequence of
          encodings of previous utterances into an encoding of the
          current conversational state.
        :param  encoded_utterance: the result of EncoderRNN
        :param  prev_hidden: (h,c) if LSTM or h if GRU
                      -must be provided if not the first timestep
        :return: (b) The last hidden state for all layers
                       -to be used as the initial hidden state
                        for the Decoder RNN"""

        # this is likely always going to be the case
        if len(encoded_utterance.size()) == 2:
            encoded_utterance = torch.unsqueeze(encoded_utterance)

        result = self.rnn(encoded_utterance, prev_hidden)

        return result[1] # hidden_activations (and content gate if LSTM)


# Exclusively for testing
if __name__ == "__main__":
    # v100's not viable (1080 Ti) if not using fp16 arithematic
    half_precision = True

    params = {"input_size": 512, "hidden_size":512, "n_layers":3}
    enc = ContextRNN(**params)
    seq = Variable(torch.Tensor(np.random.random((3, 5, 512))))

    if torch.cuda.is_available():
        enc.cuda() # module method
        seq = seq.cuda() # Variable method

    # must port all of the weights in enc to HalfTensors
    if half_precision:
        enc.half()
        seq = seq.half()


    # To display a Variable on the GPU we need to call .cpu()
    print(res[0].cpu())



