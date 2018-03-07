import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np


"""
What exactly serves as the context vector(s)
-for multilayer ContextRNN does its final hidden state
  for each layer count as the initial hidden in each
  layer of the DecoderRNN?

Train the embeddings first on word2vec or something similar
-by pretraining you avoid having to train another extra 300k parameters

Assuming a batch size of 1 right now so stopping when <eos> is predicted 
  is simpler
  
Chandra questions: 
-Do you think that teacher forcing is a viable strategy early on in training
  for an end to end trainable system?
"""

class DecoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_layers, embedding, max_response_length = 30, rnn_type="GRU", corpus=None):#beam_length = -1, beam_width = -1):
        super(DecoderRNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.corpus_size = embedding.num_embeddings
        self.n_layers = n_layers
        self.max_response_length = max_response_length
        self.rnn_type = rnn_type

        # for debugging / ability to print from within this class
        self.corpus = corpus

        # we don't have to use the same embeddings
        # for the EncoderRNN as DecoderRNN
        self.embedding = [embedding]
        self.fc = nn.Linear(self.hidden_size, self.corpus_size)
        self.rnn = getattr(nn, rnn_type)(self.embedding_dim, self.hidden_size, self.n_layers)


    def forward(self, conv_context, actual_tokens=None):
        """
        :param conv_context: n_layers, batch, hidden_dim
        :return: output_probs: torch.Tensor (seq, batch, corpus_size)
        :return: output_tokens: torch.LongTensor (seq, batch)
        """

        output_probs = []
        output_tokens = []

        hidden_state = conv_context
        # 0 corresponds to <sos>

        batch_size = conv_context.size(1) if type(conv_context) is not list else conv_context[0].size(1)

        next_input_token = Variable(torch.LongTensor(np.zeros((1, batch_size),dtype=np.int_)))

        self.embedding[0].cpu()
        #if torch.cuda.is_available():
        #    next_input_token = next_input_token.cuda() # 1, batch

        use_teacher_forcing = actual_tokens is not None

        if use_teacher_forcing:
            actual_tokens = Variable(actual_tokens.squeeze())

        do_continue=True
        timestep = 0

        while do_continue:
            embedded_input = self.embedding[0](next_input_token).cuda() # 1, batch, embedding_dim
            assert len(embedded_input.size()) == 3, "embedded_input must have 3 dimensions"

            # output: 1 x batch x hidden_dim, hidden_state: n_layers, batch, hidden_dim
            output, hidden_state = self.rnn(embedded_input, hidden_state)

            logits = self.fc(output)# 1 x batch x corpus_size
            probs =  F.log_softmax(logits, dim=-1) # 1, batch, corpus_size
            output_probs.append(probs)
            # used for input at the next time step
            #top_words = torch.squeeze(torch.topk(probs, 1, -1)[1], 2) # 1 x batch
            top_words = torch.squeeze(torch.topk(probs, 1, -1)[1], 0) # 1 x batch

            if use_teacher_forcing:
                next_input_token = actual_tokens[timestep:timestep+1].unsqueeze(0)
            else:
                next_input_token = top_words.cpu()

            output_tokens += [top_words]

            timestep += 1
            # two conditions will make stop this loop
            # assumes a batch size of 1 otherwise this gets really complicated
            # need to think deeply on how to batch this
            #print(torch.squeeze(top_words.data.cpu()).numpy()[0])

            if use_teacher_forcing:
                if timestep == actual_tokens.size(0):
                    do_continue = False
                    del actual_tokens # desperate try to use less memory
            elif not use_teacher_forcing:
                if top_words.data.cpu().numpy()[0] == self.corpus.eos:
                    do_continue = False
            if timestep == self.max_response_length:
                do_continue=False

        if not use_teacher_forcing and timestep!=self.max_response_length:
            print("Decoding ended early:", timestep)

        # returns (seq, batch, corpus_size)
        return torch.cat(output_probs, 0), torch.cat(output_tokens, 0)

