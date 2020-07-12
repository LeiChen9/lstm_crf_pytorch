import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from utils import *

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, num_layers, dropout, with_ln):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # padding_idx
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(input_size = embedding_dim,
                                hidden_size = hidden_dim,
                                num_layers = num_layers,
                                dropout = dropout,
                                bidirectional=True)
        self.with_ln = with_ln 
        self.tag_size = len(tag_to_ix)
        if with_ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tag_size)
        self.crf = CRF(self.tag_size)

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.xavier_normal_(self.hidden2tag.weight)
    
    def get_lstm_features(self, seq):
        """
        Args:
            seq: (seq_len, batch_size)
        """
        embed = self.embedding(seq)
        embed = self.dropout(embed)
        lstm_output, _ = self.bilstm(embed)
        if self.with_ln:
            lstm_output = self.layer_norm(lstm_output)
        lstm_features = self.hidden2tag(lstm_output)
        return lstm_features
    
    def neg_log_likelihood(self, seq, tags):
        """
        Args:
            seq: (seq_len, batch_size)
            tags: (seq_len, batch_size)
        """
        seq = seq.transpose(0, 1).squeeze(2)
        tags = tags.transpose(0, 1).squeeze(2)
        lstm_features = self.get_lstm_features(seq)
        return -self.crf(lstm_features, tags)

    def predict(self, seq):
        """
        Args:
            seq: (seq_len, batch_size)
        """
        seq = seq.transpose(0, 1).squeeze(2)
        lstm_features = self.get_lstm_features(seq)
        best_paths = self.crf.decode(lstm_features)

        return best_paths
