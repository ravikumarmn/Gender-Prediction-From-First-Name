import torch.nn as nn
import config

# class GenderClassifier(nn.Module):
#     def __init__(self,vocab_size):
#         super().__init__()
#         self.embedding_layer = nn.Embedding(vocab_size,config.EMBEDDING_SIZE)
#         self.linear_layer = nn.Linear(config.EMBEDDING_SIZE,config.NUM_LABELS)

#     def forward(self,x):
#         x = self.embedding_layer(x)
#         x = self.linear_layer(x)
#         return x
    

import torch.nn as nn
import config


class GenderClassifier(nn.Module):
    def __init__(self, vocab_size):
        super(GenderClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_SIZE)
        self.lstm = nn.LSTM(config.EMBEDDING_SIZE, config.HIDDEN_SIZE_1, batch_first=True)
        self.fc = nn.Linear(config.HIDDEN_SIZE_1, config.NUM_LABELS)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]  # Take the last hidden state
        output = self.fc(hidden)
        # output = self.softmax(output)
        return output
    