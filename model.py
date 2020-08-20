import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnSentimentAnalysis(nn.Module):
    def __init__(self,
                input_dim = 1024,
                embedding_dim = 100,
                hidden_dim = 25,
                output = 1,
                kernel_size = [2,3],
                dropout = 0.5):

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_dim,
                                      embedding_dim)

        self.conv1d = nn.ModuleList([
                                    nn.Conv1d(embedding_dim,
                                              hidden_dim,
                                              size)
                                    for size in kernel_size
                                    ])

        self.projection =  nn.Linear(hidden_dim * len(kernel_size),
                                    output)
    def forward(self, batch):
        # batch = [batch, sent_len]
        embedded = self.embedding(batch)
        # embedded = [batch, sent_len, embed dim]
        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch, embed dim, sent_len]
        convs = [F.relu(conv(embedded)) for conv in self.conv1d]
        #convd = [batch, hidden_dim, sent_len - kernel_size[n] + 1]
        convs_pool = [torch.max(conv, axis = -1).values for conv in convs]
        #pooled_n = [batch, hidden_dim]
        cat = self.dropout(torch.cat(convs_pool, dim = 1 ))
        return self.projection(cat)
