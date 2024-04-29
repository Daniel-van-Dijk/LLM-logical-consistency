import torch
import torch.nn as nn
import torch.nn.functional as F
class BOW(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_labels):
        super(BOW, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim*4, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, num_labels),
            )

    def forward(self, sentence1, sentence2):
        embeds1 = self.token_embeddings(sentence1)
        embeds2 = self.token_embeddings(sentence2)
        u = embeds1.mean(1)
        v = embeds2.mean(1)
        diff = torch.abs(u - v)
        dotprod = u * v
        combined = torch.hstack([u, v, diff, dotprod])
        logits = self.mlp(combined)
        return logits