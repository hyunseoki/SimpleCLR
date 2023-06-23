'''
    https://jimmy-ai.tistory.com/312
    https://theaisummer.com/simclr/
    https://animilux.github.io/paper_review/2021/01/21/simclr.html
    https://github.com/sthalles/SimCLR
'''

import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
       
    def __init__(self, batch_size, n_views, temperature, logits=False):
        super().__init__()
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature
        self.logits=logits
        self.mask = torch.eye(self.batch_size * n_views, dtype=torch.bool)
        self.labels = self._create_labels()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def _create_labels(self):
        labels = torch.cat([torch.arange(self.batch_size) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels[~self.mask].view(labels.shape[0], -1)

        return labels

    def _calc_similarity(self, features):
        features = F.normalize(features, dim=1)
        return torch.matmul(features, features.T)

    def forward(self, features):
        similarity_matrix = self._calc_similarity(features)
        similarity_matrix = similarity_matrix[~self.mask].view(self.batch_size * 2, -1)

        positives = similarity_matrix[self.labels.bool()].view(self.batch_size * 2, -1)
        negatives = similarity_matrix[~self.labels.bool()].view(self.batch_size * 2, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)
        logits = logits / self.temperature

        if self.logits:
            return logits, labels

        else:
            return self.criterion(logits, labels)



if __name__ == '__main__':
    batch_size = 2
    features = torch.randn(batch_size * 2, 4, device='cuda:0')
    loss = ContrastiveLoss(2, 2, 1)

    print(loss(features))
