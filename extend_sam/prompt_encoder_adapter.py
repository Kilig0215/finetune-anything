# copyright ziqi-jin

import torch.nn as nn
from .segment_anything_ori.modeling.sam import Sam
from .utils import fix_params


class BasePromptEncodeAdapter(nn.Module):

    def __init__(self, ori_sam: Sam, fix=False):
        super(BasePromptEncodeAdapter, self).__init__()

        self.sam_prompt_encoder = ori_sam.prompt_encoder
        if fix:
            fix_params(self.sam_prompt_encoder)

    def forward(self, points=None, boxes=None, masks=None, batch=1):
        if boxes is None:
            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
            return sparse_embeddings, dense_embeddings
        else:
            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes[..., 1:], masks)
        import numpy as np
        import torch
        from copy import deepcopy
        index = deepcopy(boxes[..., 0]).cpu().numpy()
        most_common_value = np.argmax(np.bincount(index))
        count = np.bincount(index)[most_common_value]
        new_dense_embedding = self.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).repeat(
                count*batch, 1, self.sam_prompt_encoder.image_embedding_size[0], self.sam_prompt_encoder.image_embedding_size[1]
            )
        new_sparse_embeddings = torch.zeros((batch,count, 2, self.sam_prompt_encoder.embed_dim), device=self.sam_prompt_encoder._get_device())
        new_dense_embedding = new_dense_embedding.reshape(batch, count, -1, self.sam_prompt_encoder.image_embedding_size[0], self.sam_prompt_encoder.image_embedding_size[1])
        for i in range(batch):
            indices = torch.where(torch.tensor(index) == i)[0]
            new_dense_embedding[i, :len(indices), ...] = dense_embeddings[indices, ...]
            new_sparse_embeddings[i, :len(indices), ...] = sparse_embeddings[indices, ...].clone()

        return new_sparse_embeddings.reshape(batch*count, 2, self.sam_prompt_encoder.embed_dim),\
             new_dense_embedding.reshape(batch*count,-1, self.sam_prompt_encoder.image_embedding_size[0], self.sam_prompt_encoder.image_embedding_size[1])

