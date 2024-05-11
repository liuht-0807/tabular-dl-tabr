import lib
import delu
import faiss
import math
import torch
import torch.nn.functional as F

from torch import Tensor, nn
from typing import Union, Literal, Optional        
from loguru import logger
            
class Encoder(nn.Module):
    """
    Further project reduced set points into a shared embedding space, where retrieving neighboring reduced points 
    from other heterogeneous learnwares improve the prediction for the current learnwares' tasks.
    """
    def __init__(
        self,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        num_embeddings: Optional[dict],
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        mixer_normalization: Union[bool, Literal['auto']],
        dropout0: float,
        dropout1: Union[float, Literal['dropout0']],
        normalization: str,
        activation: str,
    ):
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()
        if dropout1 == 'dropout0':
            dropout1 = dropout0

        self.one_hot_encoder = (
            lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        )
        self.num_embeddings = (
            None
            if num_embeddings is None
            else lib.make_module(num_embeddings, n_features=n_num_features)
        )
        
        d_in = (
            n_num_features
            * (1 if num_embeddings is None else num_embeddings['d_embedding'])
            + n_bin_features
            + sum(cat_cardinalities)
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)
        
        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )
        
        self.d_main = d_main
        self.d_multiplier = d_multiplier
        self.d_block = d_block
        self.dropout0 = dropout0
        self.dropout1 = dropout1
        
        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.activation = Activation()
    
    def forward(self, x_: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x_num = x_.get('num')
        x_bin = x_.get('bin')
        x_cat = x_.get('cat')
        del x_

        x = []
        if x_num is None:
            assert self.num_embeddings is None
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_bin is not None:
            x.append(x_bin)
        if x_cat is None:
            assert self.one_hot_encoder is None
        else:
            assert self.one_hot_encoder is not None
            x.append(self.one_hot_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1)

        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)    
        return x
    

class RetrieveModule(nn.Module):
    def __init__(
        self,
        encoder,
        n_classes,
        context_dropout,
        memory_efficient,
        candidate_encoding_batch_size
    ):
        super().__init__()
        self.encoder = encoder
        self.d_main = encoder.d_main
        self.d_multiplier = encoder.d_multiplier
        self.dropout0 = encoder.dropout0
        self.normalization = encoder.normalization
        self.activation = encoder.activation
        self.label_encoder = (
            nn.Linear(1, encoder.d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, self.d_main), delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        
        self.K = nn.Linear(self.d_main, self.d_main)
        
        d_block = int(self.d_main * self.d_multiplier)
        self.T = nn.Sequential(
            nn.Linear(self.d_main, d_block),
            self.activation,
            nn.Dropout(self.dropout0),
            nn.Linear(d_block, self.d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)
        
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()
        
    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_):
        x = self.encoder(x_)
        return x, self.K(x if self.normalization is None else self.normalization(x))
    
    def forward(self, x_, y, candidate_x_, candidate_y, context_size, is_train):
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            candidate_k = (
                self._encode(candidate_x_)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x)[1]
                        for x in delu.iter_batches(
                            candidate_x_, self.candidate_encoding_batch_size
                        )
                    ]
                )
            )
        
        x, k = self._encode(x_)
        if is_train:
            # NOTE: here, we add the training batch back to the candidates after the
            # function `apply_model` removed them. The further code relies
            # on the fact that the first batch_size candidates come from the
            # training batch.
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None
        
        batch_size, d_main = k.shape
        assert d_main == self.d_main
        device = k.device
        
        with torch.no_grad():
            
            #! only for debugging
            logger.info(f"Data type: {candidate_k.dtype}")  
            logger.info(f"Data shape: {candidate_k.shape}")
    
            if self.search_index is None:
                # self.search_index = (
                #     faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), self.d_main)
                #     if device.type == 'cuda'
                #     else faiss.IndexFlatL2(self.d_main)
                # )
                
                #! only for debugging
                try:
                    if device.type == 'cuda':
                        gpu_resource = faiss.StandardGpuResources() 
                        self.search_index = faiss.GpuIndexFlatL2(gpu_resource, self.d_main)
                    else:
                        self.search_index = faiss.IndexFlatL2(self.d_main)
                except Exception as e:
                    logger.error(f"Failed to initialize GPU resources or index: {str(e)}")
                    raise
                
            # Updating the index is much faster than creating a new one.
            #! only for debugging
            try:
                self.search_index.reset()
                self.search_index.add(candidate_k)  # type: ignore[code]
            except Exception as e:
                logger.error(f"Error during adding vectors to index: {str(e)}")
                raise
            
            distances: Tensor
            context_idx: Tensor
            #! only for debugging
            try:
                distances, context_idx = self.search_index.search(  # type: ignore[code]
                    k, context_size + (1 if is_train else 0)
                )
            except Exception as e:
                logger.error(f"Error during search: {str(e)}")
                raise
            
            if is_train:
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed). This is implemented by setting
                # self-to-self element as inf within the disntances matrix.
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(
                {
                    ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                        context_idx
                    ].flatten(0, 1)
                    for ftype in x_
                }
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        # In theory, when autograd is off, the distances obtained during the search
        # can be reused. However, this is not a bottleneck, so let's keep it simple
        # and use the same code to compute `similarities` during both
        # training and evaluation.
        similarities = (
            -k.square().sum(-1, keepdim=True) # [batch_size, 1]
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2) # k[..., None, :] -> [batch_size, 1, d_main]
                                                                                # context_k -> [batch_size, context_size, d_main]
                                                                                # context_k.transpose(-1, -2) -> [batch_size, d_main, context_size]
                                                                                # (k[..., None, :] @ context_k.transpose(-1, -2)) -> [batch_size, 1, context_size]
                                                                                # squeeze(-2) -> [batch_size, context_size]
            - context_k.square().sum(-1) # [batch_size, context_size]
        )
        probs = F.softmax(similarities, dim=-1)                                 # probs -> [batch_size, context_size]
        probs = self.dropout(probs)

        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None]) # candidate_y -> [n_candidates]
                                                                                # candidate_y[context_idx] -> same as context_idx, [batch_dize, context_size]
                                                                                # candidate_y[context_idx][..., None] -> [batch_size, context_size, 1]
                                                                                # self.label_encoder -> [batch_size, context_size, d_main]
        
        values = context_y_emb + self.T(k[:, None] - context_k)                 # k[:, None] -> [batch_size, 1, d_main], context_k -> [batch_size, context_size, d_main]
                                                                                # self.T -> [batch_size, context_size, d_main]
                                                                                
        context_x = (probs[:, None] @ values).squeeze(1)                        # probs[:, None] -> [batch_size, 1, context_size]
                                                                                # squeeze(1), [batch_size, 1, d_main] -> [batch_size, d_main]
        x = x + context_x
        
        return x