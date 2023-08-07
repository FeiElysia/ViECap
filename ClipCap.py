import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Tuple, Optional, List
from transformers import GPT2LMHeadModel

class MlpTransformer(nn.Module):

    def __init__(
        self,
        input_size: int,                   # the input size of mlp
        hidden_size: int,                  # the hidden layer size of mlp
        output_size: Optional[int] = None, # the output size of mlp
        act = nnf.relu,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        output_size = output_size if output_size is not None else input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = act
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        query_size: int,
        key_value_size: int,
        num_heads: int,
        bias = True,
        dropout: float = 0.0
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = query_size // num_heads # the size of each head
        self.scale = self.head_size ** -0.5      # normalization factor for each head
        self.to_queries = nn.Linear(query_size, query_size, bias = bias)
        #  projecting key and value together and spliting them for computing efficiently
        self.to_keys_values = nn.Linear(key_value_size, 2 * query_size, bias = bias)
        self.project = nn.Linear(query_size, query_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        key_value = key_value if key_value is not None else query
        b, n, d_query = query.shape
        _, m, _ = key_value.shape
        queries = self.to_queries(query).reshape(b, n, self.num_heads, self.head_size) # (batch_size, n_seq, num_heads, head_size)
        keys_values = self.to_keys_values(key_value).reshape(b, m, 2, self.num_heads, self.head_size) # (batch_size, m_seq, 2, num_heads, head_size)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1] # (batch_size, m_seq, num_heads, head_size), (batch_size, m_seq, num_heads, head_size)
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale # (batch_size, n_seq, m_seq, num_heads)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(dim = 1) # expending dimension, shape: (batch_size, 1, m_seq)
            attention = attention.masked_fill(mask.unsqueeze(dim = 3), float("-inf")) # expending dimension n_seq head and fill -inf according to mask

        attention = attention.softmax(dim = 2) # softmax alongside the dimension of key_value pairs
        outputs = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, d_query) # (batch_size, n_seq, d_query)
        outputs = self.project(outputs)
        return outputs, attention

class TransformerLayer(nn.Module):

    def __init__(
            self,
            query_size: int,
            key_value_size: int,
            num_heads: int,
            mlp_ratio = 4.0,
            bias = False,
            dropout: float = 0.0,
            act = nnf.relu,
            norm_layer: nn.Module = nn.LayerNorm
        ) -> None:
        super(TransformerLayer, self).__init__()
        self.norm1 = norm_layer(query_size)
        self.attn = MultiHeadAttention(query_size, key_value_size, num_heads, bias = bias, dropout = dropout)
        self.norm2 = norm_layer(query_size)
        self.mlp = MlpTransformer(query_size, int(query_size * mlp_ratio), act = act, dropout = dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        query_, self.attention = self.attn(self.norm1(query), key_value, mask)
        query = query + query_
        query = query + self.mlp(self.norm2(query))
        return query

class Transformer(nn.Module):

    def __init__(
            self,
            query_size: int,                      # query size
            num_layers: int,                      # number of layer
            num_heads: int,                       # number of head
            key_value_size: Optional[int] = None, # key/value size
            mlp_ratio: float = 2.0,               # ratio for hidden size in mlp
            act = nnf.relu,                       # activation
            norm_layer: nn.Module = nn.LayerNorm  # normalization
        ) -> None:
        super(Transformer, self).__init__()
        key_value_size = key_value_size if key_value_size is not None else query_size
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerLayer(query_size, key_value_size, num_heads, mlp_ratio = mlp_ratio, act = act, norm_layer = norm_layer))
        self.layers = nn.Sequential(*layers)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        self.attentions = []
        for layer in self.layers:
            query = layer(query, key_value, mask)
            self.attentions.append(layer.attention)
        return query

class MappingNetwork(nn.Module):

    def __init__(
        self,
        clip_project_length: int,
        clip_hidden_size: int,
        prefix_length: int,
        d_model: int,              # the hidden size of language model
        num_layers: int = 8,
        num_heads: int = 8
    ) -> None:
        super(MappingNetwork, self).__init__()
        self.clip_project_length = clip_project_length
        # projector for input
        self.linear = nn.Linear(clip_hidden_size, clip_project_length * d_model)
        # learnable prefix embeddings
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, d_model), requires_grad = True)
        self.transformer = Transformer(d_model, num_layers, num_heads)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: clip cls feature with a shape of (batch_size, clip_hidden_size)
        Return:
            the embeddings of prefix with the shape of (batch_size, prefix_length, d_model)
        """
        x = self.linear(x).view(x.shape[0], self.clip_project_length, -1)  # (b, clip_project_length, d_model)
        prefix = self.prefix_const.unsqueeze(dim = 0).expand(x.shape[0], *self.prefix_const.shape) # (b, prefix_length, d_model)
        inputs = torch.cat((x, prefix), dim = 1)                           # (b, clip_project_length + prefix_length, d_model)
        outputs = self.transformer(inputs)[:,self.clip_project_length:,:]  # (b, prefix_length, d_model)

        return outputs

def get_language_mode(lm_type):
    if 'gpt' in lm_type:
        model = GPT2LMHeadModel.from_pretrained(lm_type)
        hidden_size = model.config.hidden_size
    elif 'opt' in lm_type:
        from modeling_opt import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(lm_type, torch_dtype = torch.float16)
        hidden_size = model.config.word_embed_proj_dim
    return model, hidden_size

class ClipCaptionModel(nn.Module):

    def __init__(
        self,
        continuous_length: int = 10,
        clip_project_length: int = 10,
        clip_hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        gpt_type: str = 'gpt2',
        soft_prompt_first: bool = False,
        only_hard_prompt: bool = False
    ) -> None:
        """
        Args:
            continuous_length: the length of soft prompts which will be fed into language model as continuous part
            clip_project_length: clip cls features (b, 1, d) -> (b, n, d)
            clip_hidden_size: the dimensions of CLIP features
            num_layers: the number of layer in projector
            num_heads: the number of heads each layer
            gpt_type: the language model
            soft_prompt_first: False -> hard prompt + soft prompt; True -> soft prompt + hard prompt
            only_hard_prompt: using the hard prompts only
        """
        super(ClipCaptionModel, self).__init__()
        self.soft_prompt_first = soft_prompt_first
        self.only_hard_prompt = only_hard_prompt
        self.continuous_length = continuous_length
        self.gpt, self.gpt_hidden_size  = get_language_mode(gpt_type)
        self.mapping_network = MappingNetwork(clip_project_length, clip_hidden_size, continuous_length, self.gpt_hidden_size, num_layers, num_heads)
        self.gpt_type = gpt_type
    
    def word_embed(self, caption_tokens):
        if 'gpt' in self.gpt_type:
            caption_embeddings = self.gpt.transformer.wte(caption_tokens)         # (b, caption_length, gpt_hidden_size)
        elif 'opt' in self.gpt_type:
            caption_embeddings = self.gpt.model.decoder.embed_tokens(caption_tokens)
        return caption_embeddings
    
    def forward(
        self,
        continuous_prompt: torch.Tensor,
        caption_tokens: torch.Tensor,
        hard_prompts_length: Optional[List] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            continuous_prompt: tensor with a shape of (b, clip_hidden_size), in text-only training, the caption features are eaxtracted from CLIP and used as image features
            caption_tokens: caption tokens with a shape of (b, max_length_per_caption)
            hard_prompts_length: list with len = batch size, the length of hard prompts constructed for each caption
            mask: tensor with a shape of (b, discrete_length + continuous_length + max_length_per_caption), valid texts for attention computing
        Return:
            the output of language model
        """
        caption_embeddings = self.word_embed(caption_tokens)
        continuous_embeddings = self.mapping_network(continuous_prompt).view(-1, self.continuous_length, self.gpt_hidden_size) # (b, continuous_length, gpt_hidden_size)
        if hard_prompts_length is not None:   # with hard prompts
            if self.only_hard_prompt:
                embeddings = caption_embeddings
            elif self.soft_prompt_first:      # soft prompts + hard prompts
                embeddings = torch.cat((continuous_embeddings, caption_embeddings), dim = 1)
            else:                             # hard prompts + soft prompts
                embeddings = None
                for i in range(len(hard_prompts_length)):
                    length = hard_prompts_length[i]
                    temp_embeddings = torch.cat((caption_embeddings[i][:length], continuous_embeddings[i], caption_embeddings[i][length:]), dim = 0).unsqueeze(dim = 0)
                    if embeddings is None:
                        embeddings = temp_embeddings
                    else:
                        embeddings = torch.cat((embeddings, temp_embeddings), dim = 0)
        else: # without hard prompts
            embeddings = torch.cat((continuous_embeddings, caption_embeddings), dim = 1)       # (b, continuous_length + caption_length, gpt_hidden_size)

        out = self.gpt(inputs_embeds = embeddings.type(self.gpt.dtype), attention_mask = mask)

        return out

class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.mapping_network.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self