import clip
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers import GPT2Tokenizer, GPT2LMHeadModel


@torch.no_grad()
def opt_search(
    prompts: Optional[str] = None,
    tokens: Optional[torch.Tensor] = None,
    embeddings: Optional[torch.Tensor] = None,
    max_len: int = 64,
    beam_width: int = 5,
    end_of_sentence: str = ".",
    tokenizer: GPT2Tokenizer = None,
    model: GPT2LMHeadModel = None,
) -> List[str]:
    """
    Sentence generation through choosing token guided by model confidence.
    Taking text input as prompts, tokens or embeddings, if more than one input a time, priority should follow: embeddings > tokens > prompts.
    Args:
        prompts: str, prompts for generated sentence
        tokens: tensor with shape of (b, n_seq), device = model.device, dtype = int64
        embeddings: tensor with shape of (b, n_seq, lm_hidden_size), device = model.device, dtype = float16/float32 (from clip encoder/gpt2 encoder)
        max_len: int, the maximum length of generated sentence (without considering the length of prompts/tokens/embeddings)
        end_of_sentence: str, early stopping once generated word is equal to end_of_sentence
        tokenizer: transforming word/sentence to indice/list and vice versa, i.e., str -> List[int64] or List[int64] -> str
        model: language model (taking input as either tokens or embeddings)
    Return:
        list[str] for generated sentence when batch size is greater than 1 (i.e., len(list) = batch_size), and string when batch size is equal to 1 
    """
    model.eval()
    device = model.device

    # tokenizing end of sentence, when the length of eos tokens is greater than 1, setting the first token of eos tokens as eos token
    eos = tokenizer.encode(end_of_sentence)[-1]

    # prefix should transform into word embeddings so that sentence generation is capable of processing input of prompts, tokens or embeddings unifiedly
    # priority: embeddings > tokens > prompts
    if embeddings is not None:
        generating = embeddings # (b, n_seq, lm_hidden_size)
    else:
        if tokens is None:
            tokens = torch.tensor(tokenizer.encode(prompts))  # (n_seq), tokenizing prompts
            tokens = tokens.unsqueeze(dim = 0).to(device)     # (b(=1), n_seq), adding batch dimension
        generating = word_embed(model, tokens)
        # generating = model.transformer.wte(tokens)            # (b, n_seq, lm_hidden_size), transforming to word embeddings
    generating = generating.float()                           # (b, n_seq, lm_hidden_size)
    assert generating.dim() == 3, 'The dimension of prompts should equal to 3!'
    
    b = generating.shape[0] 
    # past_key_values = None
    inputs_opt = generating
    
    use_nucleus_sampling = False
    num_beams=beam_width
    max_length=max_len
    min_length=1
    top_p=0.9
    repetition_penalty=1.0
    length_penalty=1.0
    num_captions=1
    temperature=1

    if use_nucleus_sampling:
        query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
        num_beams = 1
    else:
        query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)
    
    atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(inputs_opt.device)
    
    prompt = tokenizer.eos_token + prompts if prompts else tokenizer.eos_token
    prompt = [prompt] * b
    opt_tokens = tokenizer(prompt,  add_special_tokens=False, return_tensors="pt").to(embeddings.device)
    input_ids = opt_tokens.input_ids
    attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
    # import pdb
    # pdb.set_trace()
    
    outputs = model.generate(
        input_ids=input_ids,
        query_embeds=query_embeds.type(model.dtype),
        attention_mask=attention_mask,
        do_sample=use_nucleus_sampling,
        top_p=top_p,
        temperature=temperature,
        num_beams=num_beams,
        max_new_tokens=max_length,
        min_length=min_length,
        eos_token_id= eos,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        num_return_sequences=num_captions,
        )
    
    output_text = tokenizer.batch_decode(outputs[:, :], skip_special_tokens=True)
    output_text = [text.strip() for text in output_text]
    print(output_text)
    return output_text


@torch.no_grad()
def greedy_search(
    prompts: Optional[str] = None,
    tokens: Optional[torch.Tensor] = None,
    embeddings: Optional[torch.Tensor] = None,
    max_len: int = 64,
    end_of_sentences: List = [".", " ."],
    tokenizer: GPT2Tokenizer = None,
    model: GPT2LMHeadModel = None
) -> List[str]:
    """
    Sentence generation through choosing token guided by model confidence.
    Taking text input as prompts, tokens or embeddings, if more than one input a time, priority should follow: embeddings > tokens > prompts.
    Args:
        prompts: str, prompts for generated sentence
        tokens: tensor with shape of (b, n_seq), device = model.device, dtype = int64
        embeddings: tensor with shape of (b, n_seq, lm_hidden_size), device = model.device, dtype = float16/float32 (from clip encoder/gpt2 encoder)
        max_len: int, the maximum length of generated sentence (without considering the length of prompts/tokens/embeddings)
        end_of_sentence: str, early stopping once generated word is equal to end_of_sentence
        tokenizer: transforming word/sentence to indice/list and vice versa, i.e., str -> List[int64] or List[int64] -> str
        model: language model (taking input as either tokens or embeddings)
    Return:
        list[str] for generated sentence when batch size is greater than 1 (i.e., len(list) = batch_size), and string when batch size is equal to 1 
    """
    model.eval()
    device = model.device

    # tokenizing end of sentence, when the length of eos tokens is greater than 1, setting the first token of eos tokens as eos token
    eos = [tokenizer.encode(end_of_sentence)[-1] for end_of_sentence in end_of_sentences]

    # prefix should transform into word embeddings so that sentence generation is capable of processing input of prompts, tokens or embeddings unifiedly
    # priority: embeddings > tokens > prompts
    if embeddings is not None:
        generating = embeddings # (b, n_seq, lm_hidden_size)
    else:
        if tokens is None:
            tokens = torch.tensor(tokenizer.encode(prompts))  # (n_seq), tokenizing prompts
            tokens = tokens.unsqueeze(dim = 0).to(device)     # (b(=1), n_seq), adding batch dimension
        generating = word_embed(model, tokens)
        # generating = model.transformer.wte(tokens)            # (b, n_seq, lm_hidden_size), transforming to word embeddings
    generating = generating.float()                           # (b, n_seq, lm_hidden_size)
    assert generating.dim() == 3, 'The dimension of prompts should equal to 3!'
    
    b = generating.shape[0] 
    past_key_values = None
    for step in range(max_len):
        # generating initial states of language model
        if step == 0:
            outputs = model(inputs_embeds = generating.type(model.dtype), past_key_values = past_key_values, use_cache = True)
            next_token_logits = outputs.logits[:, -1, :]   # (b, n_seq, vocal_size) -> (b, vocal_size), logits of the last token
            past_key_values = outputs.past_key_values      # Tuple[Tuple[(b, h, n_seq, lm_hidden_size/h)]], layers -> (key, value) -> torch.tensor

        next_token = torch.argmax(next_token_logits, dim = -1, keepdim = True) # (b, 1)
        next_embedding = word_embed(model, next_token)                     # (b, 1, lm_hidden_size)
        # next_embedding = model.transformer.wte(next_token)                     # (b, 1, lm_hidden_size)
        outputs = model(inputs_embeds = next_embedding.type(model.dtype), past_key_values = past_key_values, use_cache = True)
        next_token_logits = outputs.logits[:, -1, :]           # (b, 1, vocal_size) -> (b, vocal_size)
        past_key_values = outputs.past_key_values              # Tuple[Tuple[(b, h, n_seq + 1, lm_hidden_size/h)]]

        # updating tokens
        if tokens is None:
            tokens = next_token
        else:
            tokens = torch.cat((tokens, next_token), dim = 1)                                                   # (b, n_seq + 1)

        # whether to stop early according to the end of sentence, only working when batch size is equal to 1
        if b == 1 and next_token.item() in eos:
            new_tokens = tokens.squeeze(dim = 0).tolist()
            sentence = tokenizer.decode(new_tokens)
            return sentence
        
    # tokens: (1/b, n_seq + max_len) where n_seq refers to the length of inputs tokens or prompts 
    # torch.tensor(1/b, n_seq + max_Len) -> str/list[str]
    sentence = []
    if b == 1:
        new_tokens = tokens.squeeze(dim = 0).tolist()
        sentence = tokenizer.decode(new_tokens)
    else:
        for temp_tokens in tokens:
            for i in range(len(temp_tokens)):
                if temp_tokens[i].item() in eos:
                    break
            new_tokens = temp_tokens[:i + 1].tolist()
            sentence.append(tokenizer.decode(new_tokens))
    return sentence

def beam_search(
    prompts: Optional[str] = None,
    tokens: Optional[torch.Tensor] = None,
    embeddings: Optional[torch.Tensor] = None,
    temperature = 1.0,
    max_len: int = 64,
    beam_width: int = 5,
    end_of_sentences: List = [".", " ."],
    tokenizer: GPT2Tokenizer = None,
    model: GPT2LMHeadModel = None
) -> List[str]:
    """
    Sentence generation through choosing token guided by model confidence.
    Taking text input as prompts, tokens or embeddings, if more than one input a time, priority should follow: embeddings > tokens > prompts.
    Args:
        prompts: str, prompts for generated sentence
        tokens: tensor with shape of (b, n_seq), device = model.device, dtype = int64
        embeddings: tensor with shape of (b, n_seq, lm_hidden_size), device = model.device, dtype = float16/float32 (from clip encoder/gpt2 encoder)
        max_len: int, the maximum length of generated sentence (without considering the length of prompts/tokens/embeddings)
        beam_width: the width of beam
        end_of_sentence: str, early stopping once generated word is equal to end_of_sentence
        tokenizer: transforming word/sentence to indice/list and vice versa, i.e., str -> List[int64] or List[int64] -> str
        model: language model (taking input as either tokens or embeddings)
    Return:
        list[str] for generated sentence when batch size is greater than 1 (i.e., len(list) = batch_size), and string when batch size is equal to 1 
    """
    model.eval()
    device = model.device

    # tokenizing end of sentence, when the length of eos tokens is greater than 1, setting the first token of eos tokens as eos token
    eos = [tokenizer.encode(end_of_sentence)[-1] for end_of_sentence in end_of_sentences]
    scores = None
    seq_lengths = torch.ones(beam_width, device = device)
    is_stopped = torch.zeros(beam_width, device = device, dtype=torch.bool)
    # prefix should transform into word embeddings so that sentence generation is capable of processing input of prompts, tokens or embeddings unifiedly
    # priority: embeddings > tokens > prompts
    if embeddings is not None:
        generated = embeddings # (b, n_seq, lm_hidden_size)
    else:
        if tokens is None:
            tokens = torch.tensor(tokenizer.encode(prompts))  # (n_seq), tokenizing prompts
            tokens = tokens.unsqueeze(dim = 0).to(device)     # (b(=1), n_seq), adding batch dimension
        generated = word_embed(model, tokens)
        # generated = model.transformer.wte(tokens)             # (b, n_seq, lm_hidden_size), transforming to word embeddings
    generated = generated.float()                             # (b, n_seq, lm_hidden_size)
    assert generated.dim() == 3, 'The dimension of prompts should equal to 3!'


    for i in range(max_len):
        outputs = model(inputs_embeds=generated.type(model.dtype))
        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits = logits.softmax(-1).log()
        if scores is None:
            scores, next_tokens = logits.topk(beam_width, -1)
            generated = generated.expand(beam_width, *generated.shape[1:])
            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
            if tokens is None:
                tokens = next_tokens
            else:
                tokens = tokens.expand(beam_width, *tokens.shape[1:])
                tokens = torch.cat((tokens, next_tokens), dim=1)
        else:
            logits[is_stopped] = -float(np.inf)
            logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + logits
            seq_lengths[~is_stopped] += 1
            scores_sum_average = scores_sum / seq_lengths[:, None]
            scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_width, -1)
            # next_tokens_source = torch.floor(torch.div(next_tokens, scores_sum.shape[1])).long()
            next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode = 'trunc')
            seq_lengths = seq_lengths[next_tokens_source]
            next_tokens = next_tokens % scores_sum.shape[1]
            next_tokens = next_tokens.unsqueeze(1)
            tokens = tokens[next_tokens_source]
            tokens = torch.cat((tokens, next_tokens), dim=1)
            generated = generated[next_tokens_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_tokens_source]
        next_token_embed = word_embed(model, next_tokens.squeeze()).view(generated.shape[0], 1, -1)
        # next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
        generated = torch.cat((generated, next_token_embed), dim=1)
        assert len(eos) == 2 # hack
        is_stopped = is_stopped + (next_tokens.eq(eos[0]) | next_tokens.eq(eos[1])).squeeze()
        if is_stopped.all():
            break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]

    return output_texts

def word_embed(gpt, caption_tokens):
    if hasattr(gpt, 'transformer'):
        embedding_text = gpt.transformer.wte(caption_tokens)
    elif hasattr(gpt, 'model'):
        embedding_text = gpt.model.decoder.embed_tokens(caption_tokens)
    return embedding_text
        
@torch.no_grad()
def contrastive_search(
    prompts: Optional[str] = None,
    tokens: Optional[torch.Tensor] = None,
    embeddings: Optional[torch.Tensor] = None,
    alpha: float = 0.1,
    top_k: int = 48,
    max_len: int = 64,
    end_of_sentence: str = '.',
    tokenizer: GPT2Tokenizer = None,
    model: GPT2LMHeadModel =  None
) -> List[str]:
    """
    Sentence generation through choosing token guided by model confidence, degeneration penality.
    Taking text input as prompts, tokens or embeddings, if more than one input a time, priority should follow: embeddings > tokens > prompts.
    Args:
        prompts: str, prompts for generated sentence
        tokens: tensor with shape of (b, n_seq), device = model.device, dtype = int64
        embeddings: tensor with shape of (b, n_seq, lm_hidden_size), device = model.device, dtype = float16/float32 (from clip encoder/gpt2 encoder)
        alpha: float from 0.0 to 1.0, controlling the strength of degenration penalty (i.e., avoiding repeat)
        top_k: int, generating k candidate tokens each time step in next token predicition (i.e., next token will be selected from the top k candidates) 
        max_len: int, the maximum length of generated sentence (without considering the length of prompts/tokens/embeddings)
        end_of_sentence: str, early stopping once generated word is equal to end_of_sentence
        tokenizer: transforming word/sentence to indice/list and vice versa, i.e., str -> List[int64] or List[int64] -> str
        model: language model (taking input as either tokens or embeddings)
    Return:
        list[str] for generated sentence when batch size is greater than 1 (i.e., len(list) = batch_size), and string when batch size is equal to 1 
    """
    model.eval()
    device = model.device

    # tokenizing end of sentence, when the length of eos tokens is greater than 1, setting the first token of eos tokens as eos token
    eos = tokenizer.encode(end_of_sentence)[0]

    # prefix should transform into word embeddings so that sentence generation is capable of processing input of prompts, tokens or embeddings unifiedly
    # priority: embeddings > tokens > prompts
    if embeddings is not None:
        generating = embeddings # (b, n_seq, lm_hidden_size)
    else:
        if tokens is None:
            tokens = torch.tensor(tokenizer.encode(prompts))  # (n_seq), tokenizing prompts
            tokens = tokens.unsqueeze(dim = 0).to(device)     # (b(=1), n_seq), adding batch dimension
        generated = word_embed(model, tokens)
        # generating = model.transformer.wte(tokens)            # (b, n_seq, lm_hidden_size), transforming to word embeddings
    generating = generating.float()                           # (b, n_seq, lm_hidden_size)
    assert generating.dim() == 3, 'The dimension of prompts should equal to 3!'

    past_key_values = None
    for step in range(max_len):
        # generating the initial states of model
        if step == 0:
            outputs = model(inputs_embeds = generating, past_key_values = past_key_values, use_cache = True, output_hidden_states = True)
            next_token_logits = outputs.logits[:, -1, :]   # (b, n_seq, vocal_size) -> (b, vocal_size), logits of the last token
            past_key_values = outputs.past_key_values      # Tuple[Tuple[(b, h, n_seq, lm_hidden_size/h)]], layers -> (key, value) -> torch.tensor
            past_hidden_states = outputs.hidden_states[-1] # Tuple[(b, n_seq, lm_hidden_size)] -> (b, n_seq, lm_hidden_size) (i.e., hidden state of last layer)

        # selecting top k candidates and their probability from next_tokens_logits 
        b, n_seq, lm_hidden_size = past_hidden_states.size()
        next_token_probs = F.softmax(next_token_logits, dim = -1)                    # (b, vocal_size)
        _, top_k_indices = torch.topk(next_token_logits, dim = -1, k = top_k)        # (b, k), the indices for top k candidates (i.e., tokens)
        top_k_probs = torch.gather(next_token_probs, dim = 1, index = top_k_indices) # (b, k), the probability for top k candidates

        # transformering b*k tokens to embeddings and processing past_key_values to compute simultaneously for k tokens
        top_k_embeddings = model.transformer.wte(top_k_indices.view(-1, 1))          # (b*k, 1, lm_hidden_size)
        past_key_values = reshape_from_past_key_values(past_key_values, top_k)       # Tuple[Tuple[(b*k, h, n_seq, lm_hidden_size/h)]]
        # computing hidden state of next token (b * top_k in total)
        outputs = model(inputs_embeds = top_k_embeddings, past_key_values = past_key_values, use_cache = True, output_hidden_states = True)
        logits = outputs.logits[:, -1, :]                     # (b*k, 1, vocal_size) -> (b*k, vocal_size)
        past_key_values = outputs.past_key_values             # Tuple[Tuple[(b*k, h, n_seq + 1, lm_hidden_size/h)]]
        next_hidden_state = outputs.hidden_states[-1]         # Tuple[(b*k, 1, lm_hidden_size)] -> (b*k, 1, lm_hidden_size)
        context_hidden_states = past_hidden_states.unsqueeze(dim = 1).expand(-1, top_k, -1, -1).reshape(b*top_k, n_seq, lm_hidden_size) # (b*k, n_seq, lm_hidden_size)

        # selecting next token within top k candidates for each sentence
        selected_max_prob_indices = ranking_and_selecting(context_hidden_states, next_hidden_state, top_k_probs, alpha, top_k)          # (b)
        
        # updating next_token_logits, past key-values and last hidden state
        logits = torch.stack(torch.split(logits, top_k), dim = 0)                                               # (b, k, vocal_size)
        next_token_logits = logits[range(b), selected_max_prob_indices, :]                                      # (b, vocal_size)
        past_key_values = reshape_to_past_key_values(past_key_values, selected_max_prob_indices, top_k)         # (b, h, n_seq + 1, lm_hidden_size/h)
        next_hidden_state = torch.stack(torch.split(next_hidden_state.squeeze(dim = 1), top_k), dim = 0)        # (b, k, lm_hidden_size)
        next_hidden_state = next_hidden_state[range(b), selected_max_prob_indices, :]                           # (b, lm_hidden_size)
        past_hidden_states = torch.cat([past_hidden_states, next_hidden_state.unsqueeze(dim = 1)], dim=1)       # [b, n_seq + 1, lm_hidden_size]

        # computing next token and saving it
        next_token = top_k_indices[range(b), selected_max_prob_indices].unsqueeze(dim = -1)                     # (b, 1)
        if tokens is None:
            tokens = next_token
        else:
            tokens = torch.cat((tokens, next_token), dim = 1)                                                   # (b, n_seq + 1)

        # whether to stop early according to the end of sentence, only working when batch size is equal to 1
        if b == 1 and next_token.item() == eos:
            new_tokens = tokens.squeeze(dim = 0).tolist()
            sentence = tokenizer.decode(new_tokens)
            return sentence
        
    # tokens: (1/b, n_seq + max_len) where n_seq refers to the length of inputs tokens or prompts 
    # torch.tensor(1/b, n_seq + max_Len) -> str/list[str]
    sentence = []
    if b == 1:
        new_tokens = tokens.squeeze(dim = 0).tolist()
        sentence = tokenizer.decode(new_tokens)
    else:
        for temp_tokens in tokens:
            for i in range(len(temp_tokens)):
                if temp_tokens[i].item() == eos:
                    break
            new_tokens = temp_tokens[:i + 1].tolist()
            sentence.append(tokenizer.decode(new_tokens))
    return sentence

@torch.no_grad()
def magic_search(
    prompts: Optional[str] = None,
    tokens: Optional[torch.Tensor] = None,
    embeddings: Optional[torch.Tensor] = None,
    image_path: Optional[str] = None,
    images_feature: Optional[torch.Tensor] = None,
    alpha: float = 0.1,
    beta: float = 2.0,
    top_k: int = 48,
    max_len: int = 64,
    clip_text_max_len: int = 60,
    end_of_sentence: str = '.',
    tokenizer: GPT2Tokenizer = None,
    model: GPT2LMHeadModel = None
) -> List[str]:
    """
    Sentence generation through choosing token guided by model confidence, degeneration penality and image at each time step.
    Taking text input as prompts, tokens or embeddings, if more than one input a time, priority should follow: embeddings > tokens > prompts.
    Taking image input as images_path or images_feature, if more than one input a time, priority should follow images_feature > image_path.
    Args:
        prompts: str, prompts for generated sentence
        tokens: tensor with shape of (b, n_seq), device = model.device, dtype = int64
        embeddings: tensor with shape of (b, n_seq, lm_hidden_size), device = model.device, dtype = float16/float32 (from clip encoder/gpt2 encoder)
        image_path: str, the path of a single image
        images_feature: tensor with shape of (b, clip_hidden_size), device = model.device, dtype = float32
        alpha: float from 0.0 to 1.0, controlling the strength of degenration penalty (i.e., avoiding repeat)
        beta: float, controlling image-guided strength
        top_k: int, generating k candidate tokens each time step in next token predicition (i.e., next token will be selected from the top k candidates) 
        max_len: int, the maximum length of generated sentence (without considering the length of prompts/tokens/embeddings)
        clip_text_max_len: int, the maximum length of clip textual encoder
        end_of_sentence: str, early stopping once generated word is equal to end_of_sentence
        tokenizer: transforming word/sentence to indice/list and vice versa, i.e., str -> List[int64] or List[int64] -> str
        model: language model (taking input as either tokens or embeddings)
    Return:
        list[str] for generated sentence when batch size is greater than 1 (i.e., len(list) = batch_size), and string when batch size is equal to 1 
    """
    model.eval()
    device = model.device

    # tokenizing end of sentence, when the length of eos tokens is greater than 1, setting the first token of eos tokens as eos token
    eos = tokenizer.encode(end_of_sentence)[0]

    # prefix should transform into word embeddings so that sentence generation is capable of processing input of prompts, tokens or embeddings unifiedly
    # priority: embeddings > tokens > prompts
    if embeddings is not None:
        generating = embeddings # (b, n_seq, lm_hidden_size)
    else:
        if tokens is None:
            tokens = torch.tensor(tokenizer.encode(prompts))  # (n_seq), tokenizing prompts
            tokens = tokens.unsqueeze(dim = 0).to(device)     # (b(=1), n_seq), adding batch dimension
        generating = model.transformer.wte(tokens)            # (b, n_seq, lm_hidden_size), transforming to word embeddings
    generating = generating.float()                           # (b, n_seq, lm_hidden_size)
    assert generating.dim() == 3, 'The dimension of prompts should equal to 3!'

    # generating image feature using clip visual encoder
    # note that the dtype of feature from clip visual encoder is equal to float16, transforming it into float32
    # priority: images_feature > image_path
    clip_model, preprocess = clip.load('ViT-B/32', device = device)
    clip_model.eval()
    if images_feature is None:
        image = preprocess(Image.open(image_path)).unsqueeze(dim = 0).to(device)  # (b(=1), 3, 224, 224)
        images_feature = clip_model.encode_image(image)                           # (b, clip_hidden_size)
    images_feature = images_feature.float()                                       # (b, clip_hidden_size)
    assert images_feature.dim() == 2, 'The dimension of images feature should equal to 2!'
    assert images_feature.shape[0] == generating.shape[0], 'The number of images should be equal to the number of prompts/tokens/embeddings!'

    past_key_values = None
    tokens_generated = None
    for step in range(max_len):
        # generating the initial states of model
        if step == 0:
            outputs = model(inputs_embeds = generating, past_key_values = past_key_values, use_cache = True, output_hidden_states = True)
            next_token_logits = outputs.logits[:, -1, :]   # (b, n_seq, vocal_size) -> (b, vocal_size), logits of the last token
            past_key_values = outputs.past_key_values      # Tuple[Tuple[(b, h, n_seq, lm_hidden_size/h)]], layers -> (key, value) -> torch.tensor
            past_hidden_states = outputs.hidden_states[-1] # Tuple[(b, n_seq, lm_hidden_size)] -> (b, n_seq, lm_hidden_size) (i.e., hidden state of last layer)

        # selecting top k candidates and their probability from next_tokens_logits 
        b, n_seq, lm_hidden_size = past_hidden_states.size()
        next_token_probs = F.softmax(next_token_logits, dim = -1)                    # (b, vocal_size)
        _, top_k_indices = torch.topk(next_token_logits, dim = -1, k = top_k)        # (b, k), the indices for top k candidates (i.e., tokens)
        top_k_probs = torch.gather(next_token_probs, dim = 1, index = top_k_indices) # (b, k), the probability for top k candidates

        # computing similarity between image and sentence (b * k in total)
        image_sentence_score = image_sentence_similarity(tokens_generated, top_k_indices, images_feature, top_k, clip_text_max_len, tokenizer, clip_model) # (b, k)

        # transformering b*k tokens to embeddings and processing past_key_values to compute simultaneously for k tokens
        top_k_embeddings = model.transformer.wte(top_k_indices.view(-1, 1))          # (b*k, 1, lm_hidden_size)
        past_key_values = reshape_from_past_key_values(past_key_values, top_k)       # Tuple[Tuple[(b*k, h, n_seq, lm_hidden_size/h)]]
        # computing hidden state of next token (b * top_k in total)
        outputs = model(inputs_embeds = top_k_embeddings, past_key_values = past_key_values, use_cache = True, output_hidden_states = True)
        logits = outputs.logits[:, -1, :]                     # (b*k, 1, vocal_size) -> (b*k, vocal_size)
        past_key_values = outputs.past_key_values             # Tuple[Tuple[(b*k, h, n_seq + 1, lm_hidden_size/h)]]
        next_hidden_state = outputs.hidden_states[-1]         # Tuple[(b*k, 1, lm_hidden_size)] -> (b*k, 1, lm_hidden_size)
        context_hidden_states = past_hidden_states.unsqueeze(dim = 1).expand(-1, top_k, -1, -1).reshape(b*top_k, n_seq, lm_hidden_size) # (b*k, n_seq, lm_hidden_size)

        # selecting next token within top k candidates for each sentence
        selected_max_prob_indices = ranking_and_selecting(context_hidden_states, next_hidden_state, top_k_probs, alpha, top_k, beta, image_sentence_score) # (b)
        
        # updating next_token_logits, past key-values and last hidden state
        logits = torch.stack(torch.split(logits, top_k), dim = 0)                                               # (b, k, vocal_size)
        next_token_logits = logits[range(b), selected_max_prob_indices, :]                                      # (b, vocal_size)
        past_key_values = reshape_to_past_key_values(past_key_values, selected_max_prob_indices, top_k)         # (b, h, n_seq + 1, lm_hidden_size/h)
        next_hidden_state = torch.stack(torch.split(next_hidden_state.squeeze(dim = 1), top_k), dim = 0)        # (b, k, lm_hidden_size)
        next_hidden_state = next_hidden_state[range(b), selected_max_prob_indices, :]                           # (b, lm_hidden_size)
        past_hidden_states = torch.cat([past_hidden_states, next_hidden_state.unsqueeze(dim = 1)], dim=1)       # [b, n_seq + 1, lm_hidden_size]

        # computing next token and saving it
        next_token = top_k_indices[range(b), selected_max_prob_indices].unsqueeze(dim = -1)                     # (b, 1)
        if tokens is None:
            tokens = next_token
            tokens_generated = next_token
        else:
            if tokens_generated is None:
                tokens_generated = next_token
            else:
                tokens_generated = torch.cat((tokens_generated, next_token), dim = 1)
            tokens = torch.cat((tokens, next_token), dim = 1)                                                   # (b, n_seq + 1)

        # whether to stop early according to the end of sentence, only working when batch size is equal to 1
        if b == 1 and next_token.item() == eos:
            new_tokens = tokens.squeeze(dim = 0).tolist()
            sentence = tokenizer.decode(new_tokens)
            return sentence
        
    # tokens: (1/b, n_seq + max_len) where n_seq refers to the length of inputs tokens or prompts 
    # torch.tensor(1/b, n_seq + max_Len) -> str/list[str]
    sentence = []
    if b == 1:
        new_tokens = tokens.squeeze(dim = 0).tolist()
        sentence = tokenizer.decode(new_tokens)
    else:
        for temp_tokens in tokens:
            for i in range(len(temp_tokens)):
                if temp_tokens[i].item() == eos:
                    break
            new_tokens = temp_tokens[:i + 1].tolist()
            sentence.append(tokenizer.decode(new_tokens))
    return sentence

def image_sentence_similarity(
    tokens_generated: torch.Tensor,
    top_k_indices: torch.Tensor,
    images_feature: torch.Tensor,
    top_k: int,
    clip_text_max_len: int,
    tokenizer: GPT2Tokenizer,
    clip_model: clip
) -> torch.Tensor:
    """
    Args:
        tokens_generated: tensor with shape of (b, n_seq), the sentence generated (without considering the prompts)
        top_k_indices: tensor with shape of (b, top_k), the top k candidates for each sentence
        images_feature: tensor with shape of (b, clip_hidden_size), image feature encoded by clip
        top_k: int, k candidates
        clip_text_max_len: int, the maximum length of clip textual encoder
        tokenizer: transforming word/sentence to indice/list and vice versa
        clip_model: pre-trained clip model which encodes image or image to embeddings with dtype of float16 (transforming to float32)
       
    Return:
        image-sentence similarity score with shape of (b, k), i.e., for each sentence (b in total), returning top k tokens similarity with image 
    """
    device = top_k_indices.device

    # obtaining tokens of generated (b sentences and k tokens for each sentence, i.e., b * k sentences in total)
    if tokens_generated is None:
        temp_tokens = top_k_indices.view(-1).unsqueeze(dim = 1)                                                    # (b*k, n_seq + 1), where n_seq = 0
    else:
        b, n = tokens_generated.size()
        tokens_generated = tokens_generated.unsqueeze(dim = 1).expand(-1, top_k, -1).reshape(b*top_k, n)           # (b*k, n_seq)
        top_k_indices = top_k_indices.view(-1).unsqueeze(dim = 1)                                                  # (b*k, 1)
        temp_tokens = torch.cat([tokens_generated, top_k_indices], dim = 1)                                        # (b*k, n_seq + 1)

    # converting to sentence
    sentences = []
    for temp_token in temp_tokens:
        # taking the latest clip_text_max_len tokens when tokens length is greater than clip_text_max_len
        sentence = tokenizer.decode(temp_token[-clip_text_max_len:].to('cpu').tolist())
        sentences.append(sentence)                                                                                 # len(sentences) = b*k

    # converting to text tokens and embeddings of clip
    clip_tokens = clip.tokenize(sentences).to(device)                                                              # (b*k, n_seq)
    clip_embeddings = clip_model.encode_text(clip_tokens)                                                          # (b*k, clip_hidden_size)
    clip_embeddings = torch.stack(torch.split(clip_embeddings, top_k), dim = 0).float()                            # (b, k, clip_hidden_size)
    
    # computing similarity score
    images_feature = images_feature.unsqueeze(dim = 1)                                                             # (b, 1, clip_hidden_size)
    clip_embeddings = clip_embeddings / clip_embeddings.norm(dim = -1, keepdim = True)                             # (b, k, clip_hidden_size)
    images_feature = images_feature / images_feature.norm(dim = -1, keepdim = True)                                # (b, 1, clip_hidden_size)
    scaling = clip_model.logit_scale.exp()
    score = torch.matmul(clip_embeddings, images_feature.transpose(1, 2)).squeeze(dim = 2) * scaling               # (b, k)

    return F.softmax(score, dim = -1)

def reshape_from_past_key_values(past_key_values: Tuple[Tuple[torch.Tensor]], top_k: int) -> Tuple[Tuple[torch.Tensor]]:
    """
    To compute top k candidates simultaneously for each sentence in a batch, duplicating k times for each sentence.
    Args:
        past_key_values: Tuple[Tuple[(b, h, n_seq, lm_hidden_size/h)]], the first tuple refers to layers and the second tuple refers to key-value pair
        top_k: int, k candidates
    Return:
        Tuple[Tuple[(b*k, h, n_seq, lm_hidden_size/h)]]
    """
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            b, h, n, d = item.size() # d = lm_hidden_size/h
            # duplicating k times for each sentence in a batch, the only difference between each k repeated sample is the candidate waiting to concatenate
            item = item.unsqueeze(dim = 1).expand(-1, top_k, -1, -1, -1).reshape(b*top_k, h, n, d) # (b*k, h, n_seq, lm_hidden_size/h)
            items.append(item)
        new_key_values.append(items)
    return new_key_values

def reshape_to_past_key_values(past_key_values: Tuple[Tuple[torch.Tensor]], selected_max_prob_indices: torch.Tensor, top_k: int) -> Tuple[Tuple[torch.Tensor]]:
    """
    Args:
        past_key_values: Tuple[Tuple[(b*k, h, n_seq + 1, lm_hidden_size/h)]]
        selected_max_prob_indices: tensor with shape of (b), indices of maximum probability in k candidates
        top_k: int, k candidates
    Return:
        Tuple[Tuple[(b, h, n_seq + 1, lm_hidden_size/h)]]
    """
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bk = item.shape[0]
            b = int(bk//top_k)
            item = torch.stack(torch.split(item, top_k), dim = 0)     # (b, k, h, n_seq + 1, lm_hidden_size/h)
            item = item[range(b), selected_max_prob_indices, :, :, :] # (b, h, n_seq + 1, lm_hidden_size/h)
            items.append(item)
        new_key_values.append(items)
    return new_key_values

def ranking_and_selecting(
    context_hidden_states: torch.Tensor,
    next_hidden_state: torch.Tensor,
    top_k_probs: torch.Tensor,
    alpha: float,
    top_k: int,
    beta: Optional[float] = None,
    image_sentence_score: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        context_hidden_states: tensor with shape of (b*k, n_seq, lm_hidden_size), the hidden state of each token in sentence before candidates (i.e. <t)
        next_hidden_state: tensor with shape of (b*k, 1, lm_hidden_size), the hidden state of next candidates (i.e. =t)
        top_k_probs: tensor with shape of (b, k), the probability of top k candidates
        alpha: float from 0.0 to 1.0, controlling the strength of degenration penalty (i.e., avoiding repeat)
        top_k: int, k candidates
        beta: float, controlling image-guided strength
        image_sentence_score: tensor with shape of (b, k) refers to the relevance between image and b * k sentences
    Return:
        (b), indices of maximum probability in top_k candidates for each sentence
    """
    # normalizing alongside dimension: feature (i.e., lm_hidden_size)
    # tensor.norm(), i.e., norm 2
    norm_context_hidden_states = context_hidden_states / context_hidden_states.norm(dim = -1, keepdim = True)        # (b*k, n_seq, lm_hidden_size)
    norm_next_hidden_state = next_hidden_state / next_hidden_state.norm(dim = -1, keepdim = True)                    # (b*k, 1, lm_hidden_size)
    # hidden state from next token should compute similarity with hidden state from past tokens to avoid degeeration
    cosine_matrix = torch.matmul(norm_context_hidden_states, norm_next_hidden_state.transpose(1, 2)).squeeze(-1)     # (b*k, n_seq)
    # selecting the maximum similar score for each sample (sentence * top k -> b*k in total), i.e., degeneration penalty term
    scores, _ = torch.max(cosine_matrix, dim = -1)                  # (b*k)
    # model confidence (i.e., maximum likelihood)
    top_k_probs = top_k_probs.view(-1)                              # (b*k)
    # image-guided score
    if image_sentence_score is not None:
        image_sentence_score = image_sentence_score.view(-1)        # (b*k)
    # re-computing scores by formulation: model confidence + degeneration penalty + image-sentence similarity
    if image_sentence_score is not None:
        scores = (1.0 - alpha) * top_k_probs - alpha * scores + beta * image_sentence_score                          # (b*k)
    else:
        scores = (1.0 - alpha) * top_k_probs - alpha * scores        # (b*k)
    # selecting next token from "top_k" next tokens for each sample 
    scores = torch.stack(torch.split(scores, top_k), dim = 0)        # (b, k)
    _, selected_max_prob_indices = scores.max(dim = -1)              # (b)
    return selected_max_prob_indices