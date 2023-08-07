import math
import torch
import random
from typing import List, Tuple, Union
        
def noise_injection(x, variance = 0.001, device = 'cuda:0') -> torch.Tensor:
    """
    Args:
        x: tensor with a shape of (batch_size, clip_hidden_size), prefix
        variance: the variance of noise
    Return:
        prefix with noise
    """
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    # normalization
    x = torch.nn.functional.normalize(x, dim = -1)
    # adding noise
    x = x + (torch.randn(x.shape, device = device) * std)

    return torch.nn.functional.normalize(x, dim = -1)

def entities_process(
    args,
    detected_entities: List[str],  # [man, dog, park]
    stopwords: List[str],
    people_vocabs: List[str],
    objects_vocabs: List[str],
) -> List[str]:
    process_entities = []
    for i in range(len(detected_entities)):
        if i >= args.max_num_of_entities: # There is no entity detected
            break
        detected_entity = detected_entities[i]                # processing the i-th entity
        if detected_entity in people_vocabs:                  # transforming all person concept (man, woman, kid, ...) to the same word 'person'
            detected_entity = 'person'
        elif len(detected_entity) > 1 and detected_entity not in stopwords and detected_entity in objects_vocabs: # only processing entities in visual genome
            pass
        else: # processing the next entities
            continue

        if args.random_mask:
            random_prob = random.random()
            if random_prob < args.prob_of_random_mask:         # mask
                pass
            else:                                              # remain
                process_entities.append(detected_entity)

        else: # entities with any process
            return detected_entities
    
    return process_entities

def compose_discrete_prompts(
    tokenizer,
    process_entities: List[str],
) -> torch.Tensor:

    prompt_head = 'There are'
    prompt_tail = ' in image.'

    if len(process_entities) == 0: # without entities
        discrete_prompt =  prompt_head + ' something' + prompt_tail
    else:
        discrete_prompt = ''
        for entity in process_entities: # gpt2 in transformer encoder ' ' + word into one token by default
            discrete_prompt += ' ' + entity + ','     # ' person, dog, park,'
        discrete_prompt = discrete_prompt[:-1]        # ' person, dog, park'
        discrete_prompt = prompt_head + discrete_prompt + prompt_tail # 'There are person, dog, park in image.'

    entities_tokens = torch.tensor(tokenizer.encode(discrete_prompt))   # (discrete_prompt_length, ) 

    return entities_tokens

def parse_entities(
    args,
    tokenizer,
    detected_entities: Tuple[str],      # [[man, dog, park, ...], len = batch size
    stopwords: List[str],
    people_vocabs: List[str],
    objects_vocabs: List[str],
) -> List[torch.Tensor]:
    # List[(n_seq1, ), (n_seq2, ), ...]

    discrete_tokens = []
    for idx in range(len(detected_entities)):
        # entities processing
        process_entities = entities_process(args, detected_entities[idx], stopwords, people_vocabs, objects_vocabs)
        process_entities = list(set(process_entities)) # list

        # tokenizing
        discrete_tokens.append(compose_discrete_prompts(tokenizer, process_entities))

    return discrete_tokens

def padding_captions(
    args,
    captions_tokens: torch.Tensor,   # (batch_size, caption_seq)
    masks: torch.Tensor,             # (batch_size, caption_seq)
    discrete_tokens: List[torch.Tensor] = None, # len = batch_size, [(n_seq1, ), (n_seq2, ), ...]
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, List]:
    """
    Return:
        captions_tokens:
        captions_tokens_for_loss:
        masks:
        hard_prompts_length: 
    """
    if discrete_tokens is None: # capdec
        masks = torch.cat((torch.ones(len(masks), args.continuous_prompt_length), masks), dim = -1) # (batch_size, continuous_prompt_length + caption_seq)
        captions_tokens_for_loss = torch.cat((torch.zeros((len(captions_tokens), args.continuous_prompt_length), dtype = torch.int64), captions_tokens), dim = -1) # (batch_size, continuous_prompt_length + caption_seq)
        captions_tokens_for_loss = torch.cat((captions_tokens_for_loss[:, 1:], torch.zeros((len(captions_tokens), 1), dtype = torch.int64)), dim = -1)
        return captions_tokens, captions_tokens_for_loss, masks

    else: # discrete tokens
        captions_tokens_with_hard_prompts = None
        captions_tokens_for_loss = None
        padding_masks = None
        hard_prompts_length = []
        max_length = 2 * args.max_num_of_entities - 1 + args.prompt_template_length + captions_tokens.shape[-1] # max length without soft prompt
        for i in range(len(discrete_tokens)):
            tokens = torch.cat((discrete_tokens[i], captions_tokens[i]))
            loss_tokens = torch.cat((torch.zeros((len(discrete_tokens[i])), dtype = torch.int64), captions_tokens[i]))
            padding = max_length - len(tokens)
            if padding > 0:
                tokens = torch.cat((tokens, torch.zeros((padding), dtype = torch.int64)))
                loss_tokens = torch.cat((loss_tokens, torch.zeros((padding), dtype = torch.int64)))
            tokens = tokens[:max_length].unsqueeze(dim = 0) # (1, max_length)
            if args.only_hard_prompt:
                loss_tokens = loss_tokens[:max_length]
            else:
                loss_tokens = torch.cat((torch.zeros((args.continuous_prompt_length), dtype = torch.int64), loss_tokens[:max_length]))
            loss_tokens = torch.cat((loss_tokens[1:], torch.zeros((1), dtype = torch.int64))).unsqueeze(dim = 0) # (1, max_length + continuous_prompt_length)
            if captions_tokens_with_hard_prompts is None:
                captions_tokens_with_hard_prompts = tokens
                captions_tokens_for_loss = loss_tokens
            else:
                captions_tokens_with_hard_prompts = torch.cat((captions_tokens_with_hard_prompts, tokens), dim = 0)
                captions_tokens_for_loss = torch.cat((captions_tokens_for_loss, loss_tokens), dim = 0)

            hard_prompts_length.append(len(discrete_tokens[i]))
            if padding > 0:
                if args.only_hard_prompt:
                    temp_masks = torch.cat((torch.ones(hard_prompts_length[-1]).float(), masks[i], torch.zeros(padding).float()))
                else:
                    temp_masks = torch.cat((torch.ones(hard_prompts_length[-1] + args.continuous_prompt_length).float(), masks[i], torch.zeros(padding).float()))
            else:
                if args.only_hard_prompt:
                    temp_masks = torch.cat((torch.ones(hard_prompts_length[-1]).float(), masks[i]))
                else:
                    temp_masks = torch.cat((torch.ones(hard_prompts_length[-1] + args.continuous_prompt_length).float(), masks[i]))
            if args.only_hard_prompt:
                temp_masks = temp_masks[:max_length].unsqueeze(dim = 0)
            else:
                temp_masks = temp_masks[:max_length + args.continuous_prompt_length].unsqueeze(dim = 0)
            if padding_masks is None:
                padding_masks = temp_masks
            else:
                padding_masks = torch.cat((padding_masks, temp_masks), dim = 0)
        return captions_tokens_with_hard_prompts, captions_tokens_for_loss, padding_masks, hard_prompts_length