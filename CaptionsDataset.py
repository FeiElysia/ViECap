import clip
import torch
import pickle
import random
from typing import Tuple
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import parse_entities, padding_captions
from load_annotations import load_entities_text, load_stopwords

class CaptionsDataset(Dataset):

    def __init__(
        self,
        language_model: str = 'gpt2',
        max_num_of_entities: int = 5,
        using_clip_features: bool = False,
        path_of_datasets: str = './annotations/coco/coco_with_entities.pickle',
        debug: bool = False,
        args = None,
    ) -> None:
        """
        Args:
            language_model: the used tokenizer
            max_num_of_entities: the maximum number of entities (nouns) detected in a single sentence
            using_clip_features: loading pre-extracted clip text embeddings
            path_of_datasets: the path of training datasets, i.e., ./annotations/***/***
        """

        # initializing
        tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.using_clip_features = using_clip_features

        # the format of dataset (List[List[List, str]]):
        # [[['baby', 'giraffe', 'wall', 'zoo', 'environment'], 'A baby giraffe standing against a wall in a zoo like environment.'], ...]
        # or (using_clip_features = True):
        # [[['baby', 'giraffe', 'wall', 'zoo', 'environment'],
        #   A baby giraffe standing against a wall in a zoo like environment.',
        #   torch.tensor (size = (clip_hidden_size, ))], ...]
        with open(path_of_datasets, 'rb') as infile: # loading datasets
            captions_with_entities = pickle.load(infile)
        
        # low-data settings
        if args.few_shot_ratio < 1.0:
            random.shuffle(captions_with_entities)
            N = len(captions_with_entities) * args.few_shot_ratio
            captions_with_entities = captions_with_entities[: int(N)]
            
        if debug: # debug
            captions_with_entities = captions_with_entities[:500]

        captions_lm_lengths = []
        self.detected_entities = []
        self.captions = []
        self.captions_lm_tokens = []
        if self.using_clip_features:
            self.captions_clip_features = []
        else:
            self.captions_clip_tokens = []

        for caption_with_entities in captions_with_entities:
            if self.using_clip_features:
                temp_detected_entities, temp_caption, temp_clip_features = caption_with_entities
                self.captions_clip_features.append(temp_clip_features)                                          # dtype = float16, size = (clip_hidden_size, )
            else:
                temp_detected_entities, temp_caption = caption_with_entities
                self.captions_clip_tokens.append(clip.tokenize(temp_caption, truncate = True).squeeze(dim = 0)) # dtype = int32, size = (77, )
            self.captions.append(temp_caption)
            self.detected_entities.append(temp_detected_entities[:max_num_of_entities])
            
            # captions_lm_tokens are used for auto-regressive training, while captions_clip_tokens are accounted as image features during text-only training
            self.captions_lm_tokens.append(torch.tensor(tokenizer.encode(temp_caption), dtype = torch.int64)) # dtype = int64, size = (n_seq,)
            captions_lm_lengths.append(len(self.captions_lm_tokens[-1]))

        self.captions_lm_lengths = torch.tensor(captions_lm_lengths, dtype = torch.float32)
        self.max_length_per_caption = min(int(self.captions_lm_lengths.mean() + 10 * self.captions_lm_lengths.std()), int(self.captions_lm_lengths.max()))
        self.args = args
        self.tokenizer = tokenizer
        self.stopwords = load_stopwords()

        self.people_vocabs = ['people', 'person', 'man', 'men', 'woman', 'women', 'adult','boy', 'girl', 'kid', 'children', 'child', 'baby', 'guy', 'player', 'male', 'female', 'worker']
        self.objects_vocabs = load_entities_text(args.name_of_objects_vocabs, args.path_of_objects_vocabs, all_entities = False)
        print('Dataset Loading: {} successful. Max sentence length: {}'.format(path_of_datasets, self.max_length_per_caption))
        
    def __len__(self) -> int:
        # return the size of this dataset
        return len(self.captions)
    
    def pad_tokens(self, item: int) -> Tuple[torch.Tensor, ...]:
        """
        Return:
            tokens: tensor with a shape of (n_seq, ), padding 0 or truncating caption tokens to n_seq
            mask: tensor with a shape of (n_seq, ), valid texts for attention computing
        """
        tokens = self.captions_lm_tokens[item]        # caption tokens
        padding = self.max_length_per_caption - len(tokens)
        tokens = tokens[:self.max_length_per_caption] # truncating tokens to max_seq_len
        if padding > 0:                               # padding 0 to max_seq_len
            tokens = torch.cat((tokens, torch.zeros(padding, dtype = torch.int64) - 1))
        
        mask = tokens.ge(0)
        tokens[~mask] = 0 # when calculating loss, the position where idx = 0 should be ignored
        mask = mask.float()

        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        """
        Return:
            captions_clip: tensor with a shape of (clip_hidden_size, ) for clip features, or (77) clip tokens
            captions_gpt_tokens: tensor with a shape of (n_seq, ), the caption tokens encoded by language model
            masks: tensor with a shape of (n_seq, ), valid texts for attention computing
            discrete_tokens: tensor with a shape of (len_discrete_tokens + len_prompt_templates, )
        """
        caption_lm_tokens, mask = self.pad_tokens(item)

        if self.using_clip_features:
            captions_clip = self.captions_clip_features[item]
        else:
            captions_clip =  self.captions_clip_tokens[item]
        
        detected_entities = self.detected_entities[item]
        masks = mask
        captions_gpt_tokens = caption_lm_tokens
        
        discrete_tokens = None
        if self.args.using_hard_prompt:
            discrete_tokens = parse_entities(self.args, self.tokenizer, [detected_entities], self.stopwords, self.people_vocabs, self.objects_vocabs)[0]
        return  self.args, captions_clip, captions_gpt_tokens, masks, discrete_tokens


def collate(batch):
    batch_size = len(batch)
    args = batch[0][0]
    _, captions_clip, captions_gpt_tokens, masks, discrete_tokens = zip(*batch)
    captions_clip = torch.stack(captions_clip)
    captions_gpt_tokens = torch.stack(captions_gpt_tokens, dim=0)
    masks =  torch.stack(masks)

    hard_prompts_length = None
    if args.using_hard_prompt:
        captions_gpt_tokens, captions_tokens_for_loss, masks, hard_prompts_length = padding_captions(args, captions_gpt_tokens, masks, discrete_tokens)
    else:
        captions_gpt_tokens, captions_tokens_for_loss, masks = padding_captions(args, captions_gpt_tokens, masks)
    return captions_clip, captions_gpt_tokens, captions_tokens_for_loss, masks, hard_prompts_length