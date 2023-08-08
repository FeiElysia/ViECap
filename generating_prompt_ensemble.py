import os
import clip
import torch
import pickle
from tqdm import tqdm
from typing import List
from load_annotations import load_entities_text

@torch.no_grad()
def generate_ensemble_prompt_embeddings(
    device: str,
    clip_type: str,
    entities: List[str],
    prompt_templates: List[str],
    outpath: str,
):
    if os.path.exists(outpath):
        with open(outpath, 'rb') as infile:
            embeddings = pickle.load(infile)
            return embeddings

    model, _ = clip.load(clip_type, device)
    model.eval()
    embeddings = []
    for entity in tqdm(entities):
        texts = [template.format(entity) for template in prompt_templates] # ['a picture of dog', 'photo of a dog', ...]
        tokens = clip.tokenize(texts).to(device)               # (len_of_template, 77)
        class_embeddings = model.encode_text(tokens).to('cpu') # (len_of_templates, clip_hidden_size)
        class_embeddings /= class_embeddings.norm(dim = -1, keepdim = True) # (len_of_templates, clip_hidden_size)
        class_embedding = class_embeddings.mean(dim = 0)       # (clip_hidden_size, ) 
        class_embedding /= class_embedding.norm()              # (clip_hidden_size, ) 
        embeddings.append(class_embedding)                     # [(clip_hidden_size, ), (clip_hidden_size, ), ...]
    embeddings = torch.stack(embeddings, dim = 0).to('cpu')
   
    with open(outpath, 'wb') as outfile:
        pickle.dump(embeddings, outfile)
    return embeddings

if __name__ == '__main__':

    # prompts from CLIP
    prompt_templates = [
        'itap of a {}.',
        'a bad photo of the {}.',
        'a origami {}.',
        'a photo of the large {}.',
        'a {} in a video game.',
        'art of the {}.',
        'a photo of the small {}.'
    ]

    entities = load_entities_text('vinvl_vgoi_entities', './annotations/vocabulary/vgcocooiobjects_v1_class2ind.json')
    # entities = load_entities_text('coco_entities', './annotations/vocabulary/coco_categories.json')
    # entities = load_entities_text('vinvl_vg_entities', './annotations/vocabulary/VG-SGG-dicts-vgoi6-clipped.json')
    # entities = load_entities_text('visual_genome_entities', './annotations/vocabulary/all_objects_attributes_relationships.pickle', all_entities = False)
    # entities = load_entities_text('open_image_entities', './annotations/vocabulary/oidv7-class-descriptions-boxable.csv')

    device = 'cuda:0'
    clip_type = 'ViT-B/32'
    clip_name = clip_type.replace('/', '')
    outpath = f'./annotations/vocabulary/vgoi_embeddings_{clip_name}_with_ensemble.pickle'
    # outpath = f'./annotations/vocabulary/coco_embeddings_{clip_name}_with_ensemble.pickle'
    # outpath = f'./annotations/vocabulary/vg_embeddings_{clip_name}_with_ensemble.pickle'
    # outpath = f'./annotations/vocabulary/visual_genome_embedding_{clip_name}_with_ensemble.pickle'
    # outpath = f'./annotations/vocabulary/open_image_embeddings_{clip_name}_with_ensemble.pickle'
    embeddings = generate_ensemble_prompt_embeddings(device, clip_type, entities, prompt_templates, outpath)

    print(entities[:10], len(entities))
    print(embeddings.size(), embeddings.dtype)