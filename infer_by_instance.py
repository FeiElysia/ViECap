import clip
import torch
import argparse
from PIL import Image
from ClipCap import ClipCaptionModel
from transformers import AutoTokenizer
from utils import compose_discrete_prompts
from load_annotations import load_entities_text
from search import greedy_search, beam_search, opt_search
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories

@torch.no_grad()
def main(args) -> None:
    # initializing
    device = args.device
    clip_name = args.clip_model.replace('/', '') 
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512

    # loading categories vocabulary for objects
    if args.name_of_entities_text == 'visual_genome_entities':
        entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/all_objects_attributes_relationships.pickle', not args.disable_all_entities)
        if args.prompt_ensemble: # loading ensemble embeddings
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/visual_genome_embedding_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/visual_genome_embedding_{clip_name}.pickle')
    elif args.name_of_entities_text == 'coco_entities':
        entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/coco_categories.json', not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/coco_embeddings_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/coco_embeddings_{clip_name}.pickle')
    elif args.name_of_entities_text == 'open_image_entities':
        entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/oidv7-class-descriptions-boxable.csv', not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/open_image_embeddings_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/open_image_embeddings_{clip_name}.pickle')
    elif args.name_of_entities_text == 'vinvl_vg_entities':
        entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/VG-SGG-dicts-vgoi6-clipped.json', not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vg_embeddings_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vg_embeddings_{clip_name}.pickle')
    elif args.name_of_entities_text == 'vinvl_vgoi_entities':
        entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/vgcocooiobjects_v1_class2ind.json', not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vgoi_embeddings_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vgoi_embeddings_{clip_name}.pickle')
    else:
        print('The entities text should be input correctly!')
        return
    
    # loading model
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    model = ClipCaptionModel(args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, gpt_type = args.language_model)
    model.load_state_dict(torch.load(args.weight_path, map_location = device), strict = False)
    model.to(device)
    encoder, preprocess = clip.load(args.clip_model, device = device)
   
    image = preprocess(Image.open(args.image_path)).unsqueeze(dim = 0).to(device)
    image_features = encoder.encode_image(image).float()
    image_features /= image_features.norm(2, dim = -1, keepdim = True)
    continuous_embeddings = model.mapping_network(image_features).view(-1, args.continuous_prompt_length, model.gpt_hidden_size)
    if args.using_hard_prompt:
        logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
        detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold) # List[List[]], [[category1, category2, ...], [], ...]
        detected_objects = detected_objects[0] # infering single image -> List[category1, category2, ...]
        discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)

        discrete_embeddings = model.word_embed(discrete_tokens)
        if args.only_hard_prompt:
            embeddings = discrete_embeddings
        elif args.soft_prompt_first:
            embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim = 1)
        else:
            embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim = 1)
    else:
        embeddings = continuous_embeddings
    
    if 'gpt' in args.language_model:
        if not args.using_greedy_search:
            sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt) # List[str]
            sentence = sentence[0] # selected top 1
        else:
            sentence = greedy_search(embeddings = embeddings, tokenizer = tokenizer, model = model.gpt)
    else:
        sentence = opt_search(prompts=args.text_prompt, embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt)
        sentence=sentence[0]
    
    print(f'the generated caption: {sentence}')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = 'cuda:0')
    parser.add_argument('--clip_model', default = 'ViT-B/32')
    parser.add_argument('--language_model', default = 'gpt2')
    parser.add_argument('--continuous_prompt_length', type = int, default = 10)
    parser.add_argument('--clip_project_length', type = int, default = 10)
    parser.add_argument('--temperature', type = float, default = 0.01)
    parser.add_argument('--top_k', type = int, default = 3)
    parser.add_argument('--threshold', type = float, default = 0.2)
    parser.add_argument('--disable_all_entities', action = 'store_true', default = False, help = 'whether to use entities with a single word only')
    parser.add_argument('--name_of_entities_text', default = 'vinvl_vgoi_entities', choices = ('visual_genome_entities', 'coco_entities', 'open_image_entities', 'vinvl_vg_entities', 'vinvl_vgoi_entities'))
    parser.add_argument('--prompt_ensemble', action = 'store_true', default = False)
    parser.add_argument('--weight_path', default = './checkpoints/train_coco/coco_prefix-0014.pt')
    parser.add_argument('--image_path', default = './images/')
    parser.add_argument('--using_hard_prompt', action = 'store_true', default = False)
    parser.add_argument('--soft_prompt_first', action = 'store_true', default = False)
    parser.add_argument('--only_hard_prompt', action = 'store_true', default = False)
    parser.add_argument('--using_greedy_search', action = 'store_true', default = False, help = 'greedy search or beam search')
    parser.add_argument('--beam_width', type = int, default = 5, help = 'width of beam')
    parser.add_argument('--text_prompt', type = str, default = None)
    args = parser.parse_args()
    print('args: {}\n'.format(vars(args)))

    main(args)