import os
import nltk
import pickle
from typing import List
from nltk.stem import WordNetLemmatizer
from load_annotations import load_captions

def main(captions: List[str], path: str) -> None:
    # writing list file, i.e., [[[entity1, entity2,...], caption], ...] 

    lemmatizer = WordNetLemmatizer()
    new_captions = []
    for caption in captions:
        detected_entities = []
        pos_tags = nltk.pos_tag(nltk.word_tokenize(caption)) # [('woman': 'NN'), ...]
        for entities_with_pos in pos_tags:
            if entities_with_pos[1] == 'NN' or entities_with_pos[1] == 'NNS':
                entity = lemmatizer.lemmatize(entities_with_pos[0].lower().strip())
                detected_entities.append(entity)
        detected_entities = list(set(detected_entities))
        new_captions.append([detected_entities, caption])
    
    with open(path, 'wb') as outfile:
        pickle.dump(new_captions, outfile)
    

if __name__ == '__main__':
    datasets = ['coco_captions', 'flickr30k_captions']
    captions_path = [
        './annotations/coco/train_captions.json',
        './annotations/flickr30k/train_captions.json'
    ]
    out_path = [
        './annotations/coco/coco_with_entities.pickle',
        './annotations/flickr30k/flickr30k_with_entities.pickle'
    ]
    
    idx = 0 # only need to change here! 0 -> coco training data, 1 -> flickr30k training data
    
    if os.path.exists(out_path[idx]):
        print('Read!')
        with open(out_path[idx], 'rb') as infile:
            captions_with_entities = pickle.load(infile)
        print(f'The length of datasets: {len(captions_with_entities)}')
        captions_with_entities = captions_with_entities[:20]
        for caption_with_entities in captions_with_entities:
            print(caption_with_entities)
        
    else:
        print('Writing... ...')
        captions = load_captions(datasets[idx], captions_path[idx])
        main(captions, out_path[idx])