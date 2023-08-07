import json
import pickle
import pandas as pd
from typing import List

def load_coco_captions(path: str) -> List[str]:

    with open(path, 'r') as infile:
        annotations = json.load(infile)               # dictionary -> {image_path: List[caption1, caption2, ...]}
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    captions = []
    for image_path in annotations:                  
        temp_captions = annotations[image_path]        # List: [caption1, caption2, ...], captions for the ith image
        for caption in temp_captions:                  # caption
            caption = caption.strip()                  # removing space at the end of the caption
            if caption.isupper():                      # processing the special caption in the COCO Caption, e.g., 'A BOY IS PLAYING BASEBALL.'
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:] # capitalizing the first letter in the caption
            if caption[-1] not in punctuations:        # adding a '.' at the end of the caption if there are no punctuations.
                caption += '.'
            captions.append(caption)                   # final versin: A boy is playing baseball.

    return captions

def load_flickr30k_captions(path: str) -> List[str]:
    
    with open(path, 'r') as infile:
        annotations = json.load(infile) # dictionary -> {image_path: List[caption1, caption2, ...]}
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    captions = []
    for image_path in annotations:                  
        temp_captions = annotations[image_path]
        for caption in temp_captions:
            caption = caption.strip()
            if caption.isupper():
                caption = caption.lower()
            caption = caption[0].upper() + caption[1:]
            if caption[-1] not in punctuations:
                caption += '.'
            captions.append(caption)

    return captions

def load_captions(name_of_datasets: str, path_of_datasets: str) -> List[str]:
    """
    Args:
        name_of_datasets: specifying the name of datasets
        path_of_datasets: specifying the path of datasets
    Return:
        [caption1, caption2, ...]
    """
    if name_of_datasets == 'coco_captions':
        return load_coco_captions(path_of_datasets)

    if name_of_datasets == 'flickr30k_captions':
        return load_flickr30k_captions(path_of_datasets)

    print('The datasets for training fail to load!')

def load_stopwords() -> List[str]:
    # Return: stopwords and punctuations

    stopwords = {'per', '’ll', 'could', 'fifteen', 'been', "isn't", 'whoever', 'any', 'whole', 'front', "won't", 'upon', 'there', 's', 'am', 'via', 'the', 'as', "haven't", 'on', 'km', 'further', 'their', 'quite', 'have', 'twenty', 'during', 'full', 'it', 'thin', 'so', 'what', 'an', 't', 'less', 'if', 'sixty', 'everyone', 'us', 'were', 'side', 'she', 'cannot', 'thereby', '‘ve', 'amount', 'n’t', 'be', 'nine', 'isn', 'wouldn', 'by', 'along', "'ll", 'themselves', 'forty', 'everywhere', "'d", 'thru', 'sometimes', 'hasnt', 'seeming', 'own', 'that', "'ve", 'least', 'with', 'inc', 'really', 'afterwards', 'due', 'for', 'sometime', 'last', 'find', 'therein', 'all', 'thick', 'detail', 'few', 'hundred', 'some', 'even', 'off', '’m', 'ain', '’re', 'hence', 'etc', 'into', 'rather', 'where', 'm', 'its', 'onto', '’s', 'get', 'other', 'moreover', 'noone', 'being', 'must', 'bill', "wasn't", 'system', 'neither', "you'll", 'third', 'whereby', 'nobody', 'among', 'throughout', 'except', 'beforehand', "didn't", 'was', 'without', 'whose', 'hasn', '‘d', 'or', 'theirs', 'various', 'name', 'twelve', 'myself', 'former', 'though', 'we', 'ours', 'many', 'sincere', 'regarding', 'had', 'before', 'mustn', 'either', 'doing', 'why', 'fill', 'eight', 'won', 'anything', 'hereupon', 'this', 'amoungst', '‘s', 'of', 'yourselves', 'beside', 'within', 'ourselves', '‘re', 'about', 'elsewhere', 'latter', 'through', 'll', 'i', 'wasn', 'anywhere', 'weren', 'just', 'itself', "you're", 'wherein', 'four', 'keep', 'whether', 'nothing', 'found', 'back', 'needn', "aren't", 'has', 'one', 'wherever', 'serious', 'everything', 'hadn', 'first', 'anyway', 'co', 'still', 'five', 'becomes', "don't", 'formerly', 'ever', 'part', 'nowhere', 'made', 'himself',  "couldn't", 'none', 'others', 'now', 'doesn', 'at', 'another', 'does', 'kg', 'see', 'often', 'them', 'shan', 'fifty', 'ltd', 'namely', 'they', 'somewhere', 'haven', 'take', 'latterly', 'well', 'whatever', 'nor', 'whereafter', 'might', 'only', 'de', 'our', 'hers', "mustn't", 'aren', 'you', 'his', "wouldn't", 'please', 'empty', 'but', 'mightn', 'then', 'should', 'and', 'each', 'such', 'a', 'yet', 'y', 'enough', 'someone', 'would', 'since', 'however', 'make', 'alone', 'anyone', 'amongst', 'these', 'whereupon', 'fire', "hasn't", 'shouldn', 'didn', 'do', 'me', 'becoming', 'after', 'several', 'seem', 'her', 'three', 'out', 'ten', 'whence', 'eg', 'couldn', 'un', 'did', "she's", 'whither', 'toward', 'once', "should've", 'call', "weren't", 'again', 'more', 'show', 'seems', "needn't", 'thereupon', 'used', 'most', 'hereby', 'put', 'ie', 've', 'my', 'your', 'thence', 'already', 'always', 'having', 'much', 'move', 'eleven', "'re", 'here', 'yours', 'con', 'done', 'up', 'over', 'yourself', "it's", 'o', 'six', 'can', 'how', "hadn't", 'anyhow', 'below', 'also', 'say', 'together', 'down', 'using', 'while', 'almost', 'cry', "you've", '’ve', 'two', 'towards', 'meanwhile', 'perhaps', 'when', 'ma', "shouldn't", 'both', 'hereafter', 'he', 'describe', 'ca', 'which', 'every', 'between', 'give', 'go', 'very', '’d', 'nevertheless', 'is', 'n‘t', 'therefore', '‘ll', 'unless', 'next', 'who', 'became', 'mill', 'him', 'don', 'same', "'s", 'seemed', 'mostly', 'will', 're', "you'd", 'no', 'in', 'too', "mightn't", 'besides', 'are', 'because', 'couldnt', 'd', 'against', "doesn't", 'cant', 'whenever', 'somehow', 'thereafter', 'although', 'beyond', 'from', 'whereas', 'thus', 'than', "shan't", 'to', 'top', 'until', 'those', 'whom', 'bottom', 'else', 'herein', 'something', '‘m', 'may', 'not', "that'll", "'m", 'indeed', 'never', 'herself', 'interest', "n't", 'become', 'mine', 'otherwise'}
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    other_words = {'photo', 'image', 'picture', 'pic', 'side', 'part', 'background'}
    stopwords_and_punctuations = stopwords.union(punctuations)
    stopwords_and_punctuations = stopwords_and_punctuations.union(other_words)
    stopwords_and_punctuations = [stopword.lower() for stopword in stopwords_and_punctuations]
    stopwords_and_punctuations.sort()

    return stopwords_and_punctuations

def load_visual_genome_entities(path: str, all_entities: bool = True) -> List[str]:
    # Visual Genome Vocabulary

    with open(path, 'rb') as infile:
        all_objects_attributes_relationships = pickle.load(infile) # dictionary {'relationships': dict, 'attributes': dict, 'objects': dict}
    entities = all_objects_attributes_relationships['objects']     # dictionary {'gqa': set, 'vg': set, 'joint': set}, joint = gqa + vg
    entities = entities['joint']                                   # set
    
    if all_entities:
        entities = [entity.lower().strip() for entity in entities]
    else:
        entities = [entity.lower().strip() for entity in entities if len(entity.split()) == 1]
    entities.sort()  # sort

    return entities

def load_coco_entities(path: str, all_entities: bool = True) -> List[str]:
    # COCO Vocabulary

    with open(path, 'r') as infile:
        entities = json.load(infile)       # List [category1, category2, ...]
    
    if all_entities:
        entities = [entity.lower().strip() for entity in entities]
    else:
        entities = [entity.lower().strip() for entity in entities if len(entity.split()) == 1]
    entities.sort()  # sort

    return entities

def load_open_image_entities(path: str, all_entities: bool = True) -> List[str]:
    # Open Image Vocabulary

    open_images = pd.read_csv(path)                     # 601x2, i.e., [LabelName, DisplayName]
    open_image_entities = list(open_images.DisplayName) # list
    
    for i in range(len(open_image_entities)):
        entity = open_image_entities[i].lower().strip()
        if entity[-1] == ')':
            entity = entity[:entity.find('(')].strip()
        open_image_entities[i] = entity

    if all_entities:
        entities = [entity for entity in open_image_entities]
    else:
        entities = [entity for entity in open_image_entities if len(entity.split()) == 1]
    entities.sort()  # sort

    return entities

def load_vinvl_vg_entities(path: str, all_entities: bool = True) -> List[str]:
    # VG Vocabulary

    with open(path, 'r') as infile: 
        annotations = json.load(infile)          # dictionary = {'label_to_idx':dict,'idx_to_label':dict,'attribute_to_idx':dict,'idx_to_attribute':dict,'predicate_to_idx':dict,'idx_to_predicate':dict,'object_count':dict,'attribute_count':dict,'predicate_count':dict,}
    vinvl_entities = annotations['object_count'] # dictionary = {str: int, str: int, ...}

    if all_entities:
        entities = [entity.lower().strip() for entity in vinvl_entities]
    else:
        entities = [entity.lower().strip() for entity in vinvl_entities if len(entity.split()) == 1]
    entities.sort()  # sort

    return entities

def load_vinvl_vgoi_entities(path: str, all_entities: bool = True) -> List[str]:

    with open(path, 'r') as infile: 
        vgoi_entities = json.load(infile) # dictionary = {str: int}

    if all_entities:
        entities = [entity.lower().strip() for entity in vgoi_entities]
    else:
        entities = [entity.lower().strip() for entity in vgoi_entities if len(entity.split()) == 1]
    entities.sort()  # sort

    return entities

def load_entities_text(name_of_entities: str, path_of_entities: str, all_entities: bool = True) -> List[str]:
    """
    Args:
        name_of_entities: specifying the name of entities text
        path_of_entities: specifying the path of entities text
        all_entities: whether to apply all entities text. True denotes using entities including len(entitites.split()) > 1
    Return:
        [entity1, entity2, ...]
    """
    if name_of_entities == 'visual_genome_entities':
        return load_visual_genome_entities(path_of_entities, all_entities)
    
    if name_of_entities == 'coco_entities':
        return load_coco_entities(path_of_entities, all_entities)

    if name_of_entities == 'open_image_entities':
        return load_open_image_entities(path_of_entities, all_entities)

    if name_of_entities == 'vinvl_vg_entities':
        return load_vinvl_vg_entities(path_of_entities, all_entities)

    if name_of_entities == 'vinvl_vgoi_entities':
        return load_vinvl_vgoi_entities(path_of_entities, all_entities)

    print('The entities text fails to load!')

if __name__ == '__main__':
    
    # loading captions
    datasets = ['coco_captions', 'flickr30k_captions']
    captions_path = [
        './annotations/coco/train_captions.json',
        './annotations/flickr30k/train_captions.json',
    ]
    captions_idx = 1
    captions = load_captions(datasets[captions_idx], captions_path[captions_idx])
    for caption in captions[:20]:
        print(caption)
    print(len(captions), type(captions))

    # loading stopwords
    stopwords = load_stopwords()
    print('stopwords: ', stopwords[:10], type(stopwords), len(stopwords))

    # loading entities text
    entities_text = ['visual_genome_entities', 'coco_entities', 'open_image_entities', 'vinvl_vg_entities', 'vinvl_vgoi_entities']
    entities_path = [
        './annotations/vocabulary/all_objects_attributes_relationships.pickle',
        './annotations/vocabulary/coco_categories.json',
        './annotations/vocabulary/oidv7-class-descriptions-boxable.csv',
        './annotations/vocabulary/VG-SGG-dicts-vgoi6-clipped.json',
        './annotations/vocabulary/vgcocooiobjects_v1_class2ind.json'
    ]
    # using all entities text
    entities_idx = 4
    entities = load_entities_text(entities_text[entities_idx], entities_path[entities_idx])
    print('entities text: ', entities[:10], type(entities), len(entities))
    # using entities text with a single word
    entities_idx = 4
    entities = load_entities_text(entities_text[entities_idx], entities_path[entities_idx], all_entities = False)
    print('entities text: ', entities[:10], type(entities), len(entities))