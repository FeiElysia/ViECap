import os
import clip
import torch
import pickle
from PIL import Image
from typing import List, Optional, Tuple

@torch.no_grad()
def clip_texts_embeddings(
    texts: List[str],
    outpath = '',
    device: Optional[str] = None,
    batch_size: Optional[int] = 32,
    clip_type: Optional[str] = None
) -> torch.Tensor:
    """
    Args:
        texts: name of categories, i.e., ['category1', 'category2', ...]
        outpath: saving embeddings of category texts to outpath. reading it directly if existing
        device: specifying device used
        batch_size: the number of categories that would be transformed to embeddings per epoch
        clip_type: specifying clip backbone used
    Return:
        tensor with a shape of (num_categories, clip_hidden_size), float32
    """
    if os.path.exists(outpath):
        with open(outpath, 'rb') as infile:
            texts_embeddings = pickle.load(infile) # (num_categories, clip_hidden_size)
        return texts_embeddings

    # adding prompt for each category text, i.e., Photo of an ariplane. / Photo of a bicycle. 
    vowel = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    prompt_texts = []
    for text in texts:                                              
        if text[0] in vowel:
            prompt_texts.append(f'A photo of an {text}.')
        else:
            prompt_texts.append(f'A photo of a {text}.')
 
    clip_texts_tokens = clip.tokenize(prompt_texts)    # (num_categories, 77)
    model, _ = clip.load(clip_type, device = device)   # loading clip encoder
    model.eval()
    num_categories = len(texts)
    texts_embeddings = None
    epochs = int(num_categories / batch_size) if num_categories % batch_size == 0 else 1 + int (num_categories // batch_size)
    for epoch in range(epochs):
        temp_texts_tokens = clip_texts_tokens[batch_size * epoch : batch_size * (epoch + 1)] # (batch_size/(num_categories % batch_size), 77)
        temp_texts_tokens = temp_texts_tokens.to(device)
        with torch.no_grad():
            temp_texts_embeddings = model.encode_text(temp_texts_tokens).float().to('cpu')   # (batch_size/(num_categories % batch_size), clip_hidden_size)
        if texts_embeddings is None:
            texts_embeddings = temp_texts_embeddings
        else:
            texts_embeddings = torch.cat((texts_embeddings, temp_texts_embeddings), dim = 0)

    with open(outpath, 'wb') as outfile:
        pickle.dump(texts_embeddings, outfile)

    return texts_embeddings

def image_text_simiarlity(
    texts_embeddings: torch.Tensor,
    temperature: float = 0.01,
    image_path: Optional[str] = None,
    images_features: Optional[torch.Tensor] = None,
    clip_type: Optional[str] = None,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Args:
        texts_embeddings: (num_categories, clip_hidden_size), float32, the embeddings of categories
        temperature: temperature hyperparameter for computing similarity
        image_path: Optional, the path of a single image
        images_feature: (num_images, clip_hidden_size), float32, Optional
        clip_type: clip type, using when input is image path
        device: device using when input is device
    Return:
        logits with a shape of (num_images, num_categories)
    """
    if images_features is None:
        encoder, preprocess = clip.load(clip_type, device)
        assert image_path is not None, 'Either image path or images feature should be given!'
        image = preprocess(Image.open(image_path)).unsqueeze(dim = 0).to(device)                             # (1, 3, 224, 224)
        with torch.no_grad():
            images_features = encoder.encode_image(image)                                                    # (1, clip_hidden_size)
    
    # computing on cpu to avoid out of memory
    images_features = images_features.float().to('cpu')                                                      # (num_images, clip_hidden_size)
    texts_embeddings = texts_embeddings.float().to('cpu')                                                    # (num_categories, clip_hidden_size)
    images_features /= images_features.norm(dim = -1, keepdim = True)                                        # (num_images, clip_hidden_size)
    texts_embeddings /= texts_embeddings.norm(dim = -1, keepdim = True)                                      # (num_categories, clip_hidden_size)

    image_to_text_similarity = torch.matmul(images_features, texts_embeddings.transpose(1, 0)) / temperature # (num_imegs, num_categories)
    image_to_text_logits = torch.nn.functional.softmax(image_to_text_similarity, dim = -1)                   # (num_imegs, num_categories)
    
    return image_to_text_logits

def top_k_categories(
    texts: List[str],                  # ['category1', 'category2', ...], len = num_categories
    logits: torch.Tensor,              # (num_images, num_categories)
    top_k: Optional[int] = 5,          # choosing top k categories as retrieved category
    threshold: Optional[float] = 0.0   # probability which is less than threshold will be filtered
) -> Tuple:
    
    top_k_probs, top_k_indices = torch.topk(logits, k = top_k, dim = -1) # (num_images, top_k)
    top_k_texts = []
    for i in range(len(top_k_probs)):
        per_image_top_k_probs = top_k_probs[i]            # the ith image top k probability
        per_image_top_k_indices = top_k_indices[i]        # the ith image top k indices
        temp_texts = []
        for j in range(top_k):
            if per_image_top_k_probs[j] < threshold:
                break
            temp_texts.append(texts[per_image_top_k_indices[j]])
        top_k_texts.append(temp_texts)
    
    return top_k_texts, top_k_probs