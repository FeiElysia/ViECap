## Transferable Decoding with Visual Entities for Zero-Shot Image Captioning, ICCV 2023

**Authors**: [Junjie Fei](https://feielysia.github.io/), [Teng Wang](http://ttengwang.com/), [Jinrui Zhang](https://github.com/zjr2000), Zhenyu He, Chengjie Wang, [Feng Zheng](https://faculty.sustech.edu.cn/fengzheng/)

This repository contains the official implementation of our paper: [*Transferable Decoding with Visual Entities for Zero-Shot Image Captioning*](https://openaccess.thecvf.com/content/ICCV2023/html/Fei_Transferable_Decoding_with_Visual_Entities_for_Zero-Shot_Image_Captioning_ICCV_2023_paper.html).

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2307.16525-b31b1b.svg)](https://arxiv.org/abs/2307.16525)
[![bilibili](https://img.shields.io/badge/dynamic/json?label=video&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1Uj41187Uw)](https://www.bilibili.com/video/BV1Uj41187Uw)

</div>

<div align = center>
<img src="https://github.com/FeiElysia/ViECap/blob/main/demo/poster.png">
</div>

***

## Catalogue:
* <a href='#introduction'>Introduction</a>
* <a href='#examples'>Examples</a>
* <a href='#citation'>Citation</a>
* <a href='#prerequisites'>Required Prerequisites</a>
* <a href='#data_preparation'>Data Preparation</a>
* <a href='#training'>Training</a>
* <a href='#evaluation'>Evaluation</a>
    * <a href='#cross_domain'>Cross-domain Captioning</a>
    * <a href='#in_domain'>In-domain Captioning</a>
    * <a href='#flickrstyle10k'>FlickrStyle10K</a>
* <a href='#inference'>Inference</a>
* <a href='#acknowledgments'>Acknowledgments</a>
* <a href='#contact'>Contact</a>

<span id = 'introduction'/>

***

## Introduction

This paper aims at the transferability of the zero-shot captioning for out-of-domain images. As shown in this image, we demonstrate the susceptibility of pre-trained vision-language models and large language models to *modality bias* induced by language models when adapting them into image-to-text generation. Simultaneously, these models tend to generate descriptions containing objects that do not actually exist in the image but frequently appear during training, a phenomenon known as *object hallucination*. We propose ViECap, a transferable decoding model that leverages entity-aware decoding to generate descriptions in both seen and unseen scenarios. This is the official repository for ViECap, in which you can easily reproduce our paper's results and try it on your own images.

<div align = center>
<img src="https://github.com/FeiElysia/ViECap/blob/main/demo/experiment1.png" width = 80% heigth = 80%>
</div>

***

<span id = 'examples'/>

## Examples

Here are some fantastic examples for diverse captioning scenarios of our model!

<div align = center>
<img src="https://github.com/FeiElysia/ViECap/blob/main/demo/honkai.png">
</div>

<br/>

The captioning results on the NoCaps dataset are presented here:

<div align = center>
<img src="https://github.com/FeiElysia/ViECap/blob/main/demo/nocaps.png">
</div>

***

<div align = center>

Task    | COCO $\Rightarrow$ Nocaps (In) | COCO $\Rightarrow$ Nocaps (Near) | COCO $\Rightarrow$ Nocaps (Out) | COCO $\Rightarrow$ Nocaps (Overall) | COCO $\Rightarrow$ Flickr30k | Flickr30k $\Rightarrow$ COCO | COCO | Flickr30k |
--------|------|------|------|------|------|------|------|------|
Metric	|CIDEr |CIDEr |CIDEr |CIDEr |CIDEr |CIDEr |CIDEr |CIDEr |
MAGIC   |----  |----  |----  |----  |17.5  |18.3  |49.3  |20.4  |
DeCap   |65.2  |47.8  |25.8  |45.9  |35.7  |44.4  |91.2  |56.7  |
CapDec  |60.1  |50.2  |28.7  |45.9  |35.7  |27.3  |91.8  |39.1  |
-----   |----  |----  |----  |----  |----  |----  |----  |----  |
ViECap  |61.1  |64.3  |65.0  |66.2  |38.4  |54.2  |92.9  |47.9  |

</div>

***

<span id = 'citation'/>

## Citation

If you find our paper and code helpful, we would greatly appreciate it if you could leave a star and cite our work. Thanks!

```
@InProceedings{Fei_2023_ICCV,
    author    = {Fei, Junjie and Wang, Teng and Zhang, Jinrui and He, Zhenyu and Wang, Chengjie and Zheng, Feng},
    title     = {Transferable Decoding with Visual Entities for Zero-Shot Image Captioning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {3136-3146}
}
```

```bibtex
@article{fei2023transferable,
  title={Transferable Decoding with Visual Entities for Zero-Shot Image Captioning},
  author={Fei, Junjie and Wang, Teng and Zhang, Jinrui and He, Zhenyu and Wang, Chengjie and Zheng, Feng},
  journal={arXiv preprint arXiv:2307.16525},
  year={2023}
}
```

***

<span id = 'prerequisites'/>

## Required Prerequisites

For code execution, begin by cloning this repository and downloading the annotations, checkpoints, and evaluation files from the [Releases](https://github.com/FeiElysia/ViECap/releases/tag/checkpoints) of this repository. Afterward, unzip the files and position them within the root directory. It should be noted that we only run our codes on Linux.

```
git clone git@github.com:FeiElysia/ViECap.git
```

***

<span id = 'data_preparation'/>

## Data Preparation

To utilize this code with your desired dataset, the initial step involves converting the dataset format through data preprocessing.
Firstly, extract the entities from each caption within your chosen dataset using the following command (make sure you have placed all captions from the dataset into a list):

```
python entities_extraction.py
```

(Optional) you can pre-extract the training text features.

```
python texts_features_extraction.py
```

Using these two scripts, you can now transform any dataset you wish to use for training into the appropriate data format for the dataloader. Additionally, we have made the processed COCO dataset and Flickr30k dataset available in the [Releases](https://github.com/FeiElysia/ViECap/releases/tag/checkpoints), feel free to use them directly!

To evaluate the trained ViECap, you should first construct the vocabulary and extract the embeddings of each category in the vocabulary. Utilize the vocabulary provided in the [Releases](https://github.com/FeiElysia/ViECap/releases/tag/checkpoints) and execute the following script (we also supply the extracted vocabulary embeddings here):

```
python generating_prompt_ensemble.py
```

(Optional) you can also acquire the image features beforehand for evaluation. Make sure to modify the script if you want to adapt it to your own dataset.

Note that if you choose not to use the provided image features from us, you should download the image source files for the COCO and Flickr30k dataset from their official websites. Afterwards, you should place these files into the 'ViECap/annotations/coco/val2014' directory for COCO images and the 'ViECap/annotations/flickr30k/flickr30k-images' directory for Flickr30k images.

```
python images_features_extraction.py
```

<span id = 'training'/>

## Training

To train ViECap on the COCO dataset or the Flickr30k dataset, using the following script (```bash train_*.sh n```), respectively:

```
bash train_coco.sh 0
bash train_flickr30k.sh 0
```

where ```n``` represents the ID of gpu used (*i.e., 'cuda:n'*).

***

<span id = 'evaluation'/>

## Evaluation

Now, you can evaluate the captioning performance of your trained model on the testing dataset using  the command ```bash eval_*.sh EXP_NAME n OTHER_ARGS m```, in which ```EXP_NAME``` signifies the file name for storing checkpoints, ```OTHER_ARGS``` signifies any other arguments used, and ```n``` and ```m``` refer to the GPU ID and the weight epoch used, respectively.

***

<span id = 'cross_domain'/>

### Cross-domain Captioning

To evaluate the cross-domain captioning performance from COCO to NoCaps, run the following script:

```
bash eval_nocaps.sh train_coco 0 '--top_k 3 --threshold 0.2' 14
```

<div align = center>

Task    | COCO $\Rightarrow$ NoCaps (In)| COCO $\Rightarrow$ NoCaps (In) | COCO $\Rightarrow$ NoCaps (Near) | COCO $\Rightarrow$ NoCaps (Near) | COCO $\Rightarrow$ NoCaps (Out)| COCO $\Rightarrow$ NoCaps (Out) | COCO $\Rightarrow$ NoCaps (Overall) | COCO $\Rightarrow$ NoCaps (Overall) |
--------|-----|-----|-----|-----|-----|-----|-----|-----|
Metric	|CIDEr|SPICE|CIDEr|SPICE|CIDEr|SPICE|CIDEr|SPICE|
DeCap   |65.2 |---- |47.8 |---- |25.8 |---- |45.9 |---- |
CapDec  |60.1 |10.2 |50.2 |9.3  |28.7 |6.0  |45.9 |8.3  |
-----   |---- |---- |---- |---- |---- |---- |---- |---- |
ViECap  |61.1 |10.4 |64.3 |9.9  |65.0 |8.6  |66.2 |9.5  |

</div>

***

To evaluate the cross-domain captioning performance from COCO to Flickr30k, run the following script:

```
bash eval_flickr30k.sh train_coco 0 '--top_k 3 --threshold 0.2' 14
```

<div align = center>

Metric|BLEU@4|METEOR|CIDEr|SPICE|
----- |----  |----  |---- |---  |
MAGIC |6.2   |12.2  |17.5 |5.9  |
DeCap |16.3  |17.9  |35.7 |11.1 |
CapDec|17.3  |18.6  |35.7 |---- |
----- |----  |----  |---- |---- |
ViECap|17.4  |18.0  |38.4 |11.2 |

</div>

***

To evaluate the cross-domain captioning performance from Flickr30k to COCO, run the following script:

```
bash eval_coco.sh train_flickr30k 0 '--top_k 3 --threshold 0.2 --using_greedy_search' 29
```

<div align = center>

Metric|BLEU@4|METEOR|CIDEr|SPICE|
----- |----  |----  |---- |---  |
MAGIC |5.2   |12.5  |18.3 |5.7  |
DeCap |12.1  |18.0  |44.4 |10.9 |
CapDec|9.2   |16.3  |27.3 |---- |
----- |----  |----  |---- |---- |
ViECap|12.6  |19.3  |54.2 |12.5 |

</div>

***

<span id = 'in_domain'/>

### In-domain Captioning

To evaluate the in-domain captioning performance on the COCO testing set, run the following script:

```
bash eval_coco.sh train_coco 0 '' 14
```

<div align = center>

Metric  |BLEU@4|METEOR|CIDEr|SPICE|
-----   |----  |----  |---- |---- |
ZeroCap |7.0   |15.4  |34.5 |9.2  |
MAGIC   |12.9  |17.4  |49.3 |11.3 |
DeCap   |24.7  |25.0  |91.2 |18.7 |
CapDec  |26.4  |25.1  |91.8 |---- |
-----   |----  |----  |---- |---- |
ViECap  |27.2  |24.8  |92.9 |18.2 |

</div>

***

To evaluate the in-domain captioning performance on the Flickr30k testing set, run the following script:

```
bash eval_flickr30k.sh train_flickr30k 0 '' 29
```

<div align = center>

Metric  |BLEU@4|METEOR|CIDEr|SPICE|
-----   |----  |----  |---- |---- |
ZeroCap |5.4   |11.8  |16.8 |6.2  |
MAGIC   |6.4   |13.1  |20.4 |7.1  |
DeCap   |21.2  |21.8  |56.7 |15.2 |
CapDec  |17.7  |20.0  |39.1 |---- |
-----   |----  |----  |---- |---- |
ViECap  |21.4  |20.1  |47.9 |13.6 |

</div>

***

<span id = 'flickrstyle10k'/>

### FlickrStyle10K

For FlickrStyle10K, you can easily put it into practice by adhering to the aforementioned steps. Begin by downloading the [dataset](https://zhegan27.github.io/Papers/FlickrStyle_v0.9.zip)!

***

We have provided the captioning results in the [Releases](https://github.com/FeiElysia/ViECap/releases/tag/checkpoints). You can evaluate them directly using ```bash language_eval.sh </path>```

For example, if you wish to assess the cross-domain captioning performance from COCO to NoCaps, execute the following commands:

```
bash language_eval.sh ../checkpoints/train_coco/indomain_generated_captions.json
bash language_eval.sh ../checkpoints/train_coco/neardomain_generated_captions.json
bash language_eval.sh ../checkpoints/train_coco/outdomain_generated_captions.json
bash language_eval.sh ../checkpoints/train_coco/overall_generated_captions.json
```

***

<span id = 'inference'/>

## Inference

you can describe any image you need according to the following script:

```
python infer_by_instance.py --prompt_ensemble --using_hard_prompt --soft_prompt_first --image_path ./images/instance1.jpg
```

The generated caption is: *A little girl in pink pajamas sitting on a bed.*
<div align = center>
<img src="https://github.com/FeiElysia/ViECap/blob/main/images/instance1.jpg" width = 80% heigth = 80%>
</div>

***

Change ```--image_path``` to specify the path of any image you want to describe!

<div align = center>
<img src="https://github.com/FeiElysia/ViECap/blob/main/images/instance2.jpg" width = 80% heigth = 80%>

*A little girl that is laying down on a bed.*
</div>

<div align = center>
<img src="https://github.com/FeiElysia/ViECap/blob/main/images/instance4.jpg" width = 80% heigth = 80%>

*A scenic view of a river with a waterfall in the background.*
</div>

<div align = center>
<img src="https://github.com/FeiElysia/ViECap/blob/main/demo/gakki.png" width = 80% heigth = 80%>
</div>

<div align = center>
<img src="https://github.com/FeiElysia/ViECap/blob/main/images/keqing1.jpg" width = 80% heigth = 80%>

*A girl with a ponytail is walking down the street.*
</div>

(Optional) you can also execute the following script to generate captions for all the images within a specific file.

```
python infer_by_batch.py --prompt_ensemble --using_hard_prompt --soft_prompt_first --image_path ./images
```

***

<span id = 'acknowledgments'/>

## Acknowledgments

Our repository builds on [CLIP](https://github.com/openai/CLIP), [ClipCap](https://github.com/rmokady/CLIP_prefix_caption), [CapDec](https://github.com/DavidHuji/CapDec), [MAGIC](https://github.com/yxuansu/MAGIC) and pycocotools repositories. Thanks for open-sourcing!

***

<span id = 'contact'/>

## Contact

If you have any questions, please feel free to contact me at: junjiefei@outlook.com.
