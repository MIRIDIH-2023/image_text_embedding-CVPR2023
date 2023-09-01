import os
import sys
import random
import pickle

import numpy as np
import json as jsonmod

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import nltk
from PIL import Image
from transformers import BertTokenizer

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TOKENIZER.add_special_tokens(special_tokens_dict=
                            {'additiona_special_tokens' : ['<mask>','<pad>','<start>','<end>','<unk>']})


print(TOKENIZER.tokenize(text='hello word'))

def tokenize(sentence, vocab, drop_prob):
    # Convert sentence (string) to word ids.
    def caption_augmentation(tokens):
        idxs = []
        for i, t in enumerate(tokens):
            if(i>500):
                break
            prob = random.random()
            if prob < drop_prob:
                prob /= drop_prob
                if prob < 0.5:
                    idxs += [vocab('<mask>')]
                elif prob < 0.6:
                    idxs += [random.randrange(len(vocab))]
            else:
                idxs += [vocab(t)]
        return idxs
    
    if sys.version_info.major > 2:
        #tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
        tokens = TOKENIZER.tokenize(text=str(sentence).lower())
    else:
        #tokens = nltk.tokenize.word_tokenize(str(sentence).lower().decode('utf-8'))
        tokens = TOKENIZER.tokenize(text=str(sentence).lower().decode('utf-8'))
    return torch.Tensor(
        [vocab('<start>')] + caption_augmentation(tokens) + [vocab('<end>')]
    )

def process_caption_bert(caption, tokenizer, drop_prob, train):
        output_tokens = []
        deleted_idx = []
        tokens = tokenizer.basic_tokenizer.tokenize(caption)
        
        for i, token in enumerate(tokens):
            sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()

            if prob < drop_prob and train:  # mask/remove the tokens only during training
                prob /= drop_prob

                # 50% randomly change token to mask token
                if prob < 0.5:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.6:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)
                        deleted_idx.append(len(output_tokens) - 1)
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)

        if len(deleted_idx) != 0:
            output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

        output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
        target = tokenizer.convert_tokens_to_ids(output_tokens)
        target = torch.Tensor(target)
        return target


class CustomDatasetBert(data.Dataset):

    def __init__(self, image_root, json_root, vocab, split, transform=None, ids=None, drop_prob=0):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = image_root
        self.image_len = 10000 #len(os.listdir(image_root))
        
        with open(json_root,'rb') as file:
            self.json_list = pickle.load(file) 
        
        self.train = split == 'train'
        
        self.transform = transform
        self.drop_prob = drop_prob
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.vocab = self.tokenizer.vocab
        self.vocab = vocab

    def __len__(self):
        return self.image_len


    def __getitem__(self, index):
        vocab = self.vocab
        ann_ids, anns, path, image = self.get_raw_item(index)
        if self.transform is not None:
            image = self.transform(image)
        #anns = [process_caption_bert(ann, self.tokenizer, self.drop_prob, self.train) for ann in anns]
        anns = [tokenize(ann, vocab, self.drop_prob) for ann in anns]
        
        return image, anns, index, ann_ids


    def get_raw_item(self, index):
        """

        Args:
            index (_type_): _description_

        Returns:
            ann_ids : annotation 개수
            anns : annotation ist
            path : image path
            image : image (~.png)
        """
        anns = get_annotations(self.json_list , index)
        ann_ids = len(anns)
        
        path = f"thumnail_image_{index}.png"
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        return ann_ids, anns, path, image
    

def get_annotations(data_list , index):
    """
    image와 mapping된 keyword / xml text / (image caption)을 list 형태로 return

    Args:
        data_list (_type_): json data list
        index (_type_): _description_

    Returns:
        _type_: annotation list
    """
    return_list = []
    while len(return_list)==0:
        
        keyword = data_list[index]['keyword']

        if len(keyword) == 0: #if has error, pick another index
            index = random.randint(0,len(data_list)-1)
            continue

        text = ""

        for j in range(len(data_list[index]['form'])):
            if type(data_list[index]['form'][j]['text']) == str:
                text += data_list[index]['form'][j]['text'] + "\n"
        
        return_list = [keyword , text]
    
    return return_list