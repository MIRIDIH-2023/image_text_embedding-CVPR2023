# Create a vocabulary wrapper
import pickle
from collections import Counter
from pycocotools.coco import COCO
import json
import argparse
import os

annotations = {
  'mrw':  ['mrw-v1.0.json'],
  'tgif': ['tgif-v1.0.tsv'],
  'coco': ['COCO_annotations/captions_train2014.json',
           'COCO_annotations/captions_val2014.json'],
}

class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self):
    self.idx = 0
    self.word2idx = {}
    self.idx2word = {}

  def add_word(self, word):
    if word not in self.word2idx:
      self.word2idx[word] = self.idx
      self.idx2word[self.idx] = word
      self.idx += 1

  def __call__(self, word):
    if word not in self.word2idx:
      return self.word2idx['<unk>']
    return self.word2idx[word]

  def __len__(self):
    return len(self.word2idx)

  def replace(self,before_word , new_word):
    if before_word in self.word2idx:
      target_idx = self.word2idx[before_word] #index 몇번인지 찾고
      self.idx2word.update({target_idx : new_word})  #idx2word의 해당 index 값 새로운 것으로 변경
      self.word2idx[new_word] = self.word2idx.pop(before_word) #word2idx는 이전값 삭제하고 새걸로
      #전체 길이는 그대로


def from_tgif_tsv(path):
  captions = [line.strip().split('\t')[1] \
       for line in open(path, 'r').readlines()]
  return captions


def from_mrw_json(path):
  dataset = json.load(open(path, 'r'))
  captions = []
  for i, datum in enumerate(dataset):
    cap = datum['sentence']
    cap = cap.replace('/r/','')
    cap = cap.replace('r/','')
    cap = cap.replace('/u/','')
    cap = cap.replace('u/','')
    cap = cap.replace('..','')
    cap = cap.replace('/',' ')
    cap = cap.replace('-',' ')
    captions += [cap]
  return captions


def from_coco_json(path):
  coco = COCO(path)
  ids = coco.anns.keys()
  captions = []
  for i, idx in enumerate(ids):
    captions.append(str(coco.anns[idx]['caption']))

  return captions


def from_txt(txt):
  captions = []
  with open(txt, 'rb') as f:
    for line in f:
      captions.append(line.strip())
  return captions


def vocab_from_json(j):
  with open(j) as f:
    d = json.load(f)
  vocab = Vocabulary()
  vocab.word2idx = d['word2idx']
  vocab.idx2word = d['idx2word']
  vocab.idx = d['idx']
  return vocab


def build_vocab(data_path, data_name, jsons, threshold):
  """Build a simple vocabulary wrapper."""
  import nltk
  counter = Counter()
  for path in jsons[data_name]:
    full_path = os.path.join(os.path.join(data_path, data_name), path)
    if data_name == 'tgif':
      captions = from_tgif_tsv(full_path)
    elif data_name == 'mrw':
      captions = from_mrw_json(full_path)
    elif data_name == 'coco':
      captions = from_coco_json(full_path)
    else:
      captions = from_txt(full_path)

    for i, caption in enumerate(captions):
      tokens = nltk.tokenize.word_tokenize(caption.lower())
      counter.update(tokens)

  # Discard if the occurrence of the word is less than min_word_cnt.
  words = [word for word, cnt in counter.items() if cnt >= threshold]
  print('Vocabulary size: {}'.format(len(words)))

  # Create a vocab wrapper and add some special tokens.
  vocab = Vocabulary()
  vocab.add_word('<pad>')
  vocab.add_word('<start>')
  vocab.add_word('<end>')
  vocab.add_word('<unk>')
  vocab.add_word('<mask>')
  
  # Add words to the vocabulary.
  for i, word in enumerate(words):
    vocab.add_word(word)
  return vocab


def main(data_path, data_name, threshold, from_json):
  if from_json:
    vocab = vocab_from_json(data_path)
  else:
    vocab = build_vocab(data_path, data_name, jsons=annotations, threshold=threshold)
  if not os.path.isdir('./vocab'):
    os.makedirs('./vocab')
  with open('./vocab/%s_vocab.pkl' % data_name, 'wb') as f:
    pickle.dump(vocab, f, 4)
  print("Saved vocabulary file to ", './vocab/%s_vocab.pkl' % data_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--threshold', default=0, type=int)
  parser.add_argument('--from_json', action='store_true')
  parser.add_argument('--data_path', default=os.path.join(os.getcwd(),'data'))
  parser.add_argument('--data_name', default='coco', help='coco|tgif|mrw')
  opt = parser.parse_args()
  main(opt.data_path, opt.data_name, opt.threshold, opt.from_json)
