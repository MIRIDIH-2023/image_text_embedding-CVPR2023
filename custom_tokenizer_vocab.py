from transformers import BertTokenizer
from vocab import Vocabulary

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('monologg/kobigbird-bert-base')
    
    vocab = Vocabulary()
    for keys, idx in tokenizer.vocab.items():
        vocab.add_word(keys)
    vocab.replace('三','<mask>') #三 上 下 不 丑
    vocab.replace('上','<pad>')
    vocab.replace('下','<start>')
    vocab.replace('不','<end>')
    vocab.replace('丑','<unk>')
    
    print(vocab.word2idx)
    print(tokenizer.all_special_tokens)
    print(vocab.idx)
    print(tokenizer.vocab_size)