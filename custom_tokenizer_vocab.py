from transformers import BertTokenizer
from vocab import Vocabulary

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('kykim/bert-kor-base')
    
    vocab = Vocabulary()
    for keys, idx in enumerate(tokenizer.vocab.items()):
        vocab.add_word(keys)
    vocab.replace(41994,'<mask>')
    vocab.replace(41995,'<pad>')
    vocab.replace(41996,'<start>')
    vocab.replace(41997,'<end>')
    vocab.replace(41998,'<unk>')
    
    print(vocab.word2idx)
    print(tokenizer.all_special_tokens)
    print(vocab.idx)
    print(tokenizer.vocab_size)