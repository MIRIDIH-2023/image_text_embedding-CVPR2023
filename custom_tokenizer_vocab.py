from transformers import BertTokenizer
from vocab import Vocabulary

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    vocab = Vocabulary()
    vocab.add_word('<mask>')
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    for keys, idx in enumerate(tokenizer.vocab.items()):
        vocab.add_word(keys)
    
    print(tokenizer.tokenize(text='hello word'))