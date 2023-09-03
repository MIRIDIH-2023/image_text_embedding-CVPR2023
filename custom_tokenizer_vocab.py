from transformers import BertTokenizer , BertTokenizerFast ,BertModel
from vocab import Vocabulary

if __name__ == "__main__":
    #tokenizer = BertTokenizer.from_pretrained('monologg/kobigbird-bert-base')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #bert = BertModel.from_pretrained('bert-base-uncased')
    #tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
    tokenizer = BertTokenizerFast.from_pretrained("kykim/bertshared-kor-base")
    
    vocab = Vocabulary()
    for keys, idx in tokenizer.vocab.items():
        vocab.add_word(keys)
    vocab.replace('[unused100]','<mask>') #三 上 下 不 丑
    vocab.replace('[unused101]','<pad>')
    vocab.replace('[unused102]','<start>')
    vocab.replace('[unused103]','<end>')
    vocab.replace('[unused104]','<unk>')
    
    print(vocab.word2idx)
    
    #for keys , id in vocab.word2idx.items():
    #    print(keys, id)
    #    if(id==2000):
    #        break
    
    print(tokenizer.all_special_tokens)
    print(vocab.idx)
    print(tokenizer.vocab_size)
    
    #print(bert)