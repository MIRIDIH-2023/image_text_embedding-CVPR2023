from transformers import BertTokenizer , BertModel
from vocab import Vocabulary

if __name__ == "__main__":
    #tokenizer = BertTokenizer.from_pretrained('monologg/kobigbird-bert-base')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    
    """
    vocab = Vocabulary()
    for keys, idx in tokenizer.vocab.items():
        vocab.add_word(keys)
    vocab.replace('##：','<mask>') #三 上 下 不 丑
    vocab.replace('##？','<pad>')
    vocab.replace('##～','<start>')
    vocab.replace('##／','<end>')
    vocab.replace('##．','<unk>')
    
    print(vocab.word2idx)
    print(tokenizer.all_special_tokens)
    print(vocab.idx)
    print(tokenizer.vocab_size)
    """
    print(bert)