from datasets import Dataset
from gensim.models import Word2Vec
from extract_phrases import get_pos, model


def tokenize_sentence(example):
    """
    Apply word segment to a sentence.
    E.g: slide giáo trình đầy đủ
    --> ['slide', 'giáo_trình', 'đầy_đủ', '.']
    """
    annotated = model.annotate_text(example['text'])
    words, _ = get_pos(annotated)
    example['words'] = words
    return example


if __name__ == "__main__":
    # Load training data
    with open('/content/drive/MyDrive/_UIT-VSFC/train/sents.txt') as f:
        lines = [line.rstrip('\n') for line in f]

    with open('/content/drive/MyDrive/_UIT-VSFC/train/sentiments.txt') as f:
        sentiments = [int(line.rstrip('\n')) for line in f]

    train_data = {'text': lines, 'sentiments': sentiments}
    train_data = Dataset.from_dict(train_data)
    train_data = train_data.map(tokenize_sentence)
    
    # Create word2vec embedding with dimension 100 based on context window size of 5
    custom_w2v = Word2Vec(sentences=train_data['words'], size=100,
                          window=5, min_count=1, workers=2)
    custom_w2v.wv.save_word2vec_format('custom_w2v.bin', binary=True)