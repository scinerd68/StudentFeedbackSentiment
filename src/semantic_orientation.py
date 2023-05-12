import torch
import torch.nn.functional as F
import gensim


word2vec = gensim.models.KeyedVectors.load_word2vec_format('models/custom_w2v/custom_w2v.bin',
                                                           binary=True)


def word_embed(word):
    """Get word embeddings from custom models"""
    try:
        return torch.Tensor(word2vec[word])
    except KeyError:
        return None


def phrase_embed(phrase):
    """Get phrase embeddings by averaging all word embeddings in phrase"""
    embed = word2vec.vector_size
    count = 0
    for word in phrase:
        # Ignore words not in vocabulary
        if word_embed(word) is None:
            continue
        
        embed += word_embed(word)
        count += 1
    return embed / count


def semantic_orientation(phrases):
    """
    First method to calculate sentence semantic:
    - Compute phrase embedding (by averaging word embeddings in phrase)
    - Compute phrase semantic orientaion
    - Sentence semantic orientation is the sum of semantic orientations of all phrases
    """
    pos_anchor = word_embed('tốt')
    neg_anchor = word_embed('kém')

    sentence_semantic = 0
    for phrase in phrases:
        phrase_embedded = phrase_embed(phrase)
        pos_similarity = F.cosine_similarity(phrase_embedded, pos_anchor, dim=0)
        neg_similarity = F.cosine_similarity(phrase_embedded, neg_anchor, dim=0)
        phrase_semantic = pos_similarity - neg_similarity
        sentence_semantic += phrase_semantic
    return sentence_semantic


def semantic_orientation2(phrases):
    """
    Second method to calculate sentence semantic:
    - Do not compute phrase embedding (by averagin word embeddings in phrase)
    - Sentence semantic orientation is the sum of semantic orientations of all words
    in sentence
    """
    pos_anchor = word_embed('tốt')
    neg_anchor = word_embed('kém')

    sentence_semantic = 0
    for phrase in phrases:
        for word in phrase:
            word_embedded = word_embed(word)
            if word_embedded is None:
                continue
            pos_similarity = F.cosine_similarity(word_embedded, pos_anchor, dim=0)
            neg_similarity = F.cosine_similarity(word_embedded, neg_anchor, dim=0)
            word_semantic = pos_similarity - neg_similarity
            sentence_semantic += word_semantic
            
    return sentence_semantic




