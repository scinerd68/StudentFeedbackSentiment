from datasets import Dataset
from tqdm import tqdm
from extract_phrases import model, get_pos, extract_opinion_phrases
from semantic_orientation import semantic_orientation, semantic_orientation2


def eval_test_data(example):
    """Add semantic orientation to every examples in dataset"""
    annotated = model.annotate_text(example['text'])
    sentence, pos_tags = get_pos(annotated)
    extracted_phrases, extracted_pos = extract_opinion_phrases(sentence, pos_tags)
    if len(extracted_phrases) == 0:
        example['score'] = 0
    else:
        score = semantic_orientation(extracted_phrases)
        # score = semantic_orientation2(extracted_phrases)
        example['score'] = score
    return example


def check_example_with_phrase(example):
    """
    Filter sentence that does not contain any phrases according to annotator
    """
    annotated = model.annotate_text(example['text'])
    sentence, pos_tags = get_pos(annotated)
    extracted_phrases, _ = extract_opinion_phrases(sentence, pos_tags)
    if len(extracted_phrases) == 0:
        return False
    return True


def predicted_label(example):
    """Output label based on semantic orientation"""
    if example['scores'] > 0 :
        example['predicted'] = 2
    elif example['scores'] < 0:
        example['predicted'] = 0
    else:
        example['predicted'] = -1
    return example


if __name__ == "__main__":
    # Load test data
    with open('/content/drive/MyDrive/_UIT-VSFC/test/sents.txt') as f:
        lines = [line.rstrip('\n') for line in f]

    with open('/content/drive/MyDrive/_UIT-VSFC/test/sentiments.txt') as f:
        sentiments = [int(line.rstrip('\n')) for line in f]

    test_data = {'text': lines, 'sentiments': sentiments}
    test_data = Dataset.from_dict(test_data)
    # Filter neutral examples (167 examples) since the algorithm cannot handle this case
    pos_neg_only = test_data.filter(lambda x: x['sentiments'] != 1)
    # Filter neutral examples without any extracted phrases
    pos_neg_only_with_phrase = pos_neg_only.filter(check_example_with_phrase)

    # Add semantic orientation to dataset and predict sentiment
    scores = []
    for example in tqdm(pos_neg_only_with_phrase):
        score = eval_test_data(example)['score']
        scores.append(float(score))
    pos_neg_only_scored = pos_neg_only_with_phrase.add_column("scores", scores)
    pos_neg_only_scored = pos_neg_only_scored.map(predicted_label)

    # Accuracy
    correct = pos_neg_only_scored.filter(lambda x: x['sentiments'] == x['predicted'])
    print('Accuracy:', len(correct) / len(pos_neg_only_scored))