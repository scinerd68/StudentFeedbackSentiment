import py_vncorenlp


model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos"], save_dir='/VnCoreNLP')


def get_pos(annotated):
    sentence = [word['wordForm'] for word in annotated[0]]
    pos = [word['posTag'] for word in annotated[0]]
    return sentence, pos


def extract_opinion_phrases(sentence, pos_tags):
    PATTERNS = [['N', 'A'],
                ['V', 'A'],
                ['R', 'A'],
                ['R', 'V'],
                ['V', 'R']]
                # ['N', 'V']] # May or may not need ['N', 'V]
    extracted_phrases = []
    extracted_pos = []
    for word_idx, _ in enumerate(sentence):
        phrase = sentence[word_idx:word_idx + 2]
        pos = pos_tags[word_idx: word_idx + 2]
        if pos in PATTERNS:
            extracted_phrases.append(phrase)
            extracted_pos.append(pos)
    return extracted_phrases, extracted_pos