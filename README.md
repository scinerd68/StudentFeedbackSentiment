# Student Feedback Sentiment Analysis
## Dataset
[Vietnames Students' Feedback Corpus](https://nlp.uit.edu.vn/datasets/) from University of Information Technology - VNU HCM
## Method
Apply the unsupervised sentiment analysis with some modifications from paper ["Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of Reviews"](https://arxiv.org/abs/cs/0212032) to determine the the sentiment for each sentence in the dataset (positive or negative)

Overall Pipeline:
- Apply word segmentation and part-of-speech tagging for each example in the dataset
- Extract phrases from each example that have potential for opinion expression. Phrases with following forms are chosen:
  - Noun + Adjective
  - Verb + Adjective
  - Adverb + Adjective
  - Adverb + Verb
  - Verb + Adverb
- Determine semantic orientation (SO) of each phrases according to the following formula: *SO(t) = sim(t, ‘tốt’) - sim(t, ‘kém’)*
- Semantic orientation SO(d) of a sentence is the sum of semantic orientations of all phrases extracted from the sentence.

In this repo, similarity between phrases is computed using Word2Vec, as opposed to the PMI in the original paper.

## Result
The below table represents results from all methods:
<table>
    <thead>
        <tr>
            <th colspan=1>PMI</th>
            <th colspan=4>Word2Vec</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=1 rowspan=2>Using Training Data</td>
            <td colspan=2 styles="">PhoW2V</td>
            <td colspan=2>Custom Model</td>
        </tr>
        <tr>
            <td colspan=1>Method 1</td>
            <td colspan=1>Method 2</td>
            <td colspan=1>Method 1</td>
            <td colspan=1>Method 2</td>
        </tr>
        <tr>
            <td colspan=1>65.7</td>
            <td colspan=1>55.4</td>
            <td colspan=1>55.1</td>
            <td colspan=1>80.5</td>
            <td colspan=1>79.9</td>
        </tr>
    </tbody>
</table>
