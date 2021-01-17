# Deutsche Bahn feedback processing
Classify natural language feedback to extract automatically processable actions using sentence embedding.

![example of usage](https://github.com/atarust/db_feedback_processing/blob/main/example_output.jpg)

## tl;dr: 
In Deutsche Bahn trains,  you can [give feedback by scanning a qr-code](https://inside.bahn.de/qr-code-umfrage/), which is interpreted in real time. I fast-prototyped this functionality for using [sentence-embedding](https://github.com/UKPLab/sentence-transformers) developed in the UKPLab at TU Darmstadt. Currently I have a about 80% chance of classifying correctly, whether the user feedback wants to increase or decrease the temperature in the cabin. Next steps are to evaluate that and to plot some results. 

Try out here: [https://jkapitzke.de/rtf](https://jkapitzke.de/rtf)

This is a highly unofficial project.

## What it does: 
Given a few examples of sentences that want it warmer and a few examples of sentences that want it colder, compare the given feedback with all sentences, by comparing the distance between the  transformer embeddings of the sentences. Then, classify the feedback into warmer and colder. If decision is too close or embedding distances are high, classify label 'neither'.
 #classification #nlp #semantic similarity

## How it works: 
For labels such as 'warmer' or 'colder' I collect some sample feedback - 'seed sentences'. I use SBERT (https://github.com/UKPLab/sentence-transformers) sentence embedding to embed both the sentence from the user and the embedding of the labels. I then measure the distance between the user sentence and all the example sentences. I return the label which has a sentence with the closest distance to the user sentence.

### Technical Details:
SBERT is a to encode natural language sentences into embeddings. Embeddings are dense number vectors, that represent each sentence. Sentences with semantic similarity have, if the language encoder does a good job, similar vectors. I measure the distance between vectors with cosine similarity. SBERT [performs very well on Semantic Textual Semilarity Benchmarks](https://arxiv.org/pdf/1908.10084.pdf). 

Currently, I am naively chosing the label with the minimal distance between feedback and labeled sentence, basically 1-nearest-neighbor. One could increase the number to get improved robustness.

Also, in the future the selection of labeled sentences may be vastly improved through some kind of active learning aproach: The user types his feedback, the correct intention is predicted and shown to the user. The user may then select a checkbox, if the prediction was correct, in which case the user sentence is added to the pool of labeled sentences.
 #transformer #sentence-embedding #cosine-distance #kNN #active-learning

## How I evaluate: 
I collect some natural language feedback sentences from my friends who are either too cold or too warm and label them by hand. Then, I check the accuracy and confusion matrix of the classifier.

## Visualize: 
- Confusion Matrix of classification
- Plot similarity-distances: Plot for each feedback labeled colder (blue), warmer (red), or neither (black), the min/mean/median similarity-distance to the seed sentences. 

## Data Sets
Data contains feedback sentences and class labels of user intention ( 0: warmer, 1:colder, 2:neither).

- data/seed_sentences.csv: initial hand-labeled feedback sentences
- data/user_sentences.csv: feedback given by user, with prediction. Logging all user inputs and the classifiers prediction.
- data/test_set_labeled_sentences.csv: collected feedback sentences test set with correct labels

## How do I run it (linux)
Try out on [https://jkapitzke.de/rtf](https://jkapitzke.de/rtf) or play with it in [google colab](https://colab.research.google.com/drive/15o5TomRotSp-8SFZXT8yxwJsZ1kqywXt?usp=sharing).

- Clone repository
- pip install cython torch numpy pandas seaborn matplotlib sklearn sentence-transformers
- python main.py

## TODOs
* [x] use model that provides German language support
* [x] come up with example sentences for classification
* [ ] tune parameter when to classify 'neither'
* [ ] collect good example sentences from friends for evaluation
* [ ] try other fine-tuned models or finetune one yourself and evaluate
* [ ] come up with better sentences through active learning: log all inputs and label the feedback for future example sentences
* [ ] collect user example sentences to csv
* [ ] make a web app with a button saying "correct" and "not correct", so that it may learn from user input easily. For example, the user may type 'It is too cold.', the answer could be 'Would you like to increase heating? yes/no'. If user accepts yes, the intent was successfully recognized and the feedback can be added as a labeled example to the seed sentences.
* [ ] add cache decorator around encoder to save computation: @lru_cache(maxsize=None) 

