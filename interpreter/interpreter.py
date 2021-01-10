import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn

from sentence_transformers import SentenceTransformer, util

# Models: paraphrase-xlm-r-multilingual-v1, quora-distilbert-base
model = SentenceTransformer('quora-distilbert-base')

classes = ['warmer', 'colder', 'neither']

PATHS = {'user_sentences': 'data/user_sentences.csv', # sentences by user with prediction, 
         'seed_sentences': 'data/seed_sentences.csv', # initial labeled sentences
         'test_set_labeled_sentences': 'data/test_set_labeled_sentences.csv' # test set of sentences with correct label
}

def measure_similarity(example_feedback, given_feedback):
    # returns how similar the given feedback is, to the example feedback
    sentence_embeddings = model.encode(
        [example_feedback, given_feedback], show_progress_bar=False)
    cos_sim = util.pytorch_cos_sim(*sentence_embeddings)
    return cos_sim


def interpret(feedback, example_sentences):
    similarity = {}
    for c in range(len(classes)):
        similarity[c] = max([measure_similarity(sent1, feedback)
                             for sent1 in example_sentences[c]['feedback']])

    # return the class with the lowest similarity difference
    return max(similarity, key=similarity.get)


def manual_testing():
    user_sentences = pd.read_csv(PATHS['user_sentences'])
    example_sentences_db = pd.read_csv(PATHS['seed_sentences'])
    example_sentences = {
        c: example_sentences_db[example_sentences_db['label'] == c] for c in range(len(classes))}
    user_input = 'None'
    while user_input != '':
        user_input = input('\n\nHow do you like your trip with Deutsche Bahn?\n')
        y_pred = interpret(user_input, example_sentences)
        print('==> You want it', classes[y_pred])
        if user_input != '':
            user_sentences = user_sentences.append(
                {'feedback': user_input, 'label': y_pred}, ignore_index=True)
    user_sentences.to_csv(PATHS['user_sentences'], index=False)


def interpret_feedback(user_input=''):
    model.encode(['a simple text'], show_progress_bar=True)
    example_sentences_db = pd.read_csv(PATHS['seed_sentences'])
    example_sentences = {
        c: example_sentences_db[example_sentences_db['label'] == c] for c in range(len(classes))}

    user_sentences = pd.read_csv(PATHS['user_sentences'])
    y_pred = interpret(user_input, example_sentences)
    if user_input != '':
        user_sentences = user_sentences.append(
            {'feedback': user_input, 'label': y_pred}, ignore_index=True)
        user_sentences.to_csv(PATHS['user_sentences'], index=False)

def evaluation():
    df = pd.read_csv(PATHS['test_set_labeled_sentences'])
    X = df['feedback']
    y = df['label']
    example_sentences_db = pd.read_csv(PATHS['seed_sentences'])
    example_sentences = {
        c: example_sentences_db[example_sentences_db['label'] == c] for c in range(len(classes))}

    y_pred = [interpret(sentence, example_sentences) for sentence in X]
    accuracy = np.mean(np.array(y) == np.array(y_pred))
    print(accuracy)
    plot_confusion_matrix(y, y_pred)


def plot_confusion_matrix(y, y_pred):
    matrix = sklearn.metrics.confusion_matrix(y, y_pred)
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix,
                                                  display_labels=classes)
    disp.plot()
    disp.ax_.set_title('Sentence Prediction')
    plt.show()

    # df = pd.DataFrame(matrix, index = classes,
    #              columns = ['predicted warmer', 'predicted colder'])
    #plt.figure(figsize = (7,5))
    #sns.heatmap(df, annot=True)
    # plt.show()

# evaluation()
# manual_testing()
