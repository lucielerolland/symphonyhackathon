import pandas as pd
from gensim.models import doc2vec, Doc2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
import logging
import csv
import os
import pickle
import ast
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_data():
    questions = pd.read_csv('../Data/questions.csv')

    doc = {}
    doc['exc'] = list(df_to_doc(questions, 'excerpt'))
    doc['q'] = list(df_to_doc(questions, 'question'))

    return questions, doc


def df_to_doc(df, doc_field):
    df.reset_index(inplace=True)

    return df.apply(lambda l: doc2vec.TaggedDocument(tags=[l['index']], words=l[doc_field].strip().split(' ')), axis=1)


def doc_to_X(model, index):

    X = model.docvecs[index[0]]
    for j in index[1:]:
        X = np.vstack((X, model.docvecs[j]))

    return X


def tune(search_iter, file):

    questions, doc = load_data()

    for w in range(search_iter):
        corpus = np.random.choice(['exc', 'q'], 1, p=[.5, .5])[0]
        size = np.random.randint(4, 40) * 5
        dm = np.random.binomial(1, .5, 1)[0]
        window = np.random.randint(1, 15)
        alpha = (3) ** (np.random.randint(-10, 3))
        alpha_min = alpha / (3) ** (np.random.randint(1, 10))
        min_count = np.random.randint(0, 10)
        sample = 1 ** (np.random.randint(-5, 0))
        n_iter = np.random.randint(5, 15)

        alpha_infer = (3) ** (np.random.randint(-10, 3))
        alpha_min_infer = alpha / (3) ** (np.random.randint(1, 10))
        step_infer = np.random.randint(2, 15)

        model = Doc2Vec(doc[corpus], vector_size=size, dm=dm, window=window, min_count=min_count, seed=seed,
                        alpha=alpha,
                        min_alpha=alpha_min, sample=sample, iter=n_iter)

        # Metric 1: avg cosine similarity of a vector with its inference by the model

        cosim = []
        for i in range(len(questions)):
            a = model.docvecs[i]
            b = model.infer_vector(doc[corpus][i].words, alpha=alpha_infer, min_alpha=alpha_min_infer, steps=step_infer)
            cosim.append(cosine_similarity(np.vstack((a, b)))[0, 1])

        mean_cosim = sum(cosim) / len(cosim)

        # Metric 2: how well do vectors predict a question's tag

        train_test = np.random.permutation(len(questions))
        cutoff = int(.9 * len(questions))
        train = train_test[0:cutoff]
        test = train_test[cutoff:len(questions)]
        model_lr = LogisticRegression()
        X_train = doc_to_X(model, train)
        model_lr.fit(X_train, questions['tag_id'].ix[train])
        X_test = doc_to_X(model, test)
        y_pred_test = model_lr.predict(X_test)
        precision, recall, fscore, support = precision_recall_fscore_support(questions['tag_id'].ix[test], y_pred_test)

        if not os.path.isfile("param_exploration.csv"):
            exports_name = ['corpus', 'size', 'dm', 'window', 'min_count', 'alpha', 'alpha_min', 'sample', 'n_iter',
                            'alpha_infer', 'alpha_min_infer', 'step_infer', 'mean_cosim', 'precision', 'recall',
                            'fscore',
                            'support']

            with open(file, "w") as f:
                writer = csv.writer(f)
                writer.writerows([exports_name])

        exports = [corpus, size, dm, window, min_count, alpha, alpha_min, sample, n_iter, alpha_infer, alpha_min_infer,
                   step_infer, mean_cosim, precision, recall, fscore, support]

        with open(file, "a") as f:
            writer = csv.writer(f)
            writer.writerows([exports])


def explore_tuning(input_file, rank, n_iter, output_file):

    output_file += '_' + str(rank) + '.csv'

    questions, doc = load_data()

    row = input_file.iloc[[rank]]
    model = Doc2Vec(doc[row['corpus'].values[0]], vector_size=row['size'].values[0], dm=row['dm'].values[0], window=row['window'].values[0], min_count=row['min_count'].values[0], seed=seed,
                    alpha=row['alpha'].values[0], min_alpha=row['alpha_min'].values[0], sample=row['sample'].values[0], iter=row['n_iter'].values[0])
    train_test = np.random.permutation(len(questions))
    cutoff = int(.9 * len(questions))
    train = train_test[0:cutoff][0:50000]
    test = train_test[cutoff:len(questions)][0:10000]

    X_train = doc_to_X(model, train)
    X_test = doc_to_X(model, test)

    Y = questions['askee_id']

    Y_train = Y.ix[train]
    Y_test = Y.ix[test]

    for iterations in range(n_iter):

        if iterations % 10 == 0:
            print('Running model', row['index'], 'iter', iterations)

        penalty = 'l2'
        c = np.random.uniform(0, 1)
        multiclass = np.random.choice(('ovr', 'multinomial'), 1)[0]
        tol = 10**(-np.random.randint(2, 10))
        solver = 'newton-cg'
        max_iter = 25*np.random.randint(4, 12)

        model_lr = LogisticRegression(penalty=penalty, C=c, multi_class=multiclass, tol=tol, solver=solver,
                                      max_iter=max_iter, random_state=seed)

        model_lr.fit(X_train, Y_train)
        y_pred_test = model_lr.predict(X_test)
        precision, recall, fscore, support = precision_recall_fscore_support(Y_test, y_pred_test)

        if not os.path.isfile(output_file):
            exports_name = ['model_index', 'penalty', 'c', 'multiclass', 'tol', 'max_iter', 'train', 'precision', 'recall',
                            'fscore', 'support']

            with open(output_file, "w") as f:
                writer = csv.writer(f)
                writer.writerows([exports_name])

        exports = [row['index'].values[0], penalty, c, multiclass, tol, max_iter, train, precision, recall, fscore, support]

        with open(output_file, "a") as f:
            writer = csv.writer(f)
            writer.writerows([exports])

        print('written at', output_file)


def save_final_model(param_explo, lr_explo, row):
    questions, doc = load_data()
    d2v_tuning = pd.read_csv(param_explo)
    lr_tuning = pd.read_csv(lr_explo)
    lr_best_row = lr_tuning.iloc[[row]]
    d2v_best_row = d2v_tuning.iloc[[lr_best_row['model_index'].values[0][0]]]
    d2v_model = Doc2Vec(doc[d2v_best_row['corpus'].values[0]], vector_size=d2v_best_row['size'].values[0], dm=d2v_best_row['dm'].values[0],
                    window=d2v_best_row['window'].values[0], min_count=d2v_best_row['min_count'].values[0], seed=seed,
                    alpha=d2v_best_row['alpha'].values[0], min_alpha=d2v_best_row['alpha_min'].values[0], sample=d2v_best_row['sample'].values[0],
                    iter=d2v_best_row['n_iter'].values[0])

    d2v_model.save('d2v')

    lr_model = LogisticRegression(penalty=lr_best_row['penalty'].values[0], C=lr_best_row['c'].values[0], multi_class=lr_best_row['multiclass'].values[0],
                                  tol=float(lr_best_row['tol'].values[0]), solver='newton-cg', max_iter=lr_best_row['max_iter'].values[0],
                                  random_state=seed)

    train = ast.literal_eval(lr_best_row['train'].values[0])
    X_train = doc_to_X(d2v_model, train)
    Y_train = questions['askee_id'].ix[train]
    lr_model.fit(X_train, Y_train)

    pickle.dump(lr_model, open('lr_model.pkl', 'w'))


if __name__ == '__main__':
    mode = 'estim'
    seed = 1010

    if mode == 'estim':
        tuning = pd.read_csv("param_exploration.csv")
        tuning = tuning.sort_values('mean_cosim', ascending=False).reset_index()
        path = "lr_exploration"
        rank = 1
        n_iter = 100
        explore_tuning(tuning, rank, n_iter, path)
    else:
        param_explo = "param_exploration.csv"
        lr_explo = "lr_exploration.csv"
        row = 1
        save_final_model(param_explo, lr_explo, row)
