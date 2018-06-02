import pandas as pd
from gensim.models import doc2vec, Doc2Vec


def df_to_doc(df, id_field, doc_field):

    return df.apply(lambda l: doc2vec.LabeledSentence(tags=[l[id_field]], words=l[doc_field].split(' ')), axis=1)

questions = pd.read_csv('../Data/questions_sample.csv')

doc = df_to_doc(questions, 'id', 'question')

model = Doc2Vec(doc, size=100, window=2, alpha=.3, min_alpha=.01, min_count=5)

print('pipou')