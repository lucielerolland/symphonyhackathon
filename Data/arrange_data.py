import pandas as pd
import numpy as np
from collections import defaultdict
import datetime as dt


def gen_random_date_vector(size):
    date_vec = []
    for i in range(size):
        year = np.random.random_integers(2015, 2018, 1)[0]
        month = np.random.random_integers(1, 12, 1)[0]
        day = np.random.random_integers(1,28,1)[0]
        hour = np.random.random_integers(9, 17)
        minute = np.random.random_integers(0, 59)
        second = np.random.random_integers(0, 59)

        date_vec.append(dt.datetime(year, month, day, hour, minute, second))

    return date_vec

q_users = ['Ann', 'Bill', 'Callie', 'David', 'Elena']
a_users = ['Jimmy', 'Clement', 'Wei', 'Lucie']
q_status = ['opened', 'answer_fail', 'answer_success', 'documented']

tags = ['python', 'java', 'javascript', 'sql']

proba_dic = defaultdict(dict)

questions = pd.DataFrame({})

for i, k in enumerate(tags):
    proba_dic['rank'][k] = str(i)
    proba_dic['askee'][k] = np.ones(4)*.01
    proba_dic['askee'][k][i] = .97
    proba_dic['asker'][k] = np.ones(5)*.15
    proba_dic['asker'][k][i] = .3
    proba_dic['asker'][k][i+1] = .25
    df = pd.read_csv(k+'.csv')
    df['tag'] = k       # for testing purposes
    questions = pd.concat([questions, df])

questions['id'] = questions.apply(lambda l: int(l['id'].split('-')[2] + proba_dic['rank'][l['tag']]), axis=1)

assert len(questions) == len(set(questions['id']))

questions['asker'] = questions['tag'].apply(lambda l: np.random.choice(q_users, 1, p=proba_dic['asker'][l])[0])
questions['askee'] = questions['tag'].apply(lambda l: np.random.choice(a_users, 1, p=proba_dic['askee'][l])[0])
questions['status'] = np.random.choice(q_status, len(questions))
questions['good_match'] = questions['status'].apply(lambda l: l in ['answered', 'documented'])
questions['datetime_asked'] = gen_random_date_vector(len(questions))
questions['datetime_latest'] = questions['datetime_asked'].apply(lambda l: l+dt.timedelta(days= np.random.exponential(1)))

questions.to_csv('questions.csv', index=False)
