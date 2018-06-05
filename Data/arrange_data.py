import pandas as pd
import numpy as np
from collections import defaultdict
import datetime as dt

np.random.seed(1010)


def gen_random_date_vector(size):
    date_vec = []
    for i in range(size):
        year = np.random.randint(2015, 2018+1)
        month = np.random.randint(1, 12+1)
        day = np.random.randint(1,28+1)
        hour = np.random.randint(9, 17+1)
        minute = np.random.randint(0, 59+1)
        second = np.random.randint(0, 59+1)

        date_vec.append(dt.datetime(year, month, day, hour, minute, second))

    return date_vec

q_users = ['Ann', 'Bill', 'Callie', 'David', 'Elena']
a_users = ['Jimmy', 'Clement', 'Wei', 'Lucie']
a_users_id = {'Jimmy': 0, 'Clement':1, 'Wei':2, 'Lucie':3}
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
    df['tag_id'] = i
    questions = pd.concat([questions, df]).reset_index(drop=True)

questions['id'] = questions.apply(lambda l: int(l['id'].split('-')[2] + proba_dic['rank'][l['tag']]), axis=1)

assert len(questions) == len(set(questions['id']))

questions['asker'] = questions['tag'].apply(lambda l: np.random.choice(q_users, 1, p=proba_dic['asker'][l])[0])
questions['askee'] = questions['tag'].apply(lambda l: np.random.choice(a_users, 1, p=proba_dic['askee'][l])[0])
questions['askee_id'] = questions['askee'].apply(lambda l: a_users_id[l])
questions['status_prob'] = questions.apply(lambda l: [.02, (1-proba_dic['askee'][l['tag']][l['askee_id']]-.02),
                                                     proba_dic['askee'][l['tag']][l['askee_id']]/2,
                                                      proba_dic['askee'][l['tag']][l['askee_id']]/2], axis=1)
questions['status'] = questions['status_prob'].apply(lambda l: np.random.choice(q_status, 1, p=l))
questions['good_match'] = questions['status'].apply(lambda l: l in ['answered', 'documented'])
questions['datetime_asked'] = gen_random_date_vector(len(questions))
questions['datetime_latest'] = questions['datetime_asked'].apply(lambda l: l+dt.timedelta(days=np.random.exponential(1)))

questions.ix[np.random.permutation(len(questions))[0:5000], :].to_csv('questions_sample.csv', index=False)

questions.to_csv('questions.csv', index=False)
