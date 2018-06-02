import requests
from lxml import html
import pandas as pd
import time

def get_url(domain, page):

    return 'https://stackoverflow.com/questions/tagged/{tag}?page={npage}&sort=newest&pagesize=50'.format(tag=domain,
                                                                                                          npage=page)


def get_id_question_and_excerpt(div):

    question = [k.text for k in div.xpath('div/h3/a') if 'question-hyperlink' in k.classes]
    excerpt = [k.text for k in div.xpath('div/div') if 'excerpt' in k.classes]

    return [div.attrib['id']], question, excerpt


def get_contents(url):

    ping = requests.get(url)
    tree = html.fromstring(ping.content)
    q = []
    e = []
    i = []
    for k in tree.xpath('//div'):
        if 'question-summary' in k.classes:
            qid, question, excerpt = get_id_question_and_excerpt(k)
            q += question
            e += excerpt
            i += qid

    return pd.DataFrame({'id': i, 'question': q, 'excerpt': e})


def scrape_tag(tag, population):

    scrape_df = pd.DataFrame({})

    for i in range(population):
        i += 1
        url = get_url(tag, i)
        scrape_df = pd.concat([scrape_df, get_contents(url)]).reset_index(drop=True)
        scrape_df.drop_duplicates(inplace=True)
        time.sleep(1)

    return scrape_df

if __name__ == '__main__':
    tag = 'sql'
    pipou = scrape_tag(tag, 1000)
    pipou.to_csv(tag + '.csv', index=False, columns=['id', 'question', 'excerpt'])