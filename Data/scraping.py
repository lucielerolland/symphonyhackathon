import requests
from lxml import html
import pandas as pd
import time
import re


def get_url(domain, page):

    return 'https://stackoverflow.com/questions/tagged/{tag}?page={npage}&sort=newest&pagesize=50'.format(tag=domain,
                                                                                                          npage=page)


def get_id_question_and_excerpt(div):

    question = [k.text for k in div.xpath('div/h3/a') if 'question-hyperlink' in k.classes]
    if '[duplicate]' in question:
        question = re.sub('\[duplicate\]', '', question).strip()
        dup_id, dup_name, question_dup = get_duplicate_from_link([k.get('href') for k in div.xpath('div/h3/a') if 'question-hyperlink' in k.classes][0])

    else:
        dup_id = ['']
        dup_name = ['']
        question_dup = ['']
    excerpt = [k.text for k in div.xpath('div/div') if 'excerpt' in k.classes]

    return [div.attrib['id']], question, excerpt, dup_id, dup_name, question_dup


def get_question_from_tree(tree):

    question_text = ''

    for div in tree.xpath('//div'):
        if 'question' in div.classes:
            for question in div.xpath("div/div/div"):
                if "post-text" in question.classes:
                    for p in question.xpath("p"):
                        question_text += p.text

    return question_text


def get_duplicate_from_link(url):

    ping_dup = requests.get(url)
    tree_dup = html.fromstring(ping_dup.content)

    dup_id = ''
    dup_name = ''
    question_dup = ''

    links = tree_dup.xpath("//span[text()='possible duplicate of ']/a")
    for link in links:
        dup_link = link.attrib['href']
        dup_name = link.text  # html content of product
        ping_dup2 = requests.get(dup_link)
        tree_dup2 = html.fromstring(ping_dup2.content)
        question_dup = get_question_from_tree(tree_dup2)
        dup_id = re.findall('([0-9]+)', dup_link)[0]

    return [dup_id], [dup_name], [question_dup]


def get_contents(url):

    ping = requests.get(url)
    tree = html.fromstring(ping.content)
    q = []
    e = []
    i = []
    dq = []
    de = []
    di = []
    for k in tree.xpath('//div'):
        if 'question-summary' in k.classes:
            qid, question, excerpt, did, dquestion, dexcerpt = get_id_question_and_excerpt(k)
            q += question
            e += excerpt
            i += qid
            dq += dquestion
            de += dexcerpt
            di += did

    return pd.DataFrame({'id': i, 'question': q, 'excerpt': e, 'id_duplicate': di, 'question_duplicate': dq,
                         'excerpt_duplicate': de})


def scrape_tag(tag, population):

    scrape_df = pd.DataFrame({})

    for i in range(population):
        i += 1
        url = get_url(tag, i)
        scrape_df = pd.concat([scrape_df, get_contents(url)]).reset_index(drop=True)
        scrape_df.drop_duplicates(inplace=True)
        time.sleep(.5)

    return scrape_df

if __name__ == '__main__':
    tag = 'python'
    pipou = scrape_tag(tag, 1000)
    pipou.to_csv(tag + '_dup.csv', index=False, columns=['id', 'question', 'excerpt', 'id_duplicate',
                                                         'question_duplicate', 'excerpt_duplicate'])