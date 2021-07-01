import argparse
import numpy as np
import pandas as pd
import re
from newspaper import Article


# ending sentence characters
BOUNDARYSYMBOL = ['.', '!', '?', u'\u2026', u'…']
# space characters
WHITE_SPACE = ' '
# ... characters
ELLIPSIS = u'\u2026'

# function to check if a character is ending sentence character
def isBoundary(i, text):
    # if a character is file ending
    if i == len(text)-1:
        return 1

    # return 0 if both i-1 and i+1 characters are of type numeric
    if re.match('[0-9]', text[i-1]) and re.match('[0-9]', text[i+1]):
        return 0

    # return 1 if i-1 character is numerical and i+1 one is not
    if re.match('[0-9]', text[i-1]) and re.match('[^0-9]', text[i+1]):
        return 1

    # return 1 if i -1 character if lowercase and i+1 one is uppercase
    if text[i - 1].islower() and text[i + 1].isupper():
        return 1

    # return 0 if both i-1 and i+1 characters are uppercase
    if text[i - 1].isupper():
        return 0

    if text[i + 2].islower():
        return 0

    # return 1 if i+1 character is space
    if re.match('\s', text[i+1]):
        return 1

    return 0

# split into sentences


def split_sentence(text):
    text = text.strip()
    text = text.replace('"', '')
    text = text.replace('\t', u'.')
    text = text.replace('\v', u'.')
    text = text.replace('\r', u'.')
    text = text.replace('\n', u'.')
    text = text.replace('\r\n', u'.')
    text = text.replace(u'...', u'.')
    text = text.replace(u'..', u'.')
    text = text.replace(u'…', u'.')
    text = text.replace('Dân trí', '')

    sents = []

    i = 0

    begin = 0

    # loop over text
    while i < len(text):
        for sym in BOUNDARYSYMBOL:
            # encountering an ending character
            if sym == text[i]:
                # check if this position is sentence ending
                if isBoundary(i, text):
                    sents.append(text[begin:i].strip())
                    begin = i+1
        i += 1

    # words that do not belong to a sentence.
    word_to_remove_sent = ['ảnh:', 'nguồn:', 'ảnh minh họa']

    for sent in sents:
        for word in word_to_remove_sent:
            if word in sent.lower():
                sents.remove(sent)
        if sent == "":
            sents.remove(sent)
    return sents

# function to get content from a newspaper


def get_content(url):
    article = Article(url=url, language='vi')
    article.download()
    article.parse()
    new_text = article.text

    split_sents = split_sentence(new_text)
    return split_sents
