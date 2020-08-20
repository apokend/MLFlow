from collections import Counter
from string import punctuation
import numpy as np
import pandas as pd
import os


def review_formatting(reviews):
    all_reviews=list()
    for text in reviews:
        lower_case = text.lower()
        words = lower_case.split()
        reformed = [word for word in words]
        reformed_test=list()
        for word in reformed:
            reformed_test.append(word)
        reformed = " ".join(reformed_test)
        punct_text = "".join([ch for ch in reformed if ch not in punctuation])
        all_reviews.append(punct_text)
    all_text = " ".join(all_reviews)
    all_words = all_text.split()
    return all_reviews, all_words

def encode_reviews(reviews, vocab_to_int):
    """
    encode_reviews function will encodes review in to array of numbers
    """
    all_reviews=list()
    for text in reviews:
        text = text.lower()
        text = "".join([ch for ch in text if ch not in punctuation])
        all_reviews.append(text)
    encoded_reviews=list()
    for review in all_reviews:
        encoded_review=list()
        for word in review.split():
            if word not in vocab_to_int.keys():
                encoded_review.append(0)
            else:
                encoded_review.append(vocab_to_int[word])
        encoded_reviews.append(encoded_review)
    return encoded_reviews

def pad_sequences(encoded_reviews, sequence_length=250):
    '''
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features=np.zeros((len(encoded_reviews), sequence_length), dtype=int)

    for i, review in enumerate(encoded_reviews):
        review_len=len(review)
        if (review_len<=sequence_length):
            zeros=list(np.zeros(sequence_length-review_len))
            new=zeros+review
        else:
            new=review[:sequence_length]
        features[i,:]=np.array(new)
    return features


def preprocess(reviews, vocab_to_int):
    """
    This Function will tranform reviews in to model readable form
    """
    print('Step 1')
    formated_reviews, all_words = review_formatting(reviews)
    print('Step 2')
    encoded_reviews=encode_reviews(formated_reviews, vocab_to_int)
    print('Step 3')
    features=pad_sequences(encoded_reviews, 250)
    print('Step 4')
    return features
