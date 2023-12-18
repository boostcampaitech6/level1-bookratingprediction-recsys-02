import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset


# book category를 전처리합니다.
def preproc_book_category(books):

    # parse category
    books['category'] = books['category'].apply(lambda x: eval(x)[0] if type(x)==str else np.nan)

    # all categories except NaN
    categories = books['category'].dropna().values

    # count all categories
    unique_categories, unique_categories_count = np.unique(categories, return_counts=True)

    # sort categories by count
    sort_index = np.argsort(unique_categories_count)[::-1]
    unique_categories = unique_categories[sort_index]
    unique_categories_count = unique_categories_count[sort_index]

    # get valid category index
    # 50개 이상의 category만 사용하고 나머지는 NaN으로 대체합니다.
    valid_index = np.where(unique_categories_count >= 50)[0]

    # useless category name to NaN 
    books['category'] = books['category'].apply(
        lambda x: x if x in unique_categories[valid_index] else np.nan)

    return books


# context_data age_map의 복사본입니다.
def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

# ml을 위한 데이터 준비 코드입니다.
def process_ml_data(users, books, ratings1, ratings2):

    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """

    # users
    #users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    #users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)

    # books
    # category encoding
    books = preproc_book_category(books)

    # ratings 데이터에 user, book의 메타 정보를 병합합니다
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)
    ratings = ratings.merge(users, on='user_id', how='left').merge(
        books[['isbn', 'publisher', 'language', 'book_author', 'category']], 
        on='isbn', how='left')

    ratings2 = ratings2.merge(users, on='user_id', how='left').merge(
        books[['isbn', 'publisher', 'language', 'book_author', 'category']], 
        on='isbn', how='left')

    ratings1 = ratings1.merge(users, on='user_id', how='left').merge(
        books[['isbn', 'publisher', 'language', 'book_author', 'category']], 
        on='isbn', how='left')

    # one hot encoder를 생성합니다.
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe = ohe.fit(ratings.drop(columns=['rating']))

    # one hot encoding을 수행합니다.
    X_train = ohe.transform(ratings1.drop(columns=['rating']))
    X_test = ohe.transform(ratings2.drop(columns=['rating']))

    feature_names = ohe.get_feature_names_out()

    return X_train, ratings1.rating, X_test, feature_names


def ml_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    X_train, y_train, X_test, feature_names = process_ml_data(users, books, train, test)


    data = {
            'X_train': X_train,
            'y_train': y_train,
            'test': X_test,
            'feature_names': feature_names,
            'users':users,
            'books':books,
            }

    return data

# ml train 데이터를 train, valid로 나눕니다.
def ml_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['X_train'],
                                                        data['y_train'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )

    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data
