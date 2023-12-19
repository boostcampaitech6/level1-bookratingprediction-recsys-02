import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def preprocess_user(args, users):
    ######## user preprocessing start
    print("|-user preprocessing [start]")

    if 'location' in args.preprocess_user:
        ######## 1. remove location special symbols
        __remove_location_special_symbols__(users)

        ######## 2. location split
        __split_user_location__(users)

        ######## 3. fill unknown value with nan
        __fill_unknown_value_with_nan__(users)

        ######## 4. fill state and country from city
        __fill_state_country_from_city__(users)
    
    ######## user preprocessing end
    print("|-user preprocessing [end]")
    

def preprocess_book(args, books):
    ######## book preprocessing start
    print("|-book preprocessing [start]")

    # columns = ['isbn', 'book_title', 'book_author', 'year_of_publication', 'publisher', 'img_url', 'language', 'category', 'summary', 'img_path']
    if 'isbn' in args.preprocess_book:
        pass

    if 'book_title' in args.preprocess_book:
        pass

    if 'book_author' in args.preprocess_book:
        pass

    if 'year_of_publication' in args.preprocess_book:
        pass

    if 'publisher' in args.preprocess_book:
        pass

    if 'img' in args.preprocess_book:
        pass

    if 'language' in args.preprocess_book:
        pass

    if 'category' in args.preprocess_book:
        pass

    if 'summary' in args.preprocess_book:
        pass

    ######## book preprocessing end
    print("|-book preprocessing [end]")

def __fill_state_country_from_city__(users):
    print(" |-fill state and country from city [start]")
    city2state = {}
    city2country = {}

    for _, user in users.iterrows():
        city = user['location_city']
        if pd.isna(city) or city == 'na' or city == 'nan':
            continue
        if city not in city2state:
            state = user['location_state']
            if not pd.isna(state) and state != 'na' and state != 'nan':
                city2state[city] = state
        if city not in city2country:
            country = user['location_country']
            if not pd.isna(country) and country != 'na' and country != 'nan':
                city2country[city] = country
    
    nan_state_rows = users['location_state'].isna() & users['location_city'].notna()
    nan_country_rows = users['location_country'].isna() & users['location_city'].notna()

    users.loc[nan_state_rows, 'location_state'] = users.loc[nan_state_rows, 'location_city'].map(city2state)
    users.loc[nan_country_rows, 'location_country'] = users.loc[nan_country_rows, 'location_city'].map(city2country)

    print(" |-fill state and country from city [end]")

def __fill_unknown_value_with_nan__(users):
    print(" |-fill unknown value with nan [start]")
    users['location'] = users['location'].replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users['location'] = users['location'].replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.
    print(" |-fill unknown value with nan [end]")

def __remove_location_special_symbols__(users, regex=r'[^0-9a-zA-Z:,]'):
    print(" |-remove location special symbols [start]")
    users['location'] = users['location'].str.replace(regex, '', regex=True) # 특수문자 제거
    print(" |-remove location special symbols [end]")

def __split_user_location__(users, delimeter=','):
    print(" |-split user location [start]")
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0].strip()) # location_city 정의: location의 첫번째 부분
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1].strip()) # location_state 정의: location의 두번째 부분
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2].strip()) # location_country 정의: location의 세번째 부분
    print(" |-split user location [end]")


def dl_data_load(args):
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

    ######################## DATA PREPROCESSING
    users = preprocess_user(args, users)
    books = preprocess_book(args, books)

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)


    field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data

def dl_data_split(args, data):
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
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def dl_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
