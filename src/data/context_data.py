import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 0
    elif x >= 20 and x < 30:
        return 1
    elif x >= 30 and x < 40:
        return 2
    elif x >= 40 and x < 50:
        return 3
    elif x >= 50 and x < 60:
        return 4
    else:
        return 5

def process_context_data(users, books, ratings1, ratings2):
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

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[[
        'isbn', 'category', 'publisher', 'language', 'book_author',
        'group', 'publisher2', 'title',
        ]], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[[
        'isbn', 'category', 'publisher', 'language', 'book_author',
        'group', 'publisher2', 'title',
        ]], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[[
        'isbn', 'category', 'publisher', 'language', 'book_author',
        'group', 'publisher2', 'title',
        ]], on='isbn', how='left')

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    publisher22idx = {v:k for k,v in enumerate(context_df['publisher2'].unique())}
    group2idx = {v:k for k,v in enumerate(context_df['group'].unique())}
    title2idx = {v:k for k,v in enumerate(context_df['title'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    train_df['publisher2'] = train_df['publisher2'].map(publisher22idx)
    train_df['title'] = train_df['title'].map(title2idx)
    train_df['group'] = train_df['group'].map(group2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)
    test_df['publisher2'] = test_df['publisher2'].map(publisher22idx)
    test_df['title'] = test_df['title'].map(title2idx)
    test_df['group'] = test_df['group'].map(group2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
        "publisher22idx":publisher22idx,
        "group2idx":group2idx,
        "title2idx":title2idx,
    }

    return idx, train_df, test_df


def context_data_load(args):
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

    books['group'] = books['isbn'].apply(lambda x: x[:2]) # 2자리 최대 99
    books['publisher2'] = books['isbn'].apply(lambda x: x[2:6]) # 4자리 최대 9999
    books['title'] = books['isbn'].apply(lambda x: x[6:8]) # 3자리 최대 999

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

    idx, context_train, context_test = process_context_data(users, books, train, test)

    field_dims = np.array([
        len(user2idx), len(isbn2idx), 6,
        len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
        len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx']),
        len(idx['group2idx']), len(idx['publisher22idx']), len(idx['title2idx'])
        ], dtype=np.uint32)

    ## summary load
    if args.merge_summary:

        train_summary = np.load('data/text_vector/train_item_summary_vector.npy', allow_pickle=True)
        test_summary = np.load('data/text_vector/test_item_summary_vector.npy', allow_pickle=True)

        train_books_text_df = pd.DataFrame([train_summary[0], train_summary[1]]).T
        test_books_text_df = pd.DataFrame([test_summary[0], test_summary[1]]).T

        train_books_text_df.columns = ['isbn', 'item_summary_vector']
        test_books_text_df.columns = ['isbn', 'item_summary_vector']

        train_books_text_df['isbn'] = train_books_text_df['isbn'].astype('int')
        test_books_text_df['isbn'] = test_books_text_df['isbn'].astype('int')

        context_train = pd.merge(context_train, train_books_text_df[['isbn', 'item_summary_vector']], on='isbn', how='left')
        context_test = pd.merge(context_test, test_books_text_df[['isbn', 'item_summary_vector']], on='isbn', how='left')

        field_dims = np.array((*field_dims, 768), dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
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


def context_data_split(args, data):
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

class Text_Dataset(Dataset):
    def __init__(self, context_vector, item_summary_vector, train=True, label=None):
        """
        Parameters
        ----------
        context_vector : np.ndarray
            벡터화된 유저와 책 데이터를 입렵합니다.
        item_summary_vector : np.ndarray
            벡터화된 책에 대한 요약 정보 데이터 입력합니다.
        label : np.ndarray
            정답 데이터를 입력합니다.
        ----------
        """
        self.context_vector = context_vector
        self.item_summary_vector = item_summary_vector
        self.train = train
        if self.train:
            self.label = label

    def __len__(self):
        return self.context_vector.shape[0]

    def __getitem__(self, i):

        if self.train:
            return {
                'context_vector' : torch.tensor(self.context_vector[i], dtype=torch.long),
                'item_summary_vector' : torch.tensor(self.item_summary_vector[i].reshape(-1, 1), dtype=torch.float32),
                'label' : torch.tensor(self.label[i], dtype=torch.float32),
            }
        else:
            return {
                'context_vector' : torch.tensor(self.context_vector[i], dtype=torch.long),
                'item_summary_vector' : torch.tensor(self.item_summary_vector[i].reshape(-1, 1), dtype=torch.float32),
            }


def context_data_loader(args, data):
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

    if args.merge_summary:

        train_dataset = Text_Dataset(
            data['X_train'].drop(['item_summary_vector'], axis=1).values,
            data['X_train']['item_summary_vector'].values,
            True, data['y_train'].values
        )

        valid_dataset = Text_Dataset(
            data['X_valid'].drop(['item_summary_vector'], axis=1).values,
            data['X_valid']['item_summary_vector'].values,
            True, data['y_valid'].values
        )
        test_dataset = Text_Dataset(
            data['test'].drop(['item_summary_vector'], axis=1).values,
            data['test']['item_summary_vector'].values,
            False
        )
    else:
        train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
        valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
        test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] =\
            train_dataloader, valid_dataloader, test_dataloader

    return data
