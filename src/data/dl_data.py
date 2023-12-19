import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

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
    preprocess_user(args, users)
    preprocess_book(args, books)

    ######################## MERGE DATA
    if args.merge_users:
        target_features = list(args.preprocess_user)

        if 'location' in target_features:
            target_features.remove('location')
            for feature in ('location_city', 'location_state', 'location_country'):
                target_features.append(feature)

        train = train.merge(users[target_features], on='user_id', how='left')
        test = test.merge(users[target_features], on='user_id', how='left')
    
    if args.merge_books:
        target_features = list(args.preprocess_book)

        for feature in ('isbn_group', 'isbn_publisher', 'isbn_serial'):
            target_features.append(feature)

        train = train.merge(books[target_features], on='isbn', how='left')
        test = test.merge(books[target_features], on='isbn', how='left')

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

    field_dims = __field_dims__(train)

    print("Field_dims:", field_dims)

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

def __field_dims__(dataFrame: pd.DataFrame) -> np.array:
    field_dims = [(max(dataFrame[feature]) + 1) for feature in dataFrame.columns if feature != 'rating']
    return np.array(field_dims, dtype=np.uint32)

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


def preprocess_user(args, users):
    print("|-user preprocessing [start]")

    feature_to_function = {'user_id': __preprocess_user_id__,
                           'age': __preprocess_age__,
                           'location': __preprocess_location__,}

    for feature in args.preprocess_user:
        if feature not in feature_to_function:
            raise ValueError(f"정의되지 않은 feature입니다: {feature}")
        preprocess = feature_to_function[feature]
        preprocess(users)
    
    print(">>>>>>>", users.columns)
    print("|-user preprocessing [end]")

def __preprocess_age__(users, feature_name: str= 'age'):
    
    bins = [0, 
            14, 15, 16, 17, 18, 19, 
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
            60, 61, 62, 63, 65, 67, 
            70, 75, 
            100]
    
    binning(users, feature_name, bins)
    fill_nan_with_mode(users, feature_name)
    
def __preprocess_user_id__(users, feature_name: str= 'user_id'):
    pass

def preprocess_book(args, books):
    print("|-book preprocessing [start]")

    feature_to_function = {'isbn': __preprocess_isbn__,
                           'book_title': __preprocess_title__,
                           'book_author': __preprocess_author__,
                           'year_of_publication': __preprocess_year_of_publication__,
                           'publisher': __preprocess_publisher__,
                           'img': __preprocess_img__,
                           'language': __preprocess_language__,
                           'category': __preprocess_category__,
                           'summary': __preprocess_summary__,
                           }
    
    for feature in args.preprocess_book:
        if feature not in feature_to_function:
            raise ValueError(f"정의되지 않은 feature입니다: {feature}")
        preprocess = feature_to_function[feature]
        preprocess(books)

    print("|-book preprocessing [end]")


label_encoder = LabelEncoder()
def labeling(dataFrame: pd.DataFrame, feature_name: str) -> None:
    dataFrame[feature_name] = label_encoder.fit_transform(dataFrame[feature_name])


def binning(dataFrame: pd.DataFrame, feature_name: str, bins: list, labels: list = [], right: bool= False) -> None:
    '''
        구간의 임계값인 bins가 각 구간의 label인 labels의 개수보다 항상 1개 더 많아야 합니다.
        right: 우측 임계값을 포함하는지 여부
            True: [0, 20]
            False: [0, 20)
        예시> binning(users, 'age', [0, 20, 30, 40, 50, 60, 100], [1,2,3,4,5,6])
    '''
    if len(labels) != len(bins) - 1:
        print(f"labels의 길이가 bins에 유효하지 않아, 기본 설정값으로 labels가 대체 됩니다: {len(labels)} != {len(bins) - 1}")
        labels = [i for i in range(len(bins) - 1)]
    dataFrame[feature_name] = pd.cut(dataFrame[feature_name], labels= labels, bins= bins, right= right)


def fill_nan_with_mode(dataFrame: pd.DataFrame, feature_name: str) -> None:
    mode_value = dataFrame[feature_name].mode().iloc[0]
    dataFrame[feature_name].fillna(mode_value, inplace=True)


def remove_special_symbols(dataFrame: pd.DataFrame, feature_name: str, regex: str= r'[^0-9a-zA-Z:,]'):
    dataFrame[feature_name] = dataFrame[feature_name].str.replace(regex, '', regex=True)


def lower(dataFrame: pd.DataFrame, feature_name: str):
    dataFrame[feature_name] = dataFrame[feature_name].str.lower()


def fill_nan_with_input(dataFrame: pd.DataFrame, feature_name: str, input: str) -> None:
    dataFrame[feature_name].fillna(input, inplace=True)

def drop_feature(dataFrame: pd.DataFrame, feature_name: str) -> None:
    dataFrame.drop(feature_name, inplace=True, axis=1)


def __preprocess_isbn__(books, feature_name= 'isbn'):
    print(" |-preprocess isbn [start]")

    __split_isbn__(books)
    labeling(books, 'isbn_group')
    labeling(books, 'isbn_publisher')
    labeling(books, 'isbn_serial')

    print(" |-preprocess isbn [end]")


def __preprocess_title__(books, feature_name= 'book_title'):
    print(" |-preprocess title [start]")

    lower(books, feature_name)
    fill_nan_with_input(books, feature_name, 'unknown')
    labeling(books, feature_name)

    print(" |-preprocess title [end]")


def __preprocess_author__(books, feature_name= 'book_author'):
    print(" |-preprocess author [start]")

    lower(books, feature_name)
    fill_nan_with_input(books, feature_name, 'unknown')
    labeling(books, feature_name)

    print(" |-preprocess author [end]")


def __preprocess_year_of_publication__(books, feature_name= 'year_of_publication'):
    print(" |-preprocess year of publication [start]")

    bins = [0,
            1900, 
            1940, 
            1950, 
            1960, 
            1970, 1975, 1976, 1977, 1978, 1979, 
            1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
            1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
            2000, 2001, 2002, 2003, 2004, 2005, 2006, 2010]
    
    binning(books, feature_name, bins)
    fill_nan_with_mode(books, feature_name)

    print(" |-preprocess year of publication [end]")


def __preprocess_publisher__(books, feature_name= 'publisher'):
    print(" |-preprocess publisher [start]")

    lower(books, feature_name)
    fill_nan_with_input(books, feature_name, 'unknown')
    labeling(books, feature_name)

    print(" |-preprocess publisher [end]")


def __preprocess_img__(books, feature_name= 'img'):
    print(" |-preprocess img [start]")
    print(" |-preprocess img [end]")


def __preprocess_language__(books, feature_name= 'language'):
    print(" |-preprocess language [start]")

    lower(books, feature_name)
    fill_nan_with_input(books, feature_name, 'unknown')
    labeling(books, feature_name)

    print(" |-preprocess language [end]")


def __preprocess_category__(books, feature_name= 'category'):
    print(" |-preprocess category [start]")

    lower(books, feature_name)
    __remove_category_special_symbols__(books)
    __remain_top_n_percent_category__(books)
    fill_nan_with_mode(books, feature_name)
    labeling(books, feature_name)

    print(" |-preprocess category [end]")


def __preprocess_summary__(books, feature_name= 'summary'):
    print(" |-preprocess summary [start]")
    print(" |-preprocess summary [end]")


def __split_isbn__(books):
    print("  |-split isbn [start]")

    books['isbn_group'] = books['isbn'].str[0:2]
    books['isbn_publisher'] = books['isbn'].str[2:8]
    books['isbn_serial'] = books['isbn'].str[8]

    print("  |-split isbn [end]")


def __remain_top_n_percent_category__(books, threshold: int= 5):
    print("  |-Remain top n percent categories [start]")
    value_counts = books[books['category'].notna()]['category'].value_counts()

    threshold_count = value_counts.quantile((100 - threshold) / 100)

    valid_values = value_counts[value_counts >= threshold_count].index

    valid_categories = set(books[books['category'].isin(valid_values)]['category'])

    for idx, category in enumerate(books['category']):
        if pd.isna(category):
            continue
        if category in valid_categories:
            continue

        found = False
        for valid_category in valid_categories:
            if valid_category in category:
                books.loc[idx, 'category'] = valid_category
                found = True
                break
        
        if not found:
            books.loc[idx, 'category'] = 'unknown'
    
    print("  |-Remain top n percent categories [start]")


def __preprocess_location__(users, feature_name= 'location'):
    
    __remove_location_special_symbols__(users)

    splited_location = __splited_user_location__(users)
    
    for feature in splited_location:
        labeling(users, feature)

    __fill_unknown_location_with_nan__(users)

    __fill_unknown_state_and_country_by_city__(users)

    drop_feature(users, feature_name)

    print(">>>", users.columns)


def __fill_unknown_location_with_nan__(users):
    print(" |-fill unknown value with nan [start]")
    users['location'] = users['location'].replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users['location'] = users['location'].replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.
    print(" |-fill unknown value with nan [end]")


def __remove_location_special_symbols__(users, regex= r'[^0-9a-zA-Z:,]'):
    print(" |-remove location special symbols [start]")
    remove_special_symbols(users, 'location', regex)
    print(" |-remove location special symbols [end]")


def __remove_category_special_symbols__(books, regex= r'[^0-9a-zA-Z:, ]'):
    print(" |-remove category special symbols [start]")
    remove_special_symbols(books, 'category', regex)
    print(" |-remove category special symbols [end]")


def __splited_user_location__(users, delimeter= ','):
    print(" |-split user location [start]")
    users['location_city'] = users['location'].apply(lambda x: x.split(delimeter)[0].strip()) # location_city 정의: location의 첫번째 부분
    users['location_state'] = users['location'].apply(lambda x: x.split(delimeter)[1].strip()) # location_state 정의: location의 두번째 부분
    users['location_country'] = users['location'].apply(lambda x: x.split(delimeter)[2].strip()) # location_country 정의: location의 세번째 부분
    print(" |-split user location [end]")

    return ('location_city', 'location_state', 'location_country')


def __fill_unknown_state_and_country_by_city__(users):
    print(" |-fill state and country from city [start]")

    city2state, city2country = __state_and_country_map_by_city__(users)

    nan_state_rows, nan_country_rows = __unknown_state_and_country_with_known_city__(users)
    
    __fill_nan_state_and_country_by_city__(users, city2state, nan_state_rows, city2country, nan_country_rows)

    print(" |-fill state and country from city [end]")


def __fill_nan_state_and_country_by_city__(users, city2state, nan_state_rows, city2country, nan_country_rows):
    users.loc[nan_state_rows, 'location_state'] = users.loc[nan_state_rows, 'location_city'].map(city2state)
    users.loc[nan_country_rows, 'location_country'] = users.loc[nan_country_rows, 'location_city'].map(city2country)


def __unknown_state_and_country_with_known_city__(users):
    nan_state_rows = users['location_state'].isna() & users['location_city'].notna()
    nan_country_rows = users['location_country'].isna() & users['location_city'].notna()

    return nan_state_rows, nan_country_rows


def __state_and_country_map_by_city__(users):
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
    
    return city2state, city2country
