from sklearn.preprocessing import LabelEncoder
import pandas as pd

label_encoder = LabelEncoder()
def labeling(data_frame: pd.DataFrame, feature_name: str) -> None:
    data_frame[feature_name] = label_encoder.fit_transform(data_frame[feature_name])


def binning(data_frame: pd.DataFrame, feature_name: str, bins: list, labels: list = [], right: bool= False) -> None:
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
    data_frame[feature_name] = pd.cut(data_frame[feature_name], labels= labels, bins= bins, right= right)


def fill_nan_with_mode(data_frame: pd.DataFrame, feature_name: str) -> None:
    mode_value = data_frame[feature_name].mode().iloc[0]
    data_frame[feature_name].fillna(mode_value, inplace=True)


def fill_nan_with_max(data_frame: pd.DataFrame, feature_name: str) -> None:
    value = data_frame[feature_name].max()
    data_frame[feature_name].fillna(value, inplace=True)


def fill_nan_with_min(data_frame: pd.DataFrame, feature_name: str) -> None:
    value = data_frame[feature_name].min()
    data_frame[feature_name].fillna(value, inplace=True)


def fill_nan_with_median(data_frame: pd.DataFrame, feature_name: str) -> None:
    value = data_frame[feature_name].median()
    data_frame[feature_name].fillna(value, inplace=True)


def remove_special_symbols(data_frame: pd.DataFrame, feature_name: str, regex: str= r'[^0-9a-zA-Z:,]'):
    data_frame[feature_name] = data_frame[feature_name].str.replace(regex, '', regex=True)


def lower(data_frame: pd.DataFrame, feature_name: str):
    data_frame[feature_name] = data_frame[feature_name].str.lower()


def fill_nan_with_input(data_frame: pd.DataFrame, feature_name: str, input: str) -> None:
    data_frame[feature_name].fillna(input, inplace=True)


def drop_feature(data_frame: pd.DataFrame, feature_name: str) -> None:
    data_frame.drop(feature_name, inplace=True, axis=1)

def preprocess_dataframe(data_frame: pd.DataFrame, **config: dict) -> None:
    for feature_name, preprocess in config.items():
        preprocess(data_frame, feature_name)