from sklearn.preprocessing import LabelEncoder
import pandas as pd

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


def fill_nan_with_max(dataFrame: pd.DataFrame, feature_name: str) -> None:
    value = dataFrame[feature_name].max()
    dataFrame[feature_name].fillna(value, inplace=True)


def fill_nan_with_min(dataFrame: pd.DataFrame, feature_name: str) -> None:
    value = dataFrame[feature_name].min()
    dataFrame[feature_name].fillna(value, inplace=True)


def fill_nan_with_median(dataFrame: pd.DataFrame, feature_name: str) -> None:
    value = dataFrame[feature_name].median()
    dataFrame[feature_name].fillna(value, inplace=True)


def remove_special_symbols(dataFrame: pd.DataFrame, feature_name: str, regex: str= r'[^0-9a-zA-Z:,]'):
    dataFrame[feature_name] = dataFrame[feature_name].str.replace(regex, '', regex=True)


def lower(dataFrame: pd.DataFrame, feature_name: str):
    dataFrame[feature_name] = dataFrame[feature_name].str.lower()


def fill_nan_with_input(dataFrame: pd.DataFrame, feature_name: str, input: str) -> None:
    dataFrame[feature_name].fillna(input, inplace=True)


def drop_feature(dataFrame: pd.DataFrame, feature_name: str) -> None:
    dataFrame.drop(feature_name, inplace=True, axis=1)