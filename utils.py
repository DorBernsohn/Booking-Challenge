import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder

from config import LabelEncoderMapping

def load_data(file_path: str, min_trip_length_threshold: int = None) -> pd.DataFrame:
    """load_data load the .csv file and filter by trip length 

    the filter is because there are trips with length of 48 stops which makes the dataframe wide

    Args:
        file_path (str): the path of the .csv file
        min_trip_length_threshold (int, optional): the max number of stops each trip can have. Defaults to None.

    Returns:
        pd.DataFrame: dataframe after loading the .csv file
    """    
    data = pd.read_csv(file_path, 
                       dtype={'user_id': 'float32',
                              'checkin': 'str',
                              'checkout': 'str',
                              'city_id': 'float32',
                              'affiliate_id': 'float32',
                              'booker_country': 'str',
                              'hotel_country': 'str',
                              'utrip_id': 'str'},
                       parse_dates=['checkin', 'checkout']).sort_values(by=['user_id','checkin'])
    
    if min_trip_length_threshold:
        data = data[data.groupby('utrip_id')['user_id'].transform('count') >= min_trip_length_threshold]

    data.fillna({'user_id': 0,
           'checkin': 0,
           'checkout': 0,
           'city_id': -1,
           'device_class': 'Unknown',
           'affiliate_id': 0,
           'booker_country': 'Unknown',
           'hotel_country': 'Unknown',
           'utrip_id': 0}, inplace=True)

    return data

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """extract_features extending the dataframe with additional features such as: time features, trip info features

    Args:
        df (pd.DataFrame): the dataframe before addnig features

    Returns:
        pd.DataFrame: the dataframe after adding features
    """  
    #encode  
    columns_to_encode = ['city_id', 'device_class', 'booker_country', 'hotel_country']
    encode_columns(df, columns_to_encode)
    #time
    build_time_features(df=df, col='checkin')
    build_time_features(df=df, col='checkout')
    #length
    df['length'] = np.log((df.checkout - df.checkin).dt.days) # trip length: (in log(days))
    df['trip_length'] = df.groupby('utrip_id')['utrip_id'].transform('size') # length: number of stops
    #season
    df_season = pd.DataFrame({'checkin_month': range(1,13), 'season': ([0]*3)+([1]*3)+([2]*3)+([3]*3)}) # the season of the year
    df = df.merge(df_season, how='left', on='checkin_month')
    #count
    utrip_id_trip_length_mapping = [list(range(0, i)) for i in [x for x in df[['utrip_id', 'trip_length']].drop_duplicates(subset ='utrip_id')['trip_length']]]
    df['count'] = [item for sublist in utrip_id_trip_length_mapping for item in sublist]
    #city features
    build_prev_city(df=df, num_prev=3)
    build_first_city(df=df)
    return df

def build_time_features(df: pd.DataFrame, col: str):
    """build_time_features build time features

    checkin/checkout: year/month/woy (week of year)/dow (day of week)/weekend

    Args:
        df (pd.DataFrame): the dataframe 
        col (str): the col for applying the features
    """    
    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_woy'] = df[col].dt.weekofyear
    df[f'{col}_dow'] = df[col].dt.dayofweek
    df[f'{col}_weekend'] = df[f'{col}_dow'].isin([5,6]).astype('int8')

def build_prev_city(df: pd.DataFrame, num_prev: int):
    """build_prev_city build previous city features

    previos city: the previous city (1, 2, .., n)

    Args:
        df (pd.DataFrame): the dataframe
        num_prev (int): the number of previous stops to take
    """    
    utrip_id_city_id_mapping = df.groupby(['utrip_id', 'count'])['city_id'].apply(list).groupby(level=0).apply(list) # a series that maps utrip_id to city_id
    records_label, records_count, base_list = [], [], []
    for trip in set(df.utrip_id.values):
        base_list.append([0] * num_prev) # initialize first entry
        records_label.append(trip) # initialize first entry
        records_count.append(0) # initialize first entry
        
        data = [0] * (num_prev-1) + [x[0] for x in utrip_id_city_id_mapping[trip]] # create the data shifted by 2
        for i in range(0, len(data) - (num_prev - 1)):
            base_list.append(list(reversed(data[i:i+num_prev])))
            records_label.append(trip)
            records_count.append(i + 1)
    prev_city_df = pd.DataFrame.from_records(base_list, columns = [f'prev_city_{i}' for i in range(1, num_prev + 1)])
    prev_city_df['utrip_id'] = records_label
    prev_city_df['count'] = records_count
    df = df.merge(prev_city_df, on = ['utrip_id', 'count'], left_index=True)

def build_first_city(df: pd.DataFrame):
    """build_first_city build first city of trip feature

    first city: the first city of the trip

    Args:
        df (pd.DataFrame): the dataframe 
    """    
    # city_list = []
    # for trip in set(df.utrip_id.values):
    #     city_list.append([utrip_id_city_id_mapping[trip][0][0]] * len(utrip_id_city_id_mapping[trip]))
    # [item for sublist in city_list for item in sublist]
    df['first_city'] = df.groupby(['utrip_id'])['city_id'].transform('first')

def encode_columns(df: pd.DataFrame, cols: List[str]):
    """encode_columns encode columns into integers

    Args:
        df (pd.DataFrame): the dataframe
        cols (List[str]): the col names to encode
    """    
    for col in cols:
        le = LabelEncoder()
        le.fit(df[col])
        LabelEncoderMapping[col] = le
        if col == 'city_id': 
            df[col] = le.transform(df[col])
            continue
        df[f"{col}_encode"] = le.transform(df[col])

def flatten_features(df: pd.DataFrame) -> pd.DataFrame:
    """flatten_features flatten the data from few rows pre trip to one row

    Args:
        df (pd.DataFrame): the dataframe

    Returns:
        pd.DataFrame: the flattened dataframe
    """    
    df_out = df.set_index(['utrip_id','user_id',df.groupby(['utrip_id','user_id']).cumcount()+1]).unstack().sort_index(level=1, axis=1)
    df_out.columns = df_out.columns.map('{0[0]}_{0[1]}'.format)
    df_out = df_out.reset_index()
    return df_out

def split_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df.loc[:, 'is_label'] = df.groupby('utrip_id')['checkin'].transform('max')
    df.loc[:, 'is_label'] = df['checkin'] == df['is_label']

    label_mask = df['is_label']
    features = df[~label_mask].drop(columns=['is_label', 
                                             'device_class', 
                                             'booker_country', 
                                             'hotel_country', 
                                             'checkin', 
                                             'checkout']).sort_values(by=['utrip_id'])
    labels = df[label_mask][['utrip_id', 'city_id']].sort_values(by=['utrip_id'])

    return features, labels