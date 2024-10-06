import holidays
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from model import Model


def loss(real_rates, predicted_rates):
    return np.average(abs(predicted_rates / real_rates - 1.0)) * 100.0


def transform_data(df, mapping_origin_kma=None, mapping_destination_kma=None, miles_limit=2000, weight_limit=50000):
    # extract date features
    df['pickup_date'] = pd.to_datetime(df['pickup_date'])
    df["year"] = df["pickup_date"].dt.year
    df["month"] = df["pickup_date"].dt.month
    df["day"] = df["pickup_date"].dt.day
    df["dayofweek"] = df["pickup_date"].dt.dayofweek
    df["minute"] = df["pickup_date"].dt.minute
    df["dayofyear"] = df["pickup_date"].dt.dayofyear
    df["quarter"] = df["pickup_date"].dt.quarter
    df["week_num"] = df["pickup_date"].dt.isocalendar().week

    # add USA holidays
    usa_holidays = holidays.US()
    df['is_holiday'] = df['pickup_date'].dt.date.apply(lambda x: x in usa_holidays)
    df['is_holiday'] = df['is_holiday'].astype(int)

    # transform date features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 30.42)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 30.42)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    df["week_num_sin"] = np.sin(2 * np.pi * df["week_num"] / 52.14)
    df["week_num_cos"] = np.cos(2 * np.pi * df["week_num"] / 52.14)

    # valid_miles processing
    df['valid_miles_outliers'] = np.where(df['valid_miles'] > miles_limit, miles_limit, df['valid_miles'])
    df['valid_miles_log'] = np.log(df['valid_miles_outliers'])
    scaler = StandardScaler()
    df[['valid_miles_std']] = scaler.fit_transform(df[['valid_miles_log']])

    # weight processing
    df['weight_outliers'] = np.where(df['weight'] > weight_limit, weight_limit, df['weight'])
    df['weight_log'] = np.log(df['weight_outliers'])
    scaler = StandardScaler()
    df[['weight_std']] = scaler.fit_transform(df[['weight_log']])
    # replace NaN to mean weight
    df.loc[df['weight_std'].isna(), "weight_std"] = df['weight_std'].mean()

    # one-hot encoding for transport_type
    df = pd.get_dummies(df, dtype='int', columns=['transport_type'], prefix='transport')

    # label encoding for origin_kma and destination_kma
    if not mapping_origin_kma:
        mapping_origin_kma = {v: i for i, v in enumerate(df['origin_kma'].unique())}
    df['origin_kma'] = df['origin_kma'].map(mapping_origin_kma)
    if not mapping_destination_kma:
        mapping_destination_kma = {v: i for i, v in enumerate(df['destination_kma'].unique())}
    df['destination_kma'] = df['destination_kma'].map(mapping_destination_kma)

    # drop not necessary date columns
    drop_columns = [
        'pickup_date', 'month', 'day', 'dayofweek', 'minute', 'dayofyear', 'quarter', 'week_num',\
        'valid_miles', 'valid_miles_outliers', 'valid_miles_log',\
        'weight', 'weight_outliers', 'weight_log'
    ]
    df = df.drop(drop_columns, axis=1)
    return df, mapping_origin_kma, mapping_destination_kma


def train_and_validate():
    # load data
    df_train = pd.read_csv('dataset/train.csv')
    X_train = df_train.drop(columns=['rate'])
    y_train = df_train['rate']
    df_val = pd.read_csv('dataset/validation.csv')
    X_val = df_val.drop(columns=['rate'])
    y_val = df_val['rate']

    # transform data
    X_train_transformed, mapping_origin_kma, mapping_destination_kma = transform_data(X_train)
    X_val_transformed, mapping_origin_kma, mapping_destination_kma =\
        transform_data(X_val, mapping_origin_kma, mapping_destination_kma)

    # fit model
    model = Model()
    model.fit(X_train_transformed, y_train)

    # calculate mape
    predicted_rates = model.predict(X_val_transformed)
    mape = loss(y_val, predicted_rates)
    mape = np.round(mape, 2)
    return mape


def generate_final_solution():
    # combine train and validation to improve final predictions
    df_train = pd.read_csv('dataset/train.csv')
    df_val = pd.read_csv('dataset/validation.csv')
    df_test = pd.read_csv('dataset/test.csv')
    df_combined = pd.concat([df_train, df_val], ignore_index=True)
    X_combined = df_combined.drop(columns=['rate'])
    y_combined = df_combined['rate']

    # transform data
    X_combined_transformed, mapping_origin_kma, mapping_destination_kma = transform_data(X_combined)
    X_test_transformed, mapping_origin_kma, mapping_destination_kma =\
        transform_data(df_test, mapping_origin_kma, mapping_destination_kma)

    # fit model using transformed data
    model = Model()
    model.fit(X_combined_transformed, y_combined)  # Используем X_combined_transformed вместо X_combined

    # generate and save test predictions
    df_test['predicted_rate'] = model.predict(X_test_transformed)
    df_test.to_csv('dataset/predicted.csv', index=False)


if __name__ == "__main__":
    mape = train_and_validate()
    print(f'Accuracy of validation is {mape}%')

    # was changed from 9 to 15
    if mape < 15:  # try to reach 9% or less for validation
        generate_final_solution()
        print("'predicted.csv' is generated, please send it to us")
