import sys
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> DataFrame:
    try:
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
    except Exception as err:
        print(f'Failed to load data.\nError: {str(err)}')
        sys.exit()

    return messages.merge(categories, how='inner', on='id')


def clean_data(df: DataFrame) -> DataFrame:
    categories = split_categories(df)
    categories = convert_cat_values_to_numbers(categories)
    df = replace_categories_column(df, categories)

    df = drop_duplicates(df)

    return df


def split_categories(df: DataFrame) -> DataFrame:
    categories = df['categories'].str.split(pat=';', expand=True)

    row = categories.iloc[0]
    category_colnames = row.apply(lambda col: col[0:-2])
    categories.columns = category_colnames

    return categories


def convert_cat_values_to_numbers(categories: DataFrame) -> DataFrame:
    for column in categories:
        categories[column] = categories[column].apply(lambda row: row[-1:])
        categories[column] = pd.to_numeric(categories[column])

    return categories


def replace_categories_column(df: DataFrame, categories: DataFrame) -> DataFrame:
    df.drop(['categories'], axis=1, inplace=True)

    return pd.concat([df, categories], sort=False, axis=1)


def drop_duplicates(df: DataFrame) -> DataFrame:
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)

    df['related'].replace(2, 1, inplace=True)

    return df


def save_data(df: DataFrame, database_filename: str) -> None:
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('social_media_messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
