import sys
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> DataFrame:
    """
    Load the data from csv files

    :param messages_filepath:  Filepath to a csv file containing the social media messages
    :param categories_filepath: Filepath to a csv file containing the categories with label
    :return: Returns a merged Pandas dataframe containing data of both input files
    """
    try:
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
    except Exception as err:
        print(f'Failed to load data.\nError: {str(err)}')
        sys.exit()

    return messages.merge(categories, how='inner', on='id')


def clean_data(df: DataFrame) -> DataFrame:
    """
    Clean a dataframe by applying several cleaning steps

    :param df: The dataframe to be cleaned
    :return: Returns the cleaned dataframe
    """
    categories = split_categories(df)
    categories = convert_cat_values_to_numbers(categories)
    df = replace_categories_column(df, categories)

    df = drop_duplicates(df)

    df['related'].replace(2, 1, inplace=True)

    return df


def split_categories(df: DataFrame) -> DataFrame:
    """
    Split the categories of the dataframe into individual columns

    :param df: The dataframe to be processed
    :return: Returns the dataframe with categories split
    """
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    categories.columns = row.apply(lambda col: col[0:-2])

    return categories


def convert_cat_values_to_numbers(categories: DataFrame) -> DataFrame:
    """
    Remove the category label from each cell and only keep the binary value

    :param categories: The dataframe containing category data
    :return: Returns the dataframe with binary values for each value
    """
    for column in categories:
        categories[column] = categories[column].apply(lambda row: row[-1:])
        categories[column] = pd.to_numeric(categories[column])

    return categories


def replace_categories_column(df: DataFrame, categories: DataFrame) -> DataFrame:
    """
    Replace the original categories column with a dataframe of individual categories

    :param df: The dataframe containing the original categories column
    :param categories: The cleaned dataframe containing category labels
    :return: Returns the combined dataframe of original dataframe and the one containing cleaned category labels
    """
    df.drop(['categories'], axis=1, inplace=True)

    return pd.concat([df, categories], sort=False, axis=1)


def drop_duplicates(df: DataFrame) -> DataFrame:
    """
    Drop duplicate rows

    :param df: The dataframe for which duplicates should be dropped
    :return: Returns the dataframe without duplicates
    """
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)

    return df


def save_data(df: DataFrame, database_filename: str) -> None:
    """
    Save the dataframe into a sqlite database

    :param df:The dataframe to be saved
    :param database_filename: The filename for the database
    :return: None
    """
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
