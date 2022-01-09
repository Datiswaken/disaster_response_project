import sys
import nltk
import pandas as pd
import re
import pickle

from numpy.typing import NDArray
from pandas import DataFrame
from typing import List, Tuple
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

nltk.download(['punkt', 'wordnet', 'stopwords', 'omw-1.4'])


def load_data(database_filepath: str) -> Tuple[List[str], DataFrame, List[str]]:
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * FROM social_media_messages', engine)
    x = df['message'].values
    y = df.iloc[:, 4:]

    return x, y, y.columns


def tokenize(text: str) -> List[str]:
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model() -> GridSearchCV:
    forest = RandomForestClassifier()

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words=stopwords.words('english'))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(forest))
    ])

    parameters = {
        'clf__estimator__n_estimators': [1]
    }

    return GridSearchCV(pipeline, param_grid=parameters)


def evaluate_model(model: GridSearchCV, X_test: NDArray, Y_test: DataFrame, category_names: List[str]) -> None:
    model.fit(X_test, Y_test)
    Y_pred = model.predict(X_test)

    print(classification_report(Y_test.values[:, ], Y_pred, target_names=category_names))


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    file = open(model_filepath, 'wb')
    pickle.dump(model.best_estimator_, file)
    file.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building models...')
        model = build_model()

        print('Training models...')
        model.fit(X_train, Y_train)

        print('Evaluating models...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving models...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained models saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the models to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
