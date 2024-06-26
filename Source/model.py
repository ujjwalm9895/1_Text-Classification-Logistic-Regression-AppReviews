from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def vectorize(token_list, y, vect='bow', min_df=5, ng_low=1, ng_high=3,
              test_size=0.2, rs=42):
    """
    Vectorizes text data, splits it into train and test sets, and returns the vectorized data and a vectorizer.

    :param token_list: List of processed tokens (text data)
    :param y: Dependent variable (labels)
    :param vect: Type of vectorizer ('bow' for count vectors, 'bowb' for binary count vectors, 'ng' for n-grams, 'tf' for TF-IDF)
    :param min_df: min_df parameter in CountVectorizer or TfidfVectorizer (minimum document frequency)
    :param ng_low: Lower value for n-grams
    :param ng_high: Higher value for n-grams
    :param test_size: Size of the test split when splitting the data into train and test sets
    :param rs: Random seed for reproducibility

    :return: Train and test vectors (X_train, X_test), train and test labels (y_train, y_test), and the vectorizer used
    """
    # Create a vectorizer based on the chosen type
    if vect == 'bow':
        vectorizer = CountVectorizer(min_df=min_df)
    elif vect == 'bowb':
        vectorizer = CountVectorizer(binary=True, min_df=min_df)
    elif vect == 'ng':
        vectorizer = CountVectorizer(min_df=min_df, ngram_range=(ng_low, ng_high))
    elif vect == 'tf':
        vectorizer = TfidfVectorizer(min_df=min_df)
    else:
        raise Exception("vect has to be one of 'bow', 'bowb', 'ng', 'tf'")

    # Fit the vectorizer to the data and transform the input data into vectorized form
    X = vectorizer.fit_transform(token_list)

    # Split the data into train and test sets with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=rs)

    return X_train, X_test, y_train, y_test, vectorizer
