from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer

# Create a list of English stopwords
sw = stopwords.words('english')
# Create a RegexpTokenizer to remove punctuations
tokenizer = RegexpTokenizer(r'\w+')

def process_text(review, stem='p'):
    """
    Preprocesses a given text by converting it to lowercase, removing stopwords, punctuation, and performing stemming.

    :param review: Raw text input to be processed
    :param stem: Stemmer type ('p' for PorterStemmer, 'l' for LancasterStemmer)
    :return: Processed text after cleaning and stemming
    """
    # Convert text to lowercase
    review = review.lower()
    # Tokenize the review into words
    tokens = word_tokenize(review)
    # Remove stopwords
    tokens = [t for t in tokens if t not in sw]
    # Remove punctuation
    tokens = [tokenizer.tokenize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 0]
    tokens = ["".join(t) for t in tokens]
    # Create a stemmer based on the specified stem type
    if stem == 'p':
        stemmer = PorterStemmer()
    elif stem == 'l':
        stemmer = LancasterStemmer()
    else:
        raise Exception("stem has to be either 'p' for Porter or 'l' for Lancaster")
    # Perform stemming on the tokens
    tokens = [stemmer.stem(t) for t in tokens]
    # Return the processed text as a cleaned string
    return " ".join(tokens)
