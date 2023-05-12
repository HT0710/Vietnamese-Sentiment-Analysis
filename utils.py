from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re, yaml

stopwords_list = stopwords.words('english')

def setup():
    # Prepare
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

# Remove html
def remove_html(text):
    pattern = r'<.*?>'
    out = re.sub(pattern, ' ', text)
    return out

# Remove special characters
def remove_specials(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    out = re.sub(pattern, ' ', text)
    return out

# Stemming
def stemming(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = [stemmer.stem(word) for word in tokens]
    return ' '.join(stems)

# lemmatization
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(lemmas)

# Remove stopwords
def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    removed = [word for word in tokens if word not in stopwords_list]
    return ' '.join(removed)

# Bag of words
def bow(dataset, train=True, ngrams=(1, 1)):
    vectorizer = CountVectorizer(min_df=0, max_df=1, ngram_range=ngrams)
    if train:
        bow = vectorizer.fit_transform(dataset)
    else:
        bow = vectorizer.transform(dataset)
    return bow.toarray()

# IF-IDF
def ifidf(dataset, train=True, ngrams=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=ngrams)
    if train:
        bow = vectorizer.fit_transform(dataset)
    else:
        bow = vectorizer.transform(dataset)
    return bow.toarray()

# Save result
def save_result(prepocess: list, vectorizer: str, model: str, accuracy: float):
    with open('result.yaml', 'r') as file:
        result = yaml.safe_load(file)
        last_result = list(result.keys())[-1]
        num = int(last_result.split('_')[-1])
        result[f'run_{num+1}'] = {
            "prepocess": prepocess,
            "vectorizer": vectorizer,
            "model": model,
            "accuracy": accuracy
        }
        
    with open('result.yaml', 'w') as file:
        yaml.dump(result, file)
