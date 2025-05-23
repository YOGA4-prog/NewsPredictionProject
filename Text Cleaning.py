import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords & digits and lemmatize
    cleaned_words = [lemmatizer.lemmatize(word) for word in words 
                     if word not in stop_words and word.isalpha()]
    
    return ' '.join(cleaned_words)

# Apply to dataset
data['cleaned_text'] = data['text'].apply(clean_text)
