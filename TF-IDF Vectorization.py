from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the cleaned text
X = vectorizer.fit_transform(data['cleaned_text'])

# Target variable
y = data['label']

print(X.shape)
