import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.collocations import *
from string import punctuation
from nltk.stem.lancaster import LancasterStemmer

text = "Mary had a little lamb. Her fleece was white as snow"
# Split into sentences
sents = sent_tokenize(text)
print(sents)
# Split into words
words = [word_tokenize(sent) for sent in sents]
print(words)


#Removing stopwords
# Custom set of stopwords
customStopWords = set(stopwords.words('english')+list(punctuation))
# Remove them from our sentence
# Split into words
wordsWithoutStop = [word for word in word_tokenize(text) if word not in customStopWords]
print(wordsWithoutStop)

# Find N-Grams
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsWithoutStop)
print(sorted(finder.ngram_fd.items()))

# New sentence to demo stemming
text2 = "Mary closed on closing night when she was in the mood to close."
# Lancaster stem
st = LancasterStemmer()
stemmedWords = [st.stem(word) for word in word_tokenize(text2)]
print(stemmedWords) # Word is reduced in stemmed formed
# POS tagging
print(nltk.pos_tag(word_tokenize(text2)))

# Word sense disambiguation
for ss in wn.synsets('bass'):
    print(ss, ss.definition())
