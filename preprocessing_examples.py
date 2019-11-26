import nltk

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('tagsets')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 

''' Tokenizing '''
sentence = "vasu is a good person and he lives in aligarh"
tokens = word_tokenize(sentence)
print('Tokens = '+str(tokens))

''' Removing stop words '''
stop_words = set(stopwords.words('english'))
sentence = "vasu is a good person and he lives in aligarh"
tokens = word_tokenize(sentence)
filtered_words = [w for w in tokens if w not in stop_words]
print('Filtered words = '+str(filtered_words))

''' Lemmatization and Stemming '''
lem  = WordNetLemmatizer()
stem = PorterStemmer()
word = "multiplying"
print('Lemmatizing multiplying = '+str(lem.lemmatize(word,"v"))) #v stands for verb
print('Stemming multiplying = '+str(stem.stem(word)))

''' Part-of-speech tagging '''
sentence = "Vasu is a musician and he lives in a city named as Aligarh, He drives a bike."
token = nltk.word_tokenize(sentence)
print('POS Tagging = '+str(nltk.pos_tag(token))) 

# ''' Chunking '''
# grammar = ('''
#   NP: {<DT>?<JJ>*<NN>} # NP
#   '''
# )
# chunkParser = nltk.RegexpParser(grammar)
# tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
# tagged
# tree = chunkParser.parse(tagged)
# for subtree in tree.subtrees():
#   print(subtree)