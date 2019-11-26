import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 

import gensim 
from gensim.models import Word2Vec 

import xml.etree.ElementTree as ET

def main():
   file = open('06_3.xml').read()
   root = ET.fromstring(file)
   citations =  root.find('citations')

   final_body = ''

   for child in citations:
      text = child.find('text').text
      final_body = final_body + '\n' + text

   tokens = word_tokenize(final_body)
   stop_words = set(stopwords.words('english'))
   filtered_words = [w for w in tokens if w not in stop_words]
   
   ''' CBOW Model '''
   model1 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5) 

if __name__ == "__main__":
  main()
