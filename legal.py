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
   
   # print(filtered_words)

   ''' CBOW Model '''
   model = gensim.models.Word2Vec([filtered_words], min_count = 1, size = 100, window = 5) 
   model.train([filtered_words],total_examples=1,epochs=1)
   word_vector = model.wv
   word_vector.save('vector.kv')
   print('Word vector for justice is : \n'+str(model['justice']))

if __name__ == "__main__":
  main()
