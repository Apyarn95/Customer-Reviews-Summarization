import pandas as pd

import requests
# data = pd.read_csv('drive/My Drive/steve.txt',delimiter="\t")
#text = ["Albert Einstein was born at Ulm, in Württemberg, Germany, on March 14, 1879. Six weeks later the family moved to Munich, where he later on began his schooling at the Luitpold Gymnasium. Later, they moved to Italy and Albert continued his education at Aarau, Switzerland and in 1896 he entered the Swiss Federal Polytechnic School in Zurich to be trained as a teacher in physics and mathematics. In 1901, the year he gained his diploma, he acquired Swiss citizenship and, as he was unable to find a teaching post, he accepted a position as technical assistant in the Swiss Patent Office. In 1905 he obtained his doctor’s degree . During his stay at the Patent Office, and in his spare time, he produced much of his remarkable work and in 1908 he was appointed Privatdozent in Berne. In 1909 he became Professor Extraordinary at Zurich, in 1911 Professor of Theoretical Physics at Prague, returning to Zurich in the following year to fill a similar post. In 1914 he was appointed Director of the Kaiser Wilhelm Physical Institute and Professor in the University of Berlin. He became a German citizen in 1914 and remained in Berlin until 1933 when he renounced his citizenship for political reasons and emigrated to America to take the position of Professor of Theoretical Physics at Princeton*. He became a United States citizen in 1940 and retired from his post in 1945 . After World War II, Einstein was a leading figure in the World Government Movement, he was offered the Presidency of the State of Israel, which he declined, and he collaborated with Dr. Chaim Weizmann in establishing the Hebrew University of Jerusalem . Einstein always appeared to have a clear view of the problems of physics and the determination to solve them. He had a strategy of his own and was able to visualize the main stages on the way to his goal. He regarded his major achievements as mere stepping-stones for the next advance . At the start of his scientific work, Einstein realized the inadequacies of Newtonian mechanics and his special theory of relativity stemmed from an attempt to reconcile the laws of mechanics with the laws of the electromagnetic field. He dealt with classical problems of statistical mechanics and problems in which they were merged with quantum theory: this led to an explanation of the Brownian movement of molecules. He investigated the thermal properties of light with a low radiation density and his observations laid the foundation of the photon theory of light . In his early days in Berlin, Einstein postulated that the correct interpretation of the special theory of relativity must also furnish a theory of gravitation and in 1916 he published his paper on the general theory of relativity. During this time he also contributed to the problems of the theory of radiation and statistical mechanics . In the 1920s, Einstein embarked on the construction of unified field theories, although he continued to work on the probabilistic interpretation of quantum theory, and he persevered with this work in America. He contributed to statistical mechanics by his development of the quantum theory of a monatomic gas and he has also accomplished valuable work in connection with atomic transition probabilities and relativistic cosmology . After his retirement he continued to work towards the unification of the basic concepts of physics, taking the opposite approach, geometrisation, to the majority of physicists . Einstein’s researches are, of course, well chronicled and his more important works include Special Theory of Relativity (1905), Relativity (English translations, 1920 and 1950), General Theory of Relativity (1916), Investigations on Theory of Brownian Movement (1926), and The Evolution of Physics (1938). Among his non-scientific works, About Zionism (1930), Why War? (1933), My Philosophy (1934), and Out of My Later Years (1950) are perhaps the most important . Albert Einstein received honorary doctorate degrees in science, medicine and philosophy from many European and American universities. During the 1920’s he lectured in Europe, America and the Far East, and he was awarded Fellowships or Memberships of all the leading scientific academies throughout the world. He gained numerous awards in recognition of his work, including the Copley Medal of the Royal Society of London in 1925, and the Franklin Medal of the Franklin Institute in 1935 . Einstein’s gifts inevitably resulted in his dwelling much in intellectual solitude and, for relaxation, music played an important part in his life. He married Mileva Maric in 1903 and they had a daughter and two sons; their marriage was dissolved in 1919 and in the same year he married his cousin, Elsa Löwenthal, who died in 1936. He died on April 18, 1955 at Princeton, New Jersey. Underachieving school kids have long taken solace in the claim that Einstein flunked math as a youth, but the records show that he was actually an exceptional, if not reluctant, student. He scored high grades during his school days in Munich, and was only frustrated by what he described as the “mechanical discipline” demanded by his teachers. The future Nobel Laureate dropped out of school at age 15 and left Germany to avoid state-mandated military service, but before then he was consistently at the top of his class and was even considered something of a prodigy for his grasp of complex mathematical and scientific concepts. When later presented with a news article claiming he’d failed grade-school math, Einstein dismissed the story as a myth and said, “Before I was 15 I had mastered differential and integral calculus.” . In 1896, Einstein renounced his German citizenship and enrolled at the Swiss Federal Polytechnic School in Zurich. There, he began a passionate love affair with Mileva Maric, a fellow physicist-in-training originally from Serbia. The couple later married and had two sons after graduating, but a year before they tied the knot, Maric gave birth to an illegitimate daughter named Lieserl. Einstein never spoke about the child to his family, and biographers weren’t even aware of her existence until examining his private papers in the late-1980s. Her fate remains a mystery to this day. Some scholars think Lieserl died from scarlet fever in 1903, while others believe she survived the sickness and was given up for adoption in Maric’s native Serbia. Einstein showed flashes of brilliance during his years at the Zurich Polytechnic, but his rebellious personality and penchant for skipping classes saw his professors give him less than glowing recommendations upon his graduation in 1900. The young physicist later spent two years searching for an academic position before settling for a gig at the Swiss patent office in Bern. Though menial, the job turned out to be a perfect fit for Einstein, who found he could breeze through his office duties in a few hours and spend the rest of the day writing and conducting research. In 1905—often called his “miracle year”—the lowly clerk published four revolutionary articles that introduced his famous equation E=mc2 and the theory of special relativity. While the discoveries marked Einstein’s entrance onto the physics world stage, he didn’t win a full professorship until 1909—nearly a decade after he had left school. After his marriage to Mileva Maric hit the rocks in the early 1910s, Einstein left his family, moved to Berlin and started a new relationship with his cousin, Elsa. He and Maric finally divorced several years later in 1919. As part of their separation agreement, Einstein promised her an annual stipend plus whatever money he might receive from the Nobel Prize—which he was supremely confident he would eventually win. Maric agreed, and Einstein later handed over a small fortune upon receiving the award in 1922 for his work on the photoelectric effect. By then, he had already remarried to Elsa, who remained his wife until her death in 1936. n 1915, Einstein published his theory of general relativity, which stated that gravitational fields cause distortions in the fabric of space and time. Because it was such a bold rewriting of the laws of physics, the theory remained controversial until May 1919, when a total solar eclipse provided the proper conditions to test its claim that a supermassive object—in this case the sun—would cause a measurable curve in the starlight passing by it. Hoping to prove Einstein’s theory once and for all, English astronomer Arthur Eddington journeyed to the coast of West Africa and photographed the eclipse. Upon analyzing the pictures, he confirmed that the sun’s gravity had deflected the light by roughly 1.7 arc-seconds—exactly as predicted by general relativity. The news made Einstein an overnight celebrity. Newspapers hailed him as the heir to Sir Isaac Newton, and he went on to travel the world lecturing on his theories about the cosmos. According to Einstein biographer Walter Isaacson, in the six years after the 1919 eclipse, more than 600 books and articles were written about the theory of relativity. "]
with open('drive/My Drive/steve.txt', 'r') as file:
    data = file.read().replace('\n', '')

data = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt').text

import nltk
nltk.download('stopwords')
import nltk
import os
import re
import math
import operator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.tokenize import sent_tokenize,word_tokenize
nltk.download('averaged_perceptron_tagger')
Stopwords = set(stopwords.words('english'))
wordlemmatizer = WordNetLemmatizer()

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
cur = sent_tokenize(data)

lab_data = cur
# text = word_tokenize(text[0])
import nltk
nltk.download('wordnet')

import re

nltk.download('stopwords')
from nltk.corpus import stopwords
ext = [',','"',".",';','(',')','’',',','“','”',':','``','>',"''"]
stop_words = set(stopwords.words('english')) 
# filtered_sentence = [w for w in text if not w in stop_words] 
# filtered_sentence = [w for w in filtered_sentence if not w in ext and "'"]
from nltk.stem import PorterStemmer
porter = PorterStemmer()
new_sent=[]
for sen in lab_data:
  sen = sen.lower()
  sen = word_tokenize(sen)
  filt = [w for w in sen if not w in stop_words]
  filt = [w for w in filt if not w in ext]
  # filt = [porter.stem(w) for w in filt]
  
  jn=""
  for z in filt:
    jn += z+" "
  op = re.sub('\d+[-,:]\d+[-,:]\d+',"",jn)
  new_sent.append(op)

  ## using word2vec for word embeddings and clustering of similar sentences together
from gensim.models import Word2Vec
import nltk
import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering

import pandas as pd
rel = pd.read_csv("drive/My Drive/out.csv")

#loading glove embeddings
def read_data(file_name):
    with open(file_name,'r') as f:
        word_vocab = set() # not using list to avoid duplicate entry
        word2vector = {}
        for line in f:
            line_ = line.strip() #Remove white space
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:],dtype=float)
    print("Total Words in DataSet:",len(word_vocab))
    return word_vocab,word2vector

vocab, w2v = read_data("drive/My Drive/glove.6B.200d.txt")


def vectorizer(sent,m):
  
  numw = 0
  sent = word_tokenize(sent)
  vec = np.zeros(200,dtype=float)
  # print(sent)
  for w in sent:
    try:
      if numw == 0:
        vec = m[w]
      else:
        vec = np.add(vec,m[w])
      numw += 1
    except:
      pass
    
    
  return np.array(vec)
l=[]

for i in final:
  l.append(vectorizer(i,w2v))  
X = np.array(l)

# m = Word2Vec(final,size = 50,min_count=1,sg=1)
n_clusters = 20
clf = KMeans(n_clusters = n_clusters,
             max_iter=100,
             init='k-means++',
             n_init=1)
labels = clf.fit_predict(X)

my_dict ={}
for i in range(len(labels)):
  if labels[i] not in my_dict.keys():
    my_dict[labels[i]]=list()
  my_dict[labels[i]].append(i)

def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
       if word not in words_unique:
          words_unique.append(word)
    for word in words_unique:
          dict_freq[word] = words.count(word)
    return dict_freq

def sentence_importance(sentence,dict_freq,sentences):
     sentence_score = 0
     sentence = remove_special_characters(str(sentence))
     sentence = re.sub(r'\d+', '', sentence)
     pos_tagged_sentence = []
     no_of_sentences = len(sentences)
     pos_tagged_sentence = pos_tagging(sentence)
     for word in pos_tagged_sentence:
         if word.lower() not in Stopwords and word not in Stopwords    and len(word)>1:
             word = word.lower() 
             word = wordlemmatizer.lemmatize(word)
             sentence_score = sentence_score + word_tfidf(dict_freq,word,sentences,sentence)
     return sentence_score

def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word,tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
            pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb

def word_tfidf(dict_freq,word,sentences,sentence):
    word_tfidf = []
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf

def tf_score(word,sentence):
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf

def idf_score(no_of_sentences,word,sentences):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        sentence = remove_special_characters(str(sentence))
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.split()
        sentence = [word for word in sentence if word.lower() not in Stopwords and len(word)>1]
        sentence = [word.lower() for word in sentence]
        sentence = [wordlemmatizer.lemmatize(word) for word in sentence]
        if word in sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    idf = math.log10(no_of_sentences/no_of_sentence_containing_word)
    return idf

def tf_idf_score(tf,idf):
   return tf*idf
def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
       lemmatized_words.append(wordlemmatizer.lemmatize(word))
    return lemmatized_words
def stem_words(words):
    stemmed_words = []
    for word in words:
       stemmed_words.append(stemmer.stem(word))
    return stemmed_words
def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text   

# use tf-idf function to get popular sentences from clusters
text =""
for i in final:
  text+=i
tokenized_sentence = sent_tokenize(text)
tokenized_words_with_stopwords = word_tokenize(text)
tokenized_words = [word for word in tokenized_words_with_stopwords if len(word) > 1]
tokenized_words = [word.lower() for word in tokenized_words]
word_freq = freq(tokenized_words)
import nltk
nltk.download('averaged_perceptron_tagger')

def freq_sent(k,lb):
  topk=[]
  # for j in my_dict[lb]:
  #   topk.append([vl(X[j],av),j])

  # for sentence in my_dict[lb]:
  #       if sentence[:15] in sentenceValue: 
  #           topk.append(sentenceValue[sentence[:15]])
  #           sentence_count += 1  
  
  

  for j in my_dict[lb]:
    current_sentence_score = sentence_importance(final[j],word_freq,tokenized_sentence)
    #print(final[j])
    if len(final[j]) > 20 :
      topk.append([current_sentence_score,j])
                 
    
  topk.sort()
  topk.reverse()
  #print(topk)
  ind = []
  for i in range(min(len(my_dict[lb]),k)):
    ind.append(topk[i][1])
  
  return ind;

my_set = set()

for i in range(20):
  sen = freq_sent(4,i)
  for j in sen:
    my_set.add(j)

# for i in range(len(cur)):
#   if i in my_set:
#     print(cur[i])
# print(data)
new_text= ""
for i in range(len(cur)):
  if i in my_set:
    new_text+=cur[i]


# for i in range(len(cur)):
#   if i in my_set:
#     print(cur[i])
cnt = dict()
for i in range(len(cur)):
  if labels[i] in cnt:
    cnt[labels[i]]+=1
  else:
    cnt[labels[i]]=1  
  
 from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
text1 = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt').text
print('summary:')
print(summarize(text1,ratio=0.009)) 