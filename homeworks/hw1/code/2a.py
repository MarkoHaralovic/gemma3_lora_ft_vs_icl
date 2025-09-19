from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")

def get_wordnet_pos(tag):
   if tag.startswith("J"):
       return wordnet.ADJ
   elif tag.startswith("V"):
       return wordnet.VERB
   elif tag.startswith("N"):
       return wordnet.NOUN
   elif tag.startswith("R"):
       return wordnet.ADV
   else:
       return wordnet.NOUN

file = "../sorted_types_HW1.txt"

wnl = WordNetLemmatizer()

with open(file, "r", encoding="utf-8") as f:
   words = f.read().splitlines()

lemmatized_words = set()
for word in words:
   pos_tag = nltk.pos_tag([word])[0][1]
   pos = get_wordnet_pos(pos_tag) 
   _lem_w = wnl.lemmatize(word, pos=pos)
   lemmatized_words.add(_lem_w)
   print ("{0:20}{1:20}".format(word,_lem_w))

print(f"Total number of words: {len(words)}")
print(f"Total number of lemmatized_words: {len(lemmatized_words)}")
