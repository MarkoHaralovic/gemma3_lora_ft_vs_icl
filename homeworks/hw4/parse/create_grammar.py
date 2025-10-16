import spacy

nlp = spacy.load("en_core_web_sm")
sentence_file = "./sentences.txt"
lexicon_file = "./exercise1_test.lex"

with open(sentence_file, "r", encoding="utf-8") as f:
   lines = f.readlines()

word_dict = {
   "NOUN": set(),
   "ADP": set(),
   "VERB": set(),
   "INTJ": set(),
   "PRON": set(),
   "ADJ": set(),
   "DET": set() 
}

for line in lines:
   for word in line.split():
      token = nlp(word)[0]
      if token.pos_ == "SPACE":
         continue
      if token.pos_ in word_dict:
         word_dict[token.pos_].add(token.text)

pos_map = {
   "NOUN": "Noun",
   "VERB": "Verb",
   "ADP": "Prep",
   "DET": "Det",
   "PRON": "Pron",
   "INTJ": "Intj",
   "ADJ": "Adj"
}