import re
import os

regex = r"(?i)\b\w*([aeiou])(\1{2,})\w*\b"

pattern = re.compile(regex)

blogs_folder = "../blogs"

words_per_vowel_m = dict()
words_per_vowel_f = dict()

for file in os.listdir(blogs_folder):
   with open(os.path.join(blogs_folder, file), "r", encoding="utf-8") as f:
      content = f.read()
      matches = [m.group().lower() for m in pattern.finditer(content)]
      if matches:
         for match in matches:
            vowel = re.search(r"(?i)([aeiou])\1{2,}", match).group(1)
            vowel = vowel.lower()
            if file.lower().startswith("f"):
               if vowel not in words_per_vowel_f: words_per_vowel_f[vowel] = dict()
               match = re.sub(r"(?i)([aeiou])\1{2,}", vowel, match)
               words_per_vowel_f[vowel][match] = words_per_vowel_f[vowel].get(match, 0) + 1
            elif file.lower().startswith("m"):
               if vowel not in words_per_vowel_m: words_per_vowel_m[vowel] = dict()
               match = re.sub(r"(?i)([aeiou])\1{2,}", vowel, match)
               words_per_vowel_m[vowel][match] = words_per_vowel_m[vowel].get(match, 0) + 1     
               
three_most_commons_per_vowel_f = {k : sorted(v.items(), key = lambda item: item[1], reverse=True)[:3] for k, v in words_per_vowel_f.items()}
three_most_commons_per_vowel_m = {k : sorted(v.items(), key = lambda item: item[1], reverse=True)[:3] for k, v in words_per_vowel_m.items()}

print("Female")   
print(three_most_commons_per_vowel_f)
print("Male")
print(three_most_commons_per_vowel_m)