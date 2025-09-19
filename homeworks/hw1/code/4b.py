import re
import os

regex = r"(?i)\b\w*([aeiou])(\1{2,})\w*\b"

pattern = re.compile(regex)

blogs_folder = "../blogs"

words_per_vowel = dict()

for file in os.listdir(blogs_folder):
   with open(os.path.join(blogs_folder, file), "r", encoding="utf-8") as f:
      content = f.read()
      matches = [m.group().lower() for m in pattern.finditer(content)]
      if matches:
         for match in matches:
            vowel = re.search(r"(?i)([aeiou])\1{2,}", match).group(1)
            vowel = vowel.lower()
            if vowel not in words_per_vowel: words_per_vowel[vowel] = dict()
            match = re.sub(r"(?i)([aeiou])\1{2,}", vowel, match)
            words_per_vowel[vowel][match] = words_per_vowel[vowel].get(match, 0) + 1
three_most_commons_per_vowel = {k : sorted(v.items(), key = lambda item: item[1], reverse=True)[:3] for k, v in words_per_vowel.items()}

print(three_most_commons_per_vowel)