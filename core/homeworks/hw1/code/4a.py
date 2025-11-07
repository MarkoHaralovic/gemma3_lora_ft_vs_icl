import re
import os

regex = r"(?i)\b\w*([aeiou])(\1{2,}|\1(?!\1)[aeiou])\w*\b"
regex2 = r"(?i)\b\w*([aeiou])(\1{2,})\w*\b"

pattern = re.compile(regex)
pattern2 = re.compile(regex2)

blogs_folder = "../blogs"
blogs_output = "../outputs/4a_blogs"

count_of_matches = 0
count_of_matches2 = 0

matches_1, matches_2 = dict(),dict()

for file in os.listdir(blogs_folder):
   with open(os.path.join(blogs_folder, file), "r", encoding="utf-8") as f:
      content = f.read()
      matches = [m.group() for m in pattern.finditer(content)]
      matches2 = [m.group() for m in pattern2.finditer(content)]
      if matches:
         count_of_matches += len(matches)
         count_of_matches2 += len(matches2)
         for match in matches:
            if match not in matches_1: matches_1[match] = 1
            else: matches_1[match] += 1
         for match in matches2:
            if match not in matches_2: matches_2[match] = 1
            else: matches_2[match] += 1
         with open(os.path.join(blogs_output, file), "w", encoding="utf-8") as out:
               out.write("Regex 1\n")
               out.write("\n".join(matches))
               out.write("\n\n\n")
               out.write("Regex 2\n")
               out.write("\n".join(matches2))

matches_1_set = set(matches_1.keys())
matches_2_set = set(matches_2.keys())

diff = matches_1_set.symmetric_difference(matches_2_set)
print(f"Found {count_of_matches} matches using regex 1")
print(f"Found {count_of_matches2} matches using regex 2")
print(f"Word diff : {diff}")

for word in diff:
   print(f" - {word}, occurences in matches 2: {matches_1.get(word, 0)}")
