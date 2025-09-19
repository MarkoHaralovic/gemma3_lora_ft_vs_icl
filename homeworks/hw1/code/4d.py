import re
import os

pattern = re.compile(r'([aeiou])\1{2,}', re.IGNORECASE)

blogs_folder = "../blogs"
blogs_output = "../outputs/normalized_blogs"
os.makedirs(blogs_output, exist_ok=True)

total_subs = 0

for fname in os.listdir(blogs_folder):
   in_path = os.path.join(blogs_folder, fname)
   out_path = os.path.join(blogs_output, fname)
   with open(in_path, "r", encoding="utf-8") as f:
      text = f.read()
   norm_text, subs_n = pattern.subn(r"\1", text)
   total_subs += subs_n

   with open(out_path, "w", encoding="utf-8") as f:
      f.write(norm_text)

print(f"{total_subs} corrected words.")
