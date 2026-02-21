import re
import zipfile
import emoji

# Unpack the archive
with zipfile.ZipFile('data/raw_data.txt.zip', 'r') as z:
    z.extractall('data/')

# Read the raw data
with open('data/raw_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

sample_chars = 300
print(f"Original length: {len(text)} chars")
print(f"Sample (first {sample_chars} chars):")
print(text[:sample_chars])

# Lowercase
text = text.lower()

# Remove URLs (http/https/www links)
text = re.sub(r'http\S+|www\.\S+', '', text)

# Remove mentions (@username)
text = re.sub(r'@\w+', '', text)

# Remove emojis
text = emoji.replace_emoji(text, replace='')

# Remove all symbols except letters, numbers, and whitespace
text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)

# Remove duplicate whitespaces (spaces, tabs, etc.) and strip lines
text = re.sub(r'[^\S\n]+', ' ', text)   # collapse whitespaces to single space
text = re.sub(r' *\n *', '\n', text)    # clean spaces around newlines
text = re.sub(r'\n{2,}', '\n', text)    # collapse multiple newlines
text = text.strip()

print(f"Cleaned length: {len(text)} chars")
print(f"\nCleaned text (first {sample_chars} chars):")
print(text[:sample_chars])