
import re

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r'\n+', ' ', text)

    text = re.sub(r'[^a-zA-Z0-9+#.\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()
