import unicodedata
import re

_PUNCT_MAP = {"，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
              "、": ",", "（": "(", "）": ")", "“": '"', "”": '"', "‘": "'", "’": "'"}

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)          # folds full-width digits/latin
    text = "".join(_PUNCT_MAP.get(ch, ch) for ch in text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
