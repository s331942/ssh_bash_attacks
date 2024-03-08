import string
import base64
import re

def decode_base64(word):
    try:
        return base64.b64decode(word).decode("utf-8")
    except:
        pass

def split_session(full_session):
    words = []
    for word in re.split(r"\n|;|,|/|-|\||=|$|>|<|$|:|{|}|\(|\)| ", full_session):
        if word.startswith('"') or word.endswith('"'):
            # remove the quotation mark at the start and at the end of the word
            word = word[1:-1]
        elif len(word) == 1 and word in string.punctuation:
            # remove that punctuation
            word = None
        words.append(word)
    return list(filter(None, words))

def clean_session(full_session):
    new_full_session = []
    for session_chunck in full_session.split(";"):
        if "base64 --decode" in session_chunck or "echo" in session_chunck:
            for word in session_chunck.split("\""):
                decode = decode_base64(word)
                if decode:
                    new_full_session.append(decode)
        else:
            new_full_session.append(session_chunck)
    return split_session("".join(new_full_session))