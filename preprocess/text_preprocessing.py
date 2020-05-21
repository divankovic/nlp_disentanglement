import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])


def check_token(token):
    return token and token.string.strip() and not token.is_stop and not token.is_punct and token.is_alpha


def preprocess_token(token):
    return token.lemma_.strip().lower()


def clean_text(text):
    text = nlp(text)
    return [preprocess_token(token) for token in text if check_token(token)]


def tokenize(text):
    return nlp(text)
