import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import pandas as pd

stemmer = PorterStemmer()
remove_punctuation = lambda d: "".join([ (c if c not in string.punctuation+"\n\r\t" else " ") for c in d])
tokenize = lambda d: [stemmer.stem(w.lower()) for w in remove_punctuation(d).split(" ") if len(w)>0]

stoplist = pickle.load(open("stoplist.pkl","rb"))

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1, stop_words=stoplist)

X_all = pd.read_pickle("X_all.df")
all_doc = vec.fit_transform(X_all.description.values)

label_cols = ['IE', 'HF', 'IV', 'AD', 'SU', 'MH', 'NV', 'OK']
models = {c:pickle.load(open(c+"_model.pkl", "rb")) for c in label_cols}
rs = {c:np.load(open("r_"+c+".npy", "rb")) for c in label_cols}

def classify(text):
    preds = {}
    test_doc = vec.transform([text])
    print(test_doc.shape)
    for label, m in models.items():
        init_preds = m.predict_proba(test_doc.multiply(rs[label]))[:,1]
        print(init_preds.shape)
        preds[label] = init_preds[0]
    print(preds)
    return preds