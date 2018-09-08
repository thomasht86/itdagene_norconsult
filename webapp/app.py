from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from functions import classify

# import update function from local dir
#from update import update_model

# import HashingVectorizer from local dir
#from vectorizer import vect

# Preparing the Classifier
#cur_dir = os.path.dirname(__file__)
#clf = pickle.load(open(os.path.join(cur_dir,
#				'pkl_objects/classifier.pkl'), 'rb'), encoding="latin1")
#db = os.path.join(cur_dir, 'reviews.sqlite')

def classify2(document):
	label = {0: 'negative', 1: 'positive'}
	X = vect.transform([document])
	y = clf.predict(X)[0]
	proba = np.max(clf.predict_proba(X))
	return label[y], proba

def prep(input_text):
	prepped = "a"
	return prepped

app = Flask(__name__)

class ReviewForm(Form):
	moviereview = TextAreaField('',
			[validators.DataRequired(), validators.length(min=15)])

@app.route('/')
def index():
	form = ReviewForm(request.form)
	return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
	form = ReviewForm(request.form)
	if request.method == 'POST' and form.validate():
		raw_text = request.form['moviereview']
		#prepped_text = prep(raw_text)
		preds = classify(raw_text)
		return render_template('results.html',
	content=raw_text,
	predictionie="IE",
	probabilityie=round(preds["IE"]*100, 2),
	predictionhf="HF",
	probabilityhf=round(preds["HF"]*100, 2),
	predictioniv="IV",
	probabilityiv=round(preds["IV"]*100, 2),
	predictionad="AD",
	probabilityad=round(preds["AD"]*100, 2),
	predictionsu="SU",
	probabilitysu=round(preds["SU"]*100, 2),
	predictionmh="MH",
	probabilitymh=round(preds["MH"]*100, 2),
	predictionnv="NV",
	probabilitynv=round(preds["NV"]*100, 2),
	predictionok="OK",
	probabilityok=round(preds["OK"]*100, 2)
	)
	return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
	feedback = request.form['feedback_button']
	review = request.form['review']
	prediction = request.form['prediction']
	inv_label = {'negative': 0, 'positive': 1}
	y = inv_label[prediction]
	if feedback == 'Incorrect':
		y = int(not(y))
	train(review, y)
	sqlite_entry(db, review, y)
	return render_template('thanks.html')


if __name__ == '__main__':
	#update_model(db_path=db, model=clf, batch_size=10000)
	app.run(debug=False, host='0.0.0.0', port=8000)
