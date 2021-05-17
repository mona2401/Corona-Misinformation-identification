from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle
import joblib

app = Flask(__name__)
pickle_in = open('model-2.pkl','rb')
pac = pickle.load(pickle_in)
tfid = open('tfidfvect-2.pkl','rb')
tfidf_vectorizer = joblib.load('tfidfvect-2.pkl')

@app.route('/')
def home():
 	return render_template("index.html")

@app.route('/newscheck')
def newscheck():	
	abc = request.args.get('news')	
	input_data = [abc.rstrip()]
	# transforming input
	tfidf_test = tfidf_vectorizer.transform(input_data)
	# predicting the input
	y_pred = pac.predict(tfidf_test)
	return jsonify(result = y_pred[0])


if __name__=='__main__':
    app.run(debug=True)
