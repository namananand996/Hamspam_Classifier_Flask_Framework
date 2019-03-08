from flask import Flask, request, render_template,url_for
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split




app = Flask(__name__)
@app.route('/')
def main():
	return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():

	df = pd.read_csv("YoutubeSpamMergedData.csv")

	# Features and Labels
	X = df.iloc[:,5]
	y = df.iloc[:,6]

	# Extract Feature With CountVectorizer

	countvectorizer = CountVectorizer()
	X = countvectorizer.fit_transform(X) # Fit the Data
		
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	#Naive Bayes Classifier
		
	clasifier = MultinomialNB()
	clasifier.fit(X_train,y_train)
	clasifier.score(X_test,y_test)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = countvectorizer.transform(data).toarray()
		my_prediction = clasifier.predict(vect)
		print(my_prediction)


	return render_template("result.html" ,prediction=my_prediction)

if __name__ == "__main__":
	app.run(debug=True)






