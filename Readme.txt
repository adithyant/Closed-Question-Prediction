Closed Question Prediction
-Add the path where CLosedQPrediction.py file is present to PYTHONPATH
-Commands:

Import ClosedQPrediction
Import pandas as pd
Data=pd.read_csv(“path to your test file”)
predictedValues= ClosedQPrediction.Predict(Data)

	
-Note: Please that the path where the following models are present :
	Model->contains the trained RF model.
			FileName->ClosedQModed
Vectorizer1->contains word vectors for tag1.
		FileName->vec1
Vectorizer2->contains word vectors for body.
		FileName->vec2
Vectorizer3->contains word vectors for title.
		FileName->vec3
Vectorizer4->contains word vectors for  tag1-5.
		FileName->vec4
-Ignore the Training_util file.it is a basic feature extraction code for future data. 
