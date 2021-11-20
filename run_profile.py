import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re  # regular expression
import numpy as np  # linear algebra
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))


from sklearn.preprocessing import MultiLabelBinarizer

def run_model(input_string_value):
    filename = 'models/author_model.sav'
    with open(filename, 'rb') as f:
        model = pickle.load(f)

        data = pd.read_csv("models/blogtext.csv")

        data = data.head(10000)
        print(data.info())
        data.drop(['id', 'date'], axis=1, inplace=True)
        print(data.head())
        data['age'] = data['age'].astype('object')
        print(data.info())
        data['clean_data'] = data['text'].apply(lambda x: re.sub(r'[^A-Za-z]+', ' ', x))
        data['clean_data'] = data['clean_data'].apply(lambda x: x.lower())
        data['clean_data'] = data['clean_data'].apply(lambda x: x.strip())
        print("Actual data=======> {}".format(data['text'][1]))
        print("Cleaned data=======> {}".format(data['clean_data'][1]))
        data['clean_data'] = data['clean_data'].apply(
            lambda x: ' '.join([words for words in x.split() if words not in stopwords]))
        print(data['clean_data'][6])
        data['labels'] = data.apply(lambda col: [col['gender'], str(col['age']), col['topic'], col['sign']], axis=1)
        print(data.head())
        data = data[['clean_data', 'labels']]
        print(data.head())
        X = data['clean_data']
        print('===================')
        print(X)
        print('************')
        print(type(X))
        Y = data['labels']
        vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
        X = vectorizer.fit_transform(X)

        label_counts = dict()

        for labels in data.labels.values:
            for label in labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
        print(label_counts)
        binarizer = MultiLabelBinarizer(classes=sorted(label_counts.keys()))
        Y = binarizer.fit_transform(data.labels)

        input_str = input_string_value
        input_str = re.sub(r'[^A-Za-z]+', ' ', input_str)
        input_str = input_str.lower()
        input_str = input_str.strip()

        Xtest = vectorizer.transform([input_str])
        result = model.predict(Xtest)
        print(result)
        y_test_inversed = binarizer.inverse_transform(result)
        print(y_test_inversed)
        return y_test_inversed

if __name__ == '__main__':
    print('Running..')
    run_model('This is the day that the Lord has made')