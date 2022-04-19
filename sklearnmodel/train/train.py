import numpy as np
import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset, Run

if __name__ == "__main__":
    run = Run.get_context()
    ws = run.experiment.workspace

    dataset = Dataset.get_by_name(ws, name='comments')
    df = dataset.to_pandas_dataframe()
    df.columns = ['doc', 'sentiment']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', LatentDirichletAllocation(n_components=5,random_state=0))
    ])

    pipeline.fit(df.doc)
    topics = pipeline.transform(df.doc)
    topic_words = pipeline['model'].components_
    #  Define the number of Words that we want to print in every topic : n_top_words
    n_top_words = 5

    for i, topic_dist in enumerate(topic_words):
        
        # np.argsort to sorting an array or a list or the matrix acc to their values
        sorted_topic_dist = np.argsort(topic_dist)
        
        # Next, to view the actual words present in those indexes we can make the use of the vocab created earlier
        topic_words = np.array(pipeline['tfidf'].get_feature_names())[sorted_topic_dist]
        
        # so using the sorted_topic_indexes we ar extracting the words from the vocabulary
        # obtaining topics + words
        # this topic_words variable contains the Topics  as well as the respective words present in those Topics
        topic_words = topic_words[:-n_top_words:-1]
        print ("Topic", str(i+1), topic_words)

    with open('./outputs/model.joblib', 'wb') as fp:
        joblib.dump(pipeline, fp)
