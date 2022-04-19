import os
import time
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


local_path = "./data"
container_name = "data"
local_file_name = "sentiment.tab"
cloud_file_name = "amazon_cells_labelled.txt"
download_file_path = os.path.join(local_path, local_file_name)
storage_connection_str = endpoint = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
cog_svcs_endpoint = os.environ["AZURE_TEXT_ANALYTICS_ENDPOINT"]
cog_svcs_key = os.environ["AZURE_TEXT_ANALYTICS_KEY"]

if not os.path.exists(local_path):
    os.makedirs(local_path)

try:
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=cloud_file_name)
    
    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

except Exception as ex:
    print('Exception:')
    print(ex)

# Start Text Analytics
with open(download_file_path, "r") as fp:
    documents = fp.readlines()

# Strip off the trailing digit
labeled_documents = [d.strip().split("\t") for d in documents][0:20]
documents = [d[0] for d in labeled_documents]

text_analytics_client = TextAnalyticsClient(
    endpoint=cog_svcs_endpoint, 
    credential=AzureKeyCredential(cog_svcs_key)
)

scored_docs = []

if len(documents) > 10:
    start = 0
    end = 10
    for start in range(0,len(documents), 10):
        print(f"Iterating from {start} to {start+10}")
        batch = documents[start:(start+10)]
        scored_docs.extend(text_analytics_client.analyze_sentiment(batch, show_opinion_mining=True))
        time.sleep(1)
else:
    scored_docs = text_analytics_client.analyze_sentiment(documents, show_opinion_mining=True)
    docs = [doc for doc in scored_docs if not doc.is_error]

# Store the results for later analysis
relabeled_results = []
for doc, labeled_doc in zip(scored_docs, labeled_documents):

    true_positive = True if doc.sentiment == "positive" and labeled_doc[1] == '1' else False
    true_negative = True if doc.sentiment == "negative" and labeled_doc[1] == '0' else False
    is_correctly_labeled = (true_positive or true_negative)
    relabeled_results.append((doc.sentiment, is_correctly_labeled, doc.confidence_scores, doc.sentences))

with open('./sentimentlabeled.tab', 'w') as fp:
    for res in relabeled_results:
        fp.write(str(res)+'\n')
