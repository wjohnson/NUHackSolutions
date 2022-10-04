from xml.dom.minidom import Document
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import json

#get the Azure Language Model (or Congtive Service Multimodel) key and endpoint from Azure portal
credential = AzureKeyCredential("enter cog key here")
endpoint="enter cog endpoint here"

#get the blob storage connection details
blob_connection_string = #"blob_connection_string"
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)

#get the input container name where the audio file is stored and where the converted text file has to be uploaded
input_container_name = "sentiment-analysis-input" #"input_container_name"
input_filename = "sentiment_analysis_raw_data.txt" #"input_filename"


#get the blob storage details where the output sentiment score needs to be stored
output_container_name = "sentiment-analysis-output" #"output_container_name"


#connect to the storage and download the text/doc (raw data) to which we need to run sentiment analysis
blob_container_client = blob_service_client.get_container_client(container=input_container_name)
blob_client = blob_service_client.get_blob_client(container=input_container_name, blob=input_filename)
data = blob_client.download_blob()
data = data.readall()
data = data.decode()
str_text = data.strip()


text_analytics_client = TextAnalyticsClient(endpoint, credential)

documents = str_text.split(".")
#print(documents)

#documents = [
#    "I did not like the restaurant. The food was somehow both too spicy and underseasoned. Additionally, I thought the location was too far away from the playhouse.",
#    "The restaurant was decorated beautifully. The atmosphere was unlike any other restaurant I've been to.",
#    "The food was yummy. :)"
#]

response = text_analytics_client.analyze_sentiment(documents, language="en")
result = [doc for doc in response if not doc.is_error]

output = ""
for doc in result:
    output = output + "," + str(doc)
    print (output)
    #print(f"Overall sentiment: {doc.sentiment}")

print(output)


#Prepare & Uploade the summary (json) to the output container
blob = BlobClient.from_connection_string(conn_str=blob_connection_string, container_name=output_container_name, blob_name=input_filename+"_sentiment_output.txt")
data = output
blob.upload_blob(data, overwrite=True)
