from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

key = "<< Replace this with your azure langunage model key >>"
endpoint = "<< Replace this with your azure langunage model endpoint >>"

# Authenticate the client using your key and endpoint 
def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=ta_credential)
    return text_analytics_client

client = authenticate_client()

# Example method for summarizing text
def sample_extractive_summarization(client):
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.textanalytics import (
        TextAnalyticsClient,
        ExtractSummaryAction
    ) 

    #get the blob storage connection details
    blob_connection_string = "<< Replace this with your azure storage account connection string >>"
    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
    
    #get the input container name where the source text file is available
    input_container_name = "<< Replace this with your azure storage account container name that has the text data which needs to be summarized >>"
    input_filename = "<< Replace this with your file name (text file) that need to be summarized >>"


    #connect to the storage and download the text/doc (raw data)
    blob_container_client = blob_service_client.get_container_client(container=input_container_name)
    blob_client = blob_service_client.get_blob_client(container=input_container_name, blob=input_filename)
    data = blob_client.download_blob()
    data = data.readall()
    data = data.decode()
    str_text = data.strip()

    document = str_text.splitlines()

    poller = client.begin_analyze_actions(
        document,
        actions=[
            ExtractSummaryAction(max_sentence_count=4)
        ],
    )

    document_results = poller.result()
    for result in document_results:
        extract_summary_result = result[0]  # first document, first result
        if extract_summary_result.is_error:
            print("...Is an error with code '{}' and message '{}'".format(
                extract_summary_result.code, extract_summary_result.message
            ))
        else:
            print("Summary extracted: \n{}".format(
                " ".join([sentence.text for sentence in extract_summary_result.sentences]))
            )

sample_extractive_summarization(client)