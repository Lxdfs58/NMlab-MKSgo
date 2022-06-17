import boto3
import logging
from botocore.exceptions import ClientError
import os
from sys import argv
import os
from datetime import datetime

from apiclient import discovery
from google.oauth2 import service_account

def uploadToS3 (user, service, timestamp): 
    #Create Folder in Bucket First 
    client = boto3.client('s3')
    bucket_name = "mksuserdata"
    folder_name = f"{user}_{service}_{timestamp}"

    #Upload Image in Local Folder to AWS Folder 
    walks = os.walk(folder_name)
    for source, dirs, files in walks:
        print('Directory: ' + source)
        for filename in files:
            # construct the full local path
            local_file = os.path.join(source, filename)
            # construct the full aws path
            relative_path = os.path.relpath(local_file, folder_name)
            s3_file = os.path.join(folder_name, relative_path)
            # Invoke upload function
            client.upload_file(local_file, bucket_name,  s3_file)
    return 

def upload_googleSheet(user, b_or_r,  items_count):
    try:
        scopes = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/spreadsheets"]
        secret_file = os.path.join(os.getcwd(), 'hand-gesture-recognition-using-mediapipe/utils/client_secret.json')
        print(secret_file)
        spreadsheet_id = '1DzYlxn3dQXQfFlPS6e8uURMfzrH492sKY-Q8VGsq07k'
        range_name = 'borrow_return!A8:O'

        credentials = service_account.Credentials.from_service_account_file(secret_file, scopes=scopes)
        service = discovery.build('sheets', 'v4', credentials=credentials)
        now = datetime.now()
        values = [
            [ now.strftime("%Y-%m-%d, %H:%M:%S"), f'{user}', f'{b_or_r}'] + items_count,
        ]


        data = {
            'values' : values 
        }

        service.spreadsheets().values().append(spreadsheetId=spreadsheet_id, body=data, range=range_name, valueInputOption='USER_ENTERED', insertDataOption = "INSERT_ROWS").execute()
        return 
    except OSError as e:
        print(e)
        return 


def show_custom_labels(model,bucket,photo, min_confidence):
    client=boto3.client('rekognition')

    #Call DetectCustomLabels
    response = client.detect_custom_labels(Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
        MinConfidence=min_confidence,
        ProjectVersionArn=model)
    
    item_list = []
    score = 0
    # For object detection use case, uncomment below code to display image. 
    for customLabel in response['CustomLabels']:
        item_list.append(str(customLabel['Name']))
        score += customLabel['Confidence']
        print('Label ' + str(customLabel['Name']))
        print('Confidence ' + str(customLabel['Confidence']))

    return item_list, score

def finalCount(get_list, put_list):
    items_count = []
    items_count.append(get_list.count("brush") )
    items_count.append(get_list.count("hammer"))
    items_count.append(get_list.count("hex key"))
    items_count.append(get_list.count("hot glue"))
    items_count.append(get_list.count("saw"))
    items_count.append(get_list.count("scissors"))
    items_count.append(get_list.count("screwdriver"))
    items_count.append(get_list.count("softtape"))
    items_count.append(get_list.count("tape"))
    items_count.append(get_list.count("tapemeasure"))
    items_count.append(get_list.count("tweezer"))
    items_count.append(get_list.count("wrench"))

    items_count[0]-= put_list.count("brush") 
    items_count[1]-= put_list.count("hammer")
    items_count[2]-= put_list.count("hex key")
    items_count[3]-= put_list.count("hot glue")
    items_count[4]-= put_list.count("saw")
    items_count[5]-= put_list.count("scissors")
    items_count[6]-= put_list.count("screwdriver")
    items_count[7]-= put_list.count("softtape")
    items_count[8]-= put_list.count("tape")
    items_count[9]-= put_list.count("tapemeasure")
    items_count[10] -= put_list.count("tweezer")
    items_count[11] -= put_list.count("wrench")

    items_count = list(map(abs, items_count))
    items_count.append([])

    if items_count[0] != 0 : items_count[12].append("brush") 
    if items_count[1] != 0 : items_count[12].append("hammer")
    if items_count[2] != 0 : items_count[12].append("hex key")
    if items_count[3] != 0 : items_count[12].append("hot glue")
    if items_count[4] != 0 : items_count[12].append("saw")
    if items_count[5] != 0 : items_count[12].append("scissors")
    if items_count[6] != 0 : items_count[12].append("screwdriver")
    if items_count[7] != 0 : items_count[12].append("softtape")
    if items_count[8] != 0 : items_count[12].append("tape")
    if items_count[9] != 0 : items_count[12].append("tapemeasure")
    if items_count[10] != 0 : items_count[12].append("tweezer")
    if items_count[11] != 0 : items_count[12].append("wrench")
   
    items_count[12] = ', '.join(items_count[12])
    return items_count


def uploadAndRun(user, service, timestamp, getCount, putCount):
    #1. Upload images to S3
    bucket='mksuserdata'
    uploadToS3(user, service, timestamp)
    
    #2. Run Image Analysis Model in Rekognition 
    model='arn:aws:rekognition:ap-northeast-1:163487637425:project/tooldetect/version/tooldetect.2022-06-16T17.11.52/1655370712440'
    min_confidence=50
    
    #look for get items
    get_list = []
    print("get")
    for i in range (getCount):
        item_list = []
        score_list = []
        for j in range (6):
            photo=f'{user}_{service}_{timestamp}/get{i}/{j}.jpg'
            items, confidence =show_custom_labels(model,bucket,photo, min_confidence)
            item_list.append(items)
            score_list.append(confidence)
        max_value = max(score_list)
        max_index = score_list.index(max_value)
        get_list += item_list [max_index]
    print(get_list)

    #look for put items
    put_list = []
    print("put")
    for i in range (putCount):
        item_list = []
        score_list = []
        for j in range (6):
            photo=f'{user}_{service}_{timestamp}/put{i}/{j}.jpg'
            items, confidence =show_custom_labels(model,bucket,photo, min_confidence)
            item_list.append(items)
            score_list.append(confidence)
        max_value = max(score_list)
        max_index = score_list.index(max_value)
        put_list += (item_list [max_index])
    print(put_list)
    
    #Finalize detail of this service
    items_count =list(map(abs, finalCount(get_list, put_list)))
    
    #Upload Data to Google Sheet
    upload_googleSheet(user, service,  items_count)

def main():

    user = "b08901199"
    service = "borrow"
    timestamp = "202206162255"
    getCount = 1
    putCount = 3

    #1. Upload images to S3
    bucket='mksuserdata'
    uploadToS3(user, service, timestamp)
    
    #2. Run Image Analysis Model in Rekognition 
    model='arn:aws:rekognition:ap-northeast-1:163487637425:project/tooldetect/version/tooldetect.2022-06-16T17.11.52/1655370712440'
    min_confidence=50
    
    #look for get items
    get_list = []
    print("get")
    for i in range (getCount+1):
        item_list = []
        score_list = []
        for j in range (6):
            photo=f'{user}_{service}_{timestamp}/get{i}/{j}.jpg'
            items, confidence =show_custom_labels(model,bucket,photo, min_confidence)
            item_list.append(items)
            score_list.append(confidence)
        max_value = max(score_list)
        max_index = score_list.index(max_value)
        get_list += item_list [max_index]
    print(get_list)

    #look for put items
    put_list = []
    print("put")
    for i in range (putCount+1):
        item_list = []
        score_list = []
        for j in range (6):
            photo=f'{user}_{service}_{timestamp}/put{i}/{j}.jpg'
            items, confidence =show_custom_labels(model,bucket,photo, min_confidence)
            item_list.append(items)
            score_list.append(confidence)
        max_value = max(score_list)
        max_index = score_list.index(max_value)
        put_list += (item_list [max_index])
    print(put_list)
    
    #Finalize detail of this service
    items_count =finalCount(get_list, put_list)

    items_count = [0,0,0,0,0,0,0,0,0,0,0,1, "wrench"]
    
    #Upload Data to Google Sheet
    upload_googleSheet(user, service,  items_count)

if __name__ == "__main__":
    #upload_googleSheet("qq", "qq",  [0, 1,3])
    user = "b08901093"
    service = "borrow"
    items_count = [0,0,0,0,0,0,0,0,0,0,0,1, "Wrench"]
    upload_googleSheet(user, service,  items_count)
    #main()