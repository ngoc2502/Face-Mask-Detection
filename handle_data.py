import pandas as pd
import os
import xml.etree.ElementTree as ET
from PIL import Image

def getdata(path):
    # Create paths
    images_path = []
    annotations_path = []
    #Divice path follow type
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            if file_path[-3:] == 'xml':
                annotations_path.append(file_path)
            else:
                images_path.append(file_path)

    #check if loaded all data
    print(len(images_path))
    if len(images_path) == 853:
        print("Ok Load done")
    else:
        print("Not Ok...")

    #create dataframe to save data
    df=pd.DataFrame(columns=['xmin','xmax','ymin','ymax','file_name','label'])
  
    for xml_file in annotations_path:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        file_name = root.find('filename').text
    
    # foreach object
        for obj in root.findall('object'):
            r = {
                
                'xmin': int(obj.find('bndbox').find('xmin').text),
                'ymin': int(obj.find('bndbox').find('ymin').text),
                'xmax': int(obj.find('bndbox').find('xmax').text),
                'ymax': int(obj.find('bndbox').find('ymax').text),
                'file_name': file_name,
                'label': obj.find('name').text,
            }

            df = df.append(r, ignore_index=True)
    return annotations_path,images_path,df

def train_test_valid_split(df):
    Ytest=df.sample(frac=0.1, replace=False)
    Xtest=pd.DataFrame(columns=['file_name'])   
    Xtest=Ytest.pop('file_name')

    Yvali=df.sample(frac=0.1,replace=False)
    Xvali=pd.DataFrame(columns=['file_name'])
    Xvali=Yvali.pop('file_name')

    Ytrain=df.sample(frac=0.8,replace=False)
    Xvali=pd.DataFrame(columns=['file_name'])
    Xtrain=Ytrain.pop('file_name')

    return  Xtrain,Ytrain,Xtest,Ytest,Xvali,Yvali

# path='E:/UwayInternshipLearning/FaceMaskDetection/Face-Mask-Detection/input'
# xml_path,image_path,df=getdata(path)
# train_X, train_Y, test_X, test_Y, val_X, val_Y=train_test_valid_split(df)

# print('train_X.shape =', train_X.shape)
# print('train_Y.shape =', train_Y.shape)
# print('val_X.shape   =', val_X.shape)
# print('val_Y.shape   =', val_Y.shape)
# print('test_X.shape  =', test_X.shape)
# print('test_Y.shape  =', test_Y.shape)

# img=Image.open(image_path[5])
# img.show()


