import os
import pandas as pd 

dataset_folder = "./aclImdb"

def load_dataset():
    data = {'review':[],'sentiment':[]}
    for split in ['train','test']:
        for sentiment in ['pos','neg']:
            folder = os.path.join(dataset_folder,split,sentiment)
            for file_name in os.listdir(folder):
                with open(os.path.join(folder,file_name),'r',encoding="UTF-8") as file:
                    review = file.read()
                    data['review'].append(review)
                    data['sentiment'].append(sentiment)
    return pd.DataFrame(data)

dataset = load_dataset()

print(dataset.head())