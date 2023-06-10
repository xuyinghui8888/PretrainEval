import pandas as pd
import json
from tqdm import tqdm
import os

if __name__ == '__main__':

    file_path = './models_outputs/'
    for root, dirs, files in os.walk(file_path):
        df = pd.read_csv(file_path + files[0])
        df.columns = ['Question','model 0','GPT3.5','Score 0']
        df = df[['Question','GPT3.5','model 0','Score 0']]
        # print(df)
        for i, file in tqdm(enumerate(files[1:])):
            tmp = pd.read_csv(file_path + file)
            tmp.columns = ['Question','model {}'.format(i+1),'GPT3.5','Score {}'.format(i+1)]
            tmp = tmp[['model {}'.format(i+1),'Score {}'.format(i+1)]]

            # df = pd.merge(df,tmp)
            df = df.join(tmp)

        df.to_csv(file_path + 'merge.csv',index=False,encoding='utf_8_sig')
        print("Done!")