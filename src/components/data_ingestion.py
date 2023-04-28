import os
import sys
from src.logger  import logging
from src.exception  import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

#Initialise Data Ingestion config

@dataclass
class DataIngestionConfig:
    train_data_path:str =os.path.join('artifacts','CreditCardClient_clean.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')
#create a class for Data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig();
    
    def initiate_data_ingestion(self):
        try:

            logging.info("Initiate Data ingestion")         
            df=pd.read_csv(os.path.join('notebook/data','CreditCardClient_clean.csv'))
            logging.info('Dataset read using pandas Dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test data split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
            logging.info('Storing train & Test data to csv')
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion Data completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            logging.info('Exception occured at Data Ingestion Stage') 
            raise CustomException(e,sys)
    
    

