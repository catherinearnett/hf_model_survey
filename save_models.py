import pandas as pd
import time
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

import matplotlib.pyplot as plt

import numpy as np

goldfish_info = pd.read_csv('data/goldfish_data_info.tsv', sep='\t')

goldfish_langs = list(goldfish_info['lang'])

goldfish_langs = [x[:3] for x in goldfish_langs]

goldfish_langs = list(set(goldfish_langs))


def fetch_all_models(task="text-classification", delay=2.0):
    api = HfApi()
    
    try:
        time.sleep(delay)  
        
        models_iter = api.list_models(filter=task, gated=False)
        
        all_models = []
        count = 0
        for model in models_iter:
            all_models.append(model)
            count += 1
            if count % 10000 == 0:
        
        return {model.id for model in all_models}  
        
    except HfHubHTTPError as e:
        if e.response.status_code == 429:  #rate limit
            wait_time = delay * 2
            time.sleep(wait_time)

            #retry
            models_iter = api.list_models(filter=task, gated=False)
            all_models = list(models_iter)
            return {model.id for model in all_models}
        else:
            print(f"HTTP Error: {e}")
            raise
    except KeyboardInterrupt:
        return set()
    except Exception as e:
        print(f"Error: {e}")
        try:
            models_iter = api.list_models(filter=task, gated=False, limit=1000000)
            all_models = list(models_iter)
            return {model.id for model in all_models}
        except:
            return set()

def save_models(models, filename="bidirectional_models.txt"):
    with open(filename, 'w') as f:
        for model_id in sorted(models):
            f.write(f"{model_id}\n")

if __name__ == "__main__":
    models = fetch_all_models()
    if models:
        save_models(models)


def fetch_all_models(task="text-generation", delay=2.0):
    api = HfApi()
    
    try:
        time.sleep(delay)  
        
        models_iter = api.list_models(filter=task, gated=False)
        
        all_models = []
        count = 0
        for model in models_iter:
            all_models.append(model)
            count += 1
            if count % 10000 == 0:
        
        return {model.id for model in all_models}  
        
    except HfHubHTTPError as e:
        if e.response.status_code == 429:  #rate limit
            wait_time = delay * 2
            time.sleep(wait_time)

            #retry
            models_iter = api.list_models(filter=task, gated=False)
            all_models = list(models_iter)
            return {model.id for model in all_models}
        else:
            print(f"HTTP Error: {e}")
            raise
    except KeyboardInterrupt:
        return set()
    except Exception as e:
        print(f"Error: {e}")
        try:
            models_iter = api.list_models(filter=task, gated=False, limit=1000000)
            all_models = list(models_iter)
            return {model.id for model in all_models}
        except:
            return set()

def save_models(models, filename="autoregressive_models.txt"):
    with open(filename, 'w') as f:
        for model_id in sorted(models):
            f.write(f"{model_id}\n")

if __name__ == "__main__":
    models = fetch_all_models()
    if models:
        save_models(models)