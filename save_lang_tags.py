import pandas as pd
import time
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
from tqdm import tqdm
import os


def get_language_tags(all_tags):

    lang_names = {
    'afar': 'aa',
    'abkhazian': 'ab',
    'avestan': 'ae',
    'afrikaans': 'af',
    'akan': 'ak',
    'amharic': 'am',
    'aragonese': 'an',
    'arabic': 'ar',
    'assamese': 'as',
    'avaric': 'av',
    'aymara': 'ay',
    'azerbaijani': 'az',
    'bashkir': 'ba',
    'bambara': 'bm',
    'belarusian': 'be',
    'bengali': 'bn',
    'bislama': 'bi',
    'tibetan': 'bo',
    'bosnian': 'bs',
    'breton': 'br',
    'bulgarian': 'bg',
    'catalan': 'ca',
    'czech': 'cs',
    'chamorro': 'ch',
    'chechen': 'ce',
    'church slavic': 'cu',
    'chuvash': 'cv',
    'welsh': 'cy',
    'danish': 'da',
    'german': 'de',
    'divehi': 'dv',
    'dzongkha': 'dz',
    'greek': 'el',
    'english': 'en',
    'esperanto': 'eo',
    'estonian': 'et',
    'basque': 'eu',
    'ewe': 'ee',
    'faroese': 'fo',
    'persian': 'fa',
    'fijian': 'fj',
    'finnish': 'fi',
    'french': 'fr',
    'western frisian': 'fy',
    'fulah': 'ff',
    'scottish gaelic': 'gd',
    'irish': 'ga',
    'galician': 'gl',
    'manx': 'gv',
    'guaraní': 'gn',
    'gujarati': 'gu',
    'haitian': 'ht',
    'hausa': 'ha',
    'hebrew': 'he',
    'herero': 'hz',
    'hindi': 'hi',
    'hiri motu': 'ho',
    'croatian': 'hr',
    'hungarian': 'hu',
    'armenian': 'hy',
    'igbo': 'ig',
    'ido': 'io',
    'sichuan yi': 'ii',
    'inuktitut': 'iu',
    'interlingue': 'ie',
    'interlingua': 'ia',
    'indonesian': 'id',
    'inupiaq': 'ik',
    'icelandic': 'is',
    'italian': 'it',
    'javanese': 'jv',
    'japanese': 'ja',
    'kalaallisut': 'kl',
    'kannada': 'kn',
    'kashmiri': 'ks',
    'georgian': 'ka',
    'kanuri': 'kr',
    'kazakh': 'kk',
    'khmer': 'km',
    'kikuyu': 'ki',
    'kinyarwanda': 'rw',
    'kyrgyz': 'ky',
    'komi': 'kv',
    'kongo': 'kg',
    'korean': 'ko',
    'kwanyama': 'kj',
    'kurdish': 'ku',
    'lao': 'lo',
    'latin': 'la',
    'latvian': 'lv',
    'limburgish': 'li',
    'lingala': 'ln',
    'lithuanian': 'lt',
    'luxembourgish': 'lb',
    'luba-katanga': 'lu',
    'ganda': 'lg',
    'marshallese': 'mh',
    'malayalam': 'ml',
    'marathi': 'mr',
    'macedonian': 'mk',
    'malagasy': 'mg',
    'maltese': 'mt',
    'mongolian': 'mn',
    'maori': 'mi',
    'malay': 'ms',
    'burmese': 'my',
    'nauru': 'na',
    'navajo': 'nv',
    'southern ndebele': 'nr',
    'northern ndebele': 'nd',
    'ndonga': 'ng',
    'nepali': 'ne',
    'dutch': 'nl',
    'norwegian nynorsk': 'nn',
    'norwegian bokmål': 'nb',
    'norwegian': 'no',
    'chichewa': 'ny',
    'occitan': 'oc',
    'ojibwe': 'oj',
    'oriya': 'or',
    'oromo': 'om',
    'ossetian': 'os',
    'punjabi': 'pa',
    'pāli': 'pi',
    'polish': 'pl',
    'portuguese': 'pt',
    'pashto': 'ps',
    'quechua': 'qu',
    'romansh': 'rm',
    'romanian': 'ro',
    'kirundi': 'rn',
    'russian': 'ru',
    'sango': 'sg',
    'sanskrit': 'sa',
    'sinhalese': 'si',
    'slovak': 'sk',
    'slovenian': 'sl',
    'northern sami': 'se',
    'samoan': 'sm',
    'shona': 'sn',
    'sindhi': 'sd',
    'somali': 'so',
    'southern sotho': 'st',
    'spanish': 'es',
    'albanian': 'sq',
    'sardinian': 'sc',
    'serbian': 'sr',
    'swati': 'ss',
    'sundanese': 'su',
    'swahili': 'sw',
    'swedish': 'sv',
    'tahitian': 'ty',
    'tamil': 'ta',
    'tatar': 'tt',
    'telugu': 'te',
    'tajik': 'tg',
    'tagalog': 'tl',
    'thai': 'th',
    'tigrinya': 'ti',
    'tonga': 'to',
    'tswana': 'tn',
    'tsonga': 'ts',
    'turkmen': 'tk',
    'turkish': 'tr',
    'twi': 'tw',
    'uyghur': 'ug',
    'ukrainian': 'uk',
    'urdu': 'ur',
    'uzbek': 'uz',
    'venda': 've',
    'vietnamese': 'vi',
    'volapük': 'vo',
    'walloon': 'wa',
    'wolof': 'wo',
    'xhosa': 'xh',
    'yiddish': 'yi',
    'yoruba': 'yo',
    'zhuang': 'za',
    'chinese': 'zh',
    'zulu': 'zu',
    }

    iso = pd.read_csv('data/iso_all.tsv', sep='\t')
    
    iso = iso.dropna(subset=['Part2b']).reset_index(drop=True)
    
    iso2 = iso[['Part2b', 'Part2t']]
    iso2 = iso2.dropna(subset=['Part2t']).reset_index(drop=True)
    
    iso2.columns = ['iso2', 'iso3']
    
    iso2_mapping = iso2.set_index('iso2')['iso3'].to_dict()
    
    to_delete = [k for k, v in iso2_mapping.items() if k == v]
    
    for k in to_delete:
        del iso2_mapping[k]
    
    iso = iso[['Part1', 'Part2t']]
    iso = iso.dropna(subset=['Part1']).reset_index(drop=True)
    
    iso.columns = ['iso2', 'iso3']
    
    iso_mapping = iso.set_index('iso2')['iso3'].to_dict()
    
    language_tags = []
    for t in all_tags:
        tag = None  
        if t in goldfish_langs:
            tag = t
        elif t in iso_mapping:
            tag = iso_mapping[t]
        elif t in iso_mapping.values():
            tag = t
        elif t in lang_names:
            t_code = lang_names[t]
            if t_code in iso_mapping:
                tag = iso_mapping[t_code]
        elif t in iso2_mapping:
            tag = iso2_mapping[t]
        
        if tag:  # deal with special cases
            if tag == 'ara':
                tag = 'arb'
            elif tag == 'cmn':
                tag = 'zho'
            
            if tag not in language_tags:
                language_tags.append(tag)
                
    return language_tags



def load_existing_results(csv_path='model_lang_tags.csv'):
    """Load existing results and return processed model names set and dataframe"""
    try:
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            if len(existing_df) > 0 and 'model_name' in existing_df.columns:
                processed_models = set(existing_df['model_name'].tolist())
                print(f"Found existing CSV with {len(processed_models)} already processed models")
                return processed_models, existing_df
            else:
                print("Found empty CSV file - starting fresh")
                return set(), pd.DataFrame(columns=['model_name', 'type', 'language_tags'])
        else:
            print("No existing CSV found - starting fresh")
            return set(), pd.DataFrame(columns=['model_name', 'type', 'language_tags'])
    except Exception as e:
        print(f"Error reading existing CSV: {e} - starting fresh")
        return set(), pd.DataFrame(columns=['model_name', 'type', 'language_tags'])

def get_unprocessed_models(all_models, already_processed, model_type):
    """Get list of models that haven't been processed yet"""
    unprocessed = [model for model in all_models if model not in already_processed]
    skipped_count = len(all_models) - len(unprocessed)
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} already processed {model_type} models")
    print(f"Found {len(unprocessed)} unprocessed {model_type} models")
    
    return unprocessed

def load_model_names_from_file(file_path):
    """Load model names from a text file"""
    model_names = []
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                model_names = [line.strip() for line in file if line.strip()]
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return model_names

def process_single_model(model_name, model_type, api, request_times, lock, delay_between_requests):
    """Process a single model with rate limiting"""
    with lock:
        # Rate limiting
        current_time = time.time()
        if not request_times.empty():
            last_request = request_times.get()
            time_diff = current_time - last_request
            if time_diff < delay_between_requests:
                sleep_time = delay_between_requests - time_diff
                time.sleep(sleep_time)
        request_times.put(time.time())
    
    try:
        model_info = api.model_info(model_name)
        
        if hasattr(model_info, 'tags') and model_info.tags:
            language_tags = get_language_tags(model_info.tags)
        else:
            language_tags = []
            
        return {
            'model_name': model_name,
            'type': model_type,
            'language_tags': language_tags
        }
        
    except HTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 429:  # Rate limit
            logging.info(f"Rate limit hit for {model_name}")
            time.sleep(delay_between_requests * 2)
        elif hasattr(e, 'response') and e.response.status_code == 404:
            logging.info(f"Model not found: {model_name}")
        else:
            logging.info(f"HTTP Error for {model_name}: {e}")
        return None
        
    except Exception as e:
        logging.info(f"Error fetching {model_name}: {e}")
        return None

def process_models_in_batches(model_names, model_type, api, max_workers=5, delay_between_requests=0.1, batch_size=100):
    """
    Process models in batches with progress tracking and incremental saving
    """
    if not model_names:
        print(f"No {model_type} models to process")
        return []
    
    # Thread-safe queue for rate limiting
    request_times = Queue()
    lock = threading.Lock()
    
    # Split into batches
    batches = [model_names[i:i + batch_size] for i in range(0, len(model_names), batch_size)]
    total_processed = 0
    all_results = []
    
    print(f"Processing {len(model_names)} {model_type} models in {len(batches)} batches with {max_workers} workers...")
    
    for batch_num, batch_models in enumerate(batches, 1):
        print(f"\nProcessing batch {batch_num}/{len(batches)} ({len(batch_models)} models)")
        
        batch_results = []
        
        # Process this batch with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks for this batch
            future_to_model = {
                executor.submit(process_single_model, model, model_type, api, request_times, lock, delay_between_requests): model 
                for model in batch_models
            }
            
            # Process completed futures with progress bar
            with tqdm(total=len(batch_models), desc=f"Batch {batch_num}", unit="models") as pbar:
                for future in as_completed(future_to_model):
                    result = future.result()
                    if result is not None:
                        batch_results.append(result)
                    pbar.update(1)
        
        # Save batch results immediately
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            df_batch.to_csv('model_lang_tags.csv', mode='a', header=False, index=False)
            all_results.extend(batch_results)
            total_processed += len(batch_results)
            print(f"Saved {len(batch_results)} results from batch {batch_num}")
        
        print(f"Batch {batch_num} complete: {len(batch_results)}/{len(batch_models)} models processed successfully")
        print(f"Total progress: {total_processed}/{len(model_names)} models processed")
        
        # Small delay between batches
        if batch_num < len(batches):
            time.sleep(1)
    
    print(f"Finished processing {model_type} models: {total_processed}/{len(model_names)} successful")
    return all_results

def initialize_csv_file(csv_path='model_lang_tags.csv'):
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=['model_name', 'type', 'language_tags'])
        df.to_csv(csv_path, index=False)
        print("Created new CSV file with headers")
        return True
    return False

def main():
    """Main function to process both autoregressive and bidirectional models"""
    
    print("Starting model processing script...")
    print("=" * 60)
    
    # Load existing results and get already processed models
    processed_models, existing_df = load_existing_results('model_lang_tags.csv')
    
    # Initialize CSV file if needed
    csv_is_new = initialize_csv_file('model_lang_tags.csv')
    
    if not csv_is_new and not existing_df.empty:
        print("Resuming from existing CSV file")
        if 'type' in existing_df.columns:
            type_counts = existing_df['type'].value_counts()
            for model_type, count in type_counts.items():
                print(f"  - {model_type}: {count} models")
    
    # Initialize API
    try:
        api = HfApi()
        print("HuggingFace API initialized successfully")
    except Exception as e:
        print(f"Error initializing HuggingFace API: {e}")
        return
    
    # Configuration
    max_workers = 5
    delay_between_requests = 0.1  # 100ms between requests
    batch_size = 100
    
    total_start_time = time.time()
    initial_count = len(existing_df)
    
    # Process autoregressive models
    print("\n" + "=" * 60)
    print("PROCESSING AUTOREGRESSIVE MODELS")
    print("=" * 60)
    
    autoregressive_models = load_model_names_from_file('autoregressive_models.txt')
    
    if autoregressive_models:
        print(f"Loaded {len(autoregressive_models)} autoregressive models from file")
        unprocessed_auto = get_unprocessed_models(autoregressive_models, processed_models, 'autoregressive')
        
        if unprocessed_auto:
            auto_results = process_models_in_batches(
                unprocessed_auto, 
                'autoregressive', 
                api, 
                max_workers, 
                delay_between_requests,
                batch_size
            )
            print(f"Completed autoregressive models: {len(auto_results)} newly processed")
        else:
            print("All autoregressive models have already been processed!")
    else:
        print("No autoregressive models found in file")
    
    # Process bidirectional models
    print("\n" + "=" * 60)
    print("PROCESSING BIDIRECTIONAL MODELS")
    print("=" * 60)
    
    bidirectional_models = load_model_names_from_file('bidirectional_models.txt')
    
    if bidirectional_models:
        print(f"Loaded {len(bidirectional_models)} bidirectional models from file")
        unprocessed_bid = get_unprocessed_models(bidirectional_models, processed_models, 'bidirectional')
        
        if unprocessed_bid:
            bid_results = process_models_in_batches(
                unprocessed_bid, 
                'bidirectional', 
                api, 
                max_workers, 
                delay_between_requests,
                batch_size
            )
            print(f"Completed bidirectional models: {len(bid_results)} newly processed")
        else:
            print("All bidirectional models have already been processed!")
    else:
        print("No bidirectional models found in file")
    
    # Final summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Total processing time: {total_time:.2f} seconds")
    
    try:
        final_df = pd.read_csv('model_lang_tags.csv')
        print(f"Final CSV contains {len(final_df)} total records")
        
        if 'type' in final_df.columns and len(final_df) > 0:
            type_counts = final_df['type'].value_counts()
            for model_type, count in type_counts.items():
                print(f"  - {model_type}: {count} models")
        
        newly_processed = len(final_df) - initial_count
        if newly_processed > 0:
            print(f"Newly processed in this run: {newly_processed} models")
        else:
            print("No new models were processed in this run")
            
    except Exception as e:
        print(f"Could not read final CSV for summary: {e}")
    
    print(f"Results saved to 'model_lang_tags.csv'")

if __name__ == "__main__":
    main()


results = pd.DataFrame(columns = ['language', 'n_models', 'n_multilingual', 'mean_n_langs', 'n_monolingual', 'n_goldfish', 'n_english'])
results.to_csv('by_language_stats.csv', mode='w')

import ast

for lang in goldfish_langs:
    total = 0
    monolingual = 0
    multilingual = 0
    eng_models = 0
    goldfish = 0
    n_langs = []

    for i in range(len(model_lang_tags)):
        language_tags = model_lang_tags.iloc[i]['language_tags']
        model_name = model_lang_tags.iloc[i]['model_name']
        language_tags = ast.literal_eval(language_tags)
        
        if lang in language_tags: 
            total += 1
            
            if 'goldfish-models/' in model_name:
                goldfish += 1

            n_langs.append(len(language_tags))
                
            if len(language_tags) == 1:
                monolingual += 1
            elif len(language_tags) > 1:
                multilingual += 1
            if 'eng' in language_tags:
                eng_models += 1
            
    new_line = pd.DataFrame({'language': [lang], 
                         'n_models': [total], 
                         'n_multilingual': [multilingual],
                         'mean_n_langs': [np.mean(n_langs)],
                         'n_monolingual': [monolingual], 
                         'n_goldfish': [goldfish],
                         'n_english': [eng_models]})
    new_line.to_csv('by_language_stats.csv', mode='a', header=False)


results = pd.DataFrame(columns = ['language', 'architecture', 'n_models', 'n_multilingual', 'mean_n_langs', 'n_monolingual', 'n_goldfish', 'n_english'])
results.to_csv('by_language_by_arch_stats.csv', mode='w')

import ast

for lang in goldfish_langs:
    total = 0
    total_bidi = 0
    total_auto = 0
    monolingual_bidi = 0
    monolingual_auto = 0
    multilingual_bidi = 0
    multilingual_auto = 0
    eng_models_bidi = 0
    eng_models_auto = 0
    goldfish = 0
    n_langs_bidi = []
    n_langs_auto = []

    for i in range(len(model_lang_tags)):
        arch = model_lang_tags.iloc[i]['type']
        language_tags = model_lang_tags.iloc[i]['language_tags']
        model_name = model_lang_tags.iloc[i]['model_name']
        language_tags = ast.literal_eval(language_tags)
        
        if lang in language_tags: 
            total += 1
            
            if 'goldfish-models/' in model_name:
                goldfish += 1
                
            if arch == 'autoregressive':
                total_auto += 1

                n_langs_auto.append(len(language_tags))
                    
                if len(language_tags) == 1:
                    monolingual_auto += 1
                elif len(language_tags) > 1:
                    multilingual_auto += 1
                if 'eng' in language_tags:
                    eng_models_auto += 1

            elif arch == 'bidirectional':
                total_bidi += 1

                n_langs_bidi.append(len(language_tags))
                    
                if len(language_tags) == 1:
                    monolingual_bidi += 1
                elif len(language_tags) > 1:
                    multilingual_bidi += 1
                if 'eng' in language_tags:
                    eng_models_bidi += 1

            
    new_line = pd.DataFrame({'language': [lang], 
                             'architecture': ['autoregressive'],
                         'overall_n_models': [total],
                         'n_models': [total_auto], 
                         'n_multilingual': [multilingual_auto],
                         'mean_n_langs': [np.mean(n_langs_auto)],
                         'n_monolingual': [monolingual_auto], 
                         'n_goldfish': [goldfish],
                         'n_english': [eng_models_auto]})
    new_line.to_csv('by_language_by_arch_stats.csv', mode='a', header=False)

    new_line = pd.DataFrame({'language': [lang], 
                             'architecture': ['bidirectional'],
                         'overall_n_models': [total],
                         'n_models': [total_bidi], 
                         'n_multilingual': [multilingual_bidi],
                         'mean_n_langs': [np.mean(n_langs_bidi)],
                         'n_monolingual': [monolingual_bidi], 
                         'n_goldfish': [0],
                         'n_english': [eng_models_bidi]})
    new_line.to_csv('by_language_by_arch_stats.csv', mode='a', header=False)
