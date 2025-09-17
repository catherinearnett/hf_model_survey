import pandas as pd
import time
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
import numpy as np
import matplotlib.pyplot as plt


model_lang_tags = pd.read_csv('data/model_lang_tags.csv')


#general stats
count= 0
for i in range(len(model_lang_tags)):
    if model_lang_tags.iloc[i]['language_tags'] == '[]':
        count += 1

print(count, " models out of ", len(model_lang_tags), " do not have any language labels.")
print(len(model_lang_tags)-count, ' do have labels.')


print("Number of generation models: ", len(model_lang_tags[model_lang_tags['type'] == 'autoregressive']))

print("Number of classification models: ", len(model_lang_tags[model_lang_tags['type'] == 'bidirectional']))

print("Number of generation models without language tags: ", len(model_lang_tags[(model_lang_tags['type'] == 'autoregressive') & (model_lang_tags['language_tags'] == '[]')]))

print("Number of classification models without language tags: ", len(model_lang_tags[(model_lang_tags['type'] == 'bidirectional') & (model_lang_tags['language_tags'] == '[]')]))



by_lang_stats = pd.read_csv('data/by_language_stats.csv', mode='a', header=False)

goldfish_lang_stats = by_lang_stats[by_lang_stats['n_goldfish'] > 0]


n_model_dist = goldfish_lang_stats.sort_values(by="n_models", ascending=False).reset_index(drop=True)

plt.figure(figsize=(15, 6))

#Distribution of Models by Language (Top 25)
plt.bar(n_model_dist["language"].astype(str), n_model_dist["n_models"])
plt.xlabel("Language")
plt.ylabel("Number of Models")
plt.xlim(-0.5, 25.5)
plt.tight_layout()
plt.savefig("lang_dist_top_25.png", bbox_inches='tight')
plt.show()

n_model_dist = goldfish_lang_stats.sort_values(by="n_models", ascending=False)

plt.figure(figsize=(75, 20))

#Distribution of All Models by Language (Excluding English)
plt.bar(n_model_dist["language"].astype(str), n_model_dist["n_models"])
plt.xlabel("Language")
plt.ylabel("Number of Models")
plt.xlim(0.5, 100.5)
plt.ylim(0, 7000)

plt.tight_layout()
plt.savefig("lang_dist_top_all.png", bbox_inches='tight')
plt.show()


# Number of models with English as one of the language tags
eng_model = goldfish_lang_stats[goldfish_lang_stats['language'] == 'eng']['n_models'].item()
print(eng_model)

print(eng_model/85916)

# Number of models with Chinese as one of the language tags
zho_model = goldfish_lang_stats[goldfish_lang_stats['language'] == 'zho']['n_models'].item()
print(zho_model)

print(zho_model/eng_model)

# For how many languages is it the case that goldfish is the only monolingual model?
count = 0
for i in range(len(goldfish_lang_stats)):
    if goldfish_lang_stats.iloc[i]['n_monolingual'] == goldfish_lang_stats.iloc[i]['n_goldfish']:
        count += 1

print(count)

# For how many languages is it the case that goldfish is the only model?

count = 0
for i in range(len(goldfish_lang_stats)):
    if goldfish_lang_stats.iloc[i]['n_models'] <= goldfish_lang_stats.iloc[i]['n_goldfish']:
        count += 1
        # print(goldfish_lang_stats.iloc[i]['language'])

print(count)

# Proportion of languages for which all multilingual models are also trained on English data?
count = 0
for i in range(len(goldfish_lang_stats)):
    if goldfish_lang_stats.iloc[i]['n_multilingual'] == goldfish_lang_stats.iloc[i]['n_english']:
        count += 1

print(count)

# Mean porportion of models trained on English by language
proportion_eng = []
for i in range(len(goldfish_lang_stats)):
    n_eng = goldfish_lang_stats.iloc[i]['n_english']
    n_model = goldfish_lang_stats.iloc[i]['n_models']
    prop = n_eng/n_model
    proportion_eng.append(prop)

goldfish_lang_stats['prop_eng'] = proportion_eng

n_model_dist = goldfish_lang_stats.sort_values(by="prop_eng", ascending=True)

plt.figure(figsize=(15, 6))


plt.bar(n_model_dist["language"].astype(str), n_model_dist["prop_eng"])
plt.xlabel("Language")
plt.ylabel("Proportion of English In Models")
plt.xlim(-0.5, 325.5)
plt.tight_layout()
plt.savefig("prop_eng_all.png", bbox_inches='tight')
plt.show()

n_model_dist = goldfish_lang_stats.sort_values(by="prop_eng", ascending=True)

plt.figure(figsize=(15, 6))


plt.bar(n_model_dist["language"].astype(str), n_model_dist["prop_eng"])
plt.xlabel("Language")
plt.ylabel("Proportion of English In Models (Top 25)")
plt.xlim(-0.5, 25.5)
plt.ylim(0, 0.3)
plt.tight_layout()
plt.savefig("prop_eng_bottom_25.png", bbox_inches='tight')
plt.show()


# What is the average proportion of monolingual models available?
props = []
for i in range(len(goldfish_lang_stats)):
    n_mono = goldfish_lang_stats.iloc[i]['n_monolingual'].item()
    n_m = goldfish_lang_stats.iloc[i]['n_models'].item()
    prop = n_mono/n_m
    props.append(prop)

print(np.mean(props))

# Mean Proportion of Multilingual Models
props = []
for i in range(len(goldfish_lang_stats)):
    n_mono = goldfish_lang_stats.iloc[i]['n_multilingual'].item()
    n_m = goldfish_lang_stats.iloc[i]['n_models'].item()
    prop = n_mono/n_m
    props.append(prop)

print(np.mean(props))

# Mean proportion of English models

props = []
for i in range(len(goldfish_lang_stats)):
    n_mono = goldfish_lang_stats.iloc[i]['n_english'].item()
    n_m = goldfish_lang_stats.iloc[i]['n_models'].item()
    prop = n_mono/n_m
    props.append(prop)
    

print(np.mean(props))

# What is the distribution of average number of languages covered per model?
n_model_dist = goldfish_lang_stats.sort_values(by="mean_n_langs", ascending=True)

plt.figure(figsize=(15, 4))

plt.bar(n_model_dist["language"].astype(str), n_model_dist["mean_n_langs"])
plt.xlabel("Language")
plt.ylabel("Mean Number of Languages per Model")
plt.xlim(-0.5, 20.5)
plt.ylim(0, 30)
plt.tight_layout()
plt.savefig("mean_n_langs_dist_top_20.png", bbox_inches='tight')
plt.show()

n_model_dist = goldfish_lang_stats.sort_values(by="mean_n_langs", ascending=True)

plt.figure(figsize=(15, 4))

plt.bar(n_model_dist["language"].astype(str), n_model_dist["mean_n_langs"])
plt.xlabel("Language")
plt.ylabel("Mean Number of Languages per Model")
plt.xlim(1.5, 20.5)
plt.ylim(0, 30)
plt.tight_layout()
plt.savefig("mean_n_langs_dist_top_20_cropped.png", bbox_inches='tight')
plt.show()

## Mean number of langugages per model tagged with English

n_model_dist[n_model_dist['language'] == 'eng']['mean_n_langs'].item()


# distribution for all languages
n_model_dist = goldfish_lang_stats.sort_values(by="mean_n_langs", ascending=True)

plt.figure(figsize=(50, 5))

plt.bar(n_model_dist["language"].astype(str), n_model_dist["mean_n_langs"])
plt.xlabel("Language")
plt.ylabel("Mean Number of Languages per Model")
plt.xlim(-0.5, 325.5)
plt.tight_layout()
plt.savefig("mean_n_langs_dist_all.png", bbox_inches='tight')
plt.show()

# Proportion monolingual models

proportion_monolingual = []
for i in range(len(goldfish_lang_stats)):
    n_mono = goldfish_lang_stats.iloc[i]['n_monolingual']
    n_model = goldfish_lang_stats.iloc[i]['n_models']
    prop = n_mono/n_model
    proportion_monolingual.append(prop)

goldfish_lang_stats['prop_monolingual'] = proportion_monolingual

n_model_dist = goldfish_lang_stats.sort_values(by="prop_monolingual", ascending=False)

plt.figure(figsize=(15, 6))


plt.bar(n_model_dist["language"].astype(str), n_model_dist["prop_monolingual"])
plt.xlabel("Language")
plt.ylabel("Proportion of Monolingual Models")
plt.xlim(-0.5, 25.5)
plt.tight_layout()
plt.savefig("prop_mono_top_25.png", bbox_inches='tight')
plt.show()

n_model_dist = goldfish_lang_stats.sort_values(by="prop_monolingual", ascending=False).reset_index(drop=True)

plt.figure(figsize=(15, 6))


plt.bar(n_model_dist["language"].astype(str), n_model_dist["prop_monolingual"])
plt.xlabel("Language")
plt.ylabel("Proportion of Monolingual Models")
plt.xlim(-0.5, 325.5)
plt.tight_layout()
plt.savefig("prop_mono_all.png", bbox_inches='tight')
plt.show()

# number of languages have a minority monolingual models
len(n_model_dist[n_model_dist['prop_monolingual'] < 0.5]) 

# Average increase in number of models
non_goldfish_models = []
for i in range(len(goldfish_lang_stats)):
    n_gold = goldfish_lang_stats.iloc[i]['n_goldfish']
    n_model = goldfish_lang_stats.iloc[i]['n_models']
    prop = n_model - n_gold
    non_goldfish_models.append(prop)

goldfish_lang_stats['non_goldfish'] = non_goldfish_models

goldfish_increase = []
for i in range(len(goldfish_lang_stats)):
    non_gold = goldfish_lang_stats.iloc[i]['non_goldfish']
    n_model = goldfish_lang_stats.iloc[i]['n_models']
    prop = (n_model/non_gold) - 1
    goldfish_increase.append(prop)

goldfish_lang_stats['goldfish_increase'] = goldfish_increase

valid_vals = goldfish_lang_stats['goldfish_increase'][~np.isinf(goldfish_lang_stats['goldfish_increase'])]
mean_val = np.mean(valid_vals)
mean_val