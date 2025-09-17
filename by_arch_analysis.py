import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data/by_language_by_arch_stats.csv')

df.columns = ['language', 'architecture', 'overall_n_models', 'n_models', 'n_multilingual', 'mean_n_langs', 'n_monolingual', 'n_goldfish', 'n_english']

auto = df[df['architecture'] == 'autoregressive']
auto = auto[auto['n_goldfish'] > 0]

goldfish_langs = list(auto['language'])

bidi = df[df['architecture'] == 'bidirectional']
bidi = bidi[bidi['language'].isin(goldfish_langs)]


# For how many languages are goldfish the only autoregressive model?
count = 0
langs = ''
for i in range(len(auto)):
    n_models = auto.iloc[i]['n_models']
    n_goldfish = auto.iloc[i]['n_goldfish']
    if n_goldfish >= n_models:
        count += 1
        l = str(auto.iloc[i]['language'])
        langs += l + ', '

print(count, count/len(auto))
print(langs)

# For how many languages are goldfish the only monolingual autoregressive model?
count = 0
for i in range(len(auto)):
    n_mono = auto.iloc[i]['n_monolingual']
    n_goldfish = auto.iloc[i]['n_goldfish']
    if n_goldfish >= n_mono:
        count += 1

print(count, count/len(auto))

# how many languages don't have any bidirectional models?
count = 0
for i in range(len(bidi)):
    n_bidi = bidi.iloc[i]['n_models']
    if n_bidi == 0:
        count += 1

print(count, count/len(bidi))

# how many languages don't have any bidirectional monolingual models?

count = 0
for i in range(len(bidi)):
    n_mono = bidi.iloc[i]['n_monolingual']
    if n_mono == 0:
        count += 1

print(count, count/len(bidi))

# distribution by architecture
auto_dist = auto[['language', 'architecture', 'n_models', 'overall_n_models']]
bidi_dist = bidi[['language', 'architecture', 'n_models', 'overall_n_models']]

dist_data = pd.concat([auto_dist, bidi_dist])

auto_dist = auto_dist.sort_values(by="overall_n_models", ascending=False).reset_index(drop=True)
bidi_dist = bidi_dist.sort_values(by="overall_n_models", ascending=False).reset_index(drop=True)

plt.figure(figsize=(15, 6))

plt.bar(auto_dist["language"].astype(str), auto_dist["n_models"], color = 'orange')
plt.bar(bidi_dist["language"].astype(str), bidi_dist["n_models"], color = 'purple')
plt.xlabel("Language")
plt.ylabel("Number of Models")
plt.xlim(-0.5, 25.5)
plt.tight_layout()
plt.savefig("by_arch_lang_dist_top_25.png", bbox_inches='tight')
plt.show()

auto_dist = auto_dist.sort_values(by="overall_n_models", ascending=False).reset_index(drop=True)
bidi_dist = bidi_dist.sort_values(by="overall_n_models", ascending=False).reset_index(drop=True)

plt.figure(figsize=(15, 6))

plt.bar(bidi_dist["language"].astype(str), bidi_dist["n_models"], color = 'purple')
plt.xlabel("Language")
plt.ylabel("Number of Models")
plt.xlim(-0.5, 25.5)
plt.tight_layout()
plt.savefig("bidi_lang_dist_top_25.png", bbox_inches='tight')
plt.show()

#some data cleaning
auto_dist = auto[['language', 'n_models', 'overall_n_models']]
auto_dist.columns = ['language', 'n_auto_models', 'overall_n_models']
bidi_dist = bidi[['language', 'n_models']]
bidi_dist.columns = ['language', 'n_bidi_models']

dist_data = auto_dist.merge(bidi_dist, how='left', left_on = 'language', right_on = 'language')
dist_data

prop_auto_models = []
prop_bidi_models = []

for i in range(len(dist_data)):
    total = dist_data.iloc[i]['overall_n_models']
    auto_n = dist_data.iloc[i]['n_auto_models']
    bidi_n = dist_data.iloc[i]['n_bidi_models']
    prop_auto_models.append(auto_n/total)
    prop_bidi_models.append(bidi_n/total)

dist_data['prop_auto_models'] = prop_auto_models
dist_data['prop_bidi_models'] = prop_bidi_models

# distribution for English
dist_data[dist_data['language'] == 'eng']

#for top languages

dist_data = dist_data.sort_values(by="prop_auto_models", ascending=False).reset_index(drop=True)

plt.figure(figsize=(25, 6))

plt.bar(dist_data["language"].astype(str), dist_data["prop_auto_models"])
plt.xlabel("Language")
plt.ylabel("Proportion Text Generation Models")
plt.xlim(-0.5, 50.5)
plt.tight_layout()
plt.savefig("autoregressive_lang_dist_top_25.png", bbox_inches='tight')
plt.show()

dist_data = dist_data.sort_values(by="prop_bidi_models", ascending=False).reset_index(drop=True)

plt.figure(figsize=(20, 6))

plt.bar(dist_data["language"].astype(str), dist_data["prop_bidi_models"])
plt.xlabel("Language")
plt.ylabel("Proportion Text Classification Models")
plt.xlim(-0.5, 50.5)
plt.tight_layout()
plt.savefig("bidirectional_lang_dist_top_25.png", bbox_inches='tight')
plt.show()

# For how many languages are all multilingual generation models trained on at least some english?

count = 0
for i in range(len(auto)):
    n_english = auto.iloc[i]['n_english']
    n_multi = auto.iloc[i]['n_multilingual']

    if n_english == n_multi:
        count += 1

print(count)

# For how many languages are all multilingual classification models trained on at least some english?
count = 0
for i in range(len(bidi)):
    n_english = bidi.iloc[i]['n_english']
    n_multi = bidi.iloc[i]['n_multilingual']

    if n_english == n_multi:
        count += 1

print(count)

#prop multilingual
prop_multi_auto_models = []
prop_multi_bidi_models = []

for i in range(len(auto)):
    auto_n = auto.iloc[i]['n_models']
    multi_auto = auto.iloc[i]['n_multilingual']
    bidi_n = bidi.iloc[i]['n_models']
    multi_bidi = bidi.iloc[i]['n_multilingual']

    if auto_n == 0:
        prop_multi_auto_models.append(0)
    else:
        prop_multi_auto_models.append(multi_auto/auto_n)
    if bidi_n == 0:
        prop_multi_bidi_models.append(0)
    else:
        prop_multi_bidi_models.append(multi_bidi/bidi_n)

auto['prop_multi_auto_models'] = prop_multi_auto_models
bidi['prop_multi_bidi_models'] = prop_multi_bidi_models

# What proportion of generation models are multilingual?
print(np.mean(prop_multi_auto_models))

# What proportion of classification models are multilingual?
print(np.mean(prop_multi_bidi_models))

# prop english

prop_english_auto_models = []
prop_english_bidi_models = []

for i in range(len(auto)):
    auto_n = auto.iloc[i]['n_models']
    multi_auto = auto.iloc[i]['n_english']
    bidi_n = bidi.iloc[i]['n_models']
    multi_bidi = bidi.iloc[i]['n_english']

    if auto_n == 0:
        prop_english_auto_models.append(0)
    else:
        prop_english_auto_models.append(multi_auto/auto_n)
    if bidi_n == 0:
        prop_english_bidi_models.append(0)
    else:
        prop_english_bidi_models.append(multi_bidi/bidi_n)

auto['prop_english_auto_models'] = prop_english_auto_models
bidi['prop_english_bidi_models'] = prop_english_bidi_models

# What proportion of generation models are trained on English?
print(np.mean(prop_english_auto_models))

# What proportion of classification models are trained on English?
np.mean(prop_english_bidi_models)


## PROP MONOLINGUAL
prop_mono_auto_models = []
prop_mono_bidi_models = []

for i in range(len(auto)):
    auto_n = auto.iloc[i]['n_models']
    multi_auto = auto.iloc[i]['n_monolingual']
    bidi_n = bidi.iloc[i]['n_models']
    multi_bidi = bidi.iloc[i]['n_monolingual']

    if auto_n == 0:
        prop_mono_auto_models.append(0)
    else:
        prop_mono_auto_models.append(multi_auto/auto_n)
    if bidi_n == 0:
        prop_mono_bidi_models.append(0)
    else:
        prop_mono_bidi_models.append(multi_bidi/bidi_n)

auto['prop_mono_auto_models'] = prop_mono_auto_models
bidi['prop_mono_bidi_models'] = prop_mono_bidi_models

#proportion of monolingual generation models
print(len(auto[auto['prop_mono_auto_models'] < 0.5])/325)

#proportion of monolingual classification models
print(len(bidi[bidi['prop_mono_bidi_models'] < 0.5])/325)

