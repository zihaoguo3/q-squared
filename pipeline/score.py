# Copyright 2020 The Q2 Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
from collections import Counter

from bert_score import score


def clean_text(text):#remove space and punctuation from the text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the|in|our)\b', ' ', text)
    return re.sub(' +', ' ', text).strip()


def f1_score(a_gold, a_pred):
    if a_pred == '':
        return 0
    gold_toks = clean_text(a_gold).split() #tokenize
    pred_toks = clean_text(a_pred).split() #tokenize
    common = Counter(gold_toks) & Counter(pred_toks) #overlap token
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks) #overlap token / total token in prediction
    recall = 1.0 * num_same / len(gold_toks) #overlap token / total token in gold
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_bert_score(a_gold, a_pred): #use score from bert_score
    P, R, F1 = score(a_pred, a_gold, lang="en", verbose=True)
    return F1.mean().item()
# will consider synonyms for similar words
# having embeebeded bert_score will give better result than f1_score