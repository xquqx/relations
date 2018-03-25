from model_params import *
from entity_pair import *
import os
import re

FEATURE_SEP = ' '
POS_REGEX = re.compile('(?<=\S)_(?=\S)')

class PairFeature:
    def __init__(self, name, extraction_func):
        self.name = name
        self.extraction_func = extraction_func

    def __str__(self):
        return '<{}: {}>'.format(self.__class__.__name__, self.name)

    def extract(self, pairs, index):
        return self.extraction_func(pairs, index)

def features():
    features = [('entity1_text', lambda p, i: p[i].entity1.text),
                ('entity2_text', lambda p, i: p[i].entity2.text),
                ('entity1_POS', lambda p, i: p[i].entity1.pos),
                ('entity2_POS', lambda p, i: p[i].entity2.pos),
                ('entity1_type', lambda p, i: p[i].entity1.entity_type),
                ('entity2_type', lambda p, i: p[i].entity2.entity_type),
                ('entity1_type', lambda p, i: p[i].entity1.entity_type),
                ('entity2_type', lambda p, i: p[i].entity2.entity_type),
                ('distance', lambda p, i: p[i].entity2.start_index - p[i].entity1.end_index)]

    features = [PairFeature(name, func) for name, func in features]
    return features

def format_train_features(pairs, features):
    for i in range(len(pairs)):
        yield FEATURE_SEP.join([pairs[i].relation, format_pair_features(pairs, i, features)])

def format_test_features(pairs, features):
    for i in range(len(pairs)):
        yield format_pair_features(pairs, i, features)

def output_rows(rows, output_filename):
    with open(output_filename, 'w') as output_file:
        for r in rows:
            output_file.write(r + '\n')

def format_pair_features(pairs, index, features):
    return FEATURE_SEP.join(['{}={}'.format(feat.name, feat.extract(pairs, index)) for feat in features])

def get_pairs(filename, pos_dict=None):
    pairs = EntityPair.list_from_filename(filename)
    if pos_dict:
        for p in pairs:
            doc = pos_dict[p.doc_id]
            set_pos_tag(p.entity1, doc)
            set_pos_tag(p.entity2, doc)
            set_prev_pos_tag(p.entity1, doc)
            set_prev_pos_tag(p.entity2, doc)
    return pairs


def set_pos_tag(entity, doc):
    word, tag = doc[entity.sent_index][entity.start_index]
    entity.pos = tag

def set_prev_pos_tag(entity, doc):
    if entity.start_index > 0 and len(doc[entity.sent_index][entity.start_index-1]) != 2: print(doc[entity.sent_index][entity.start_index-1])
    word, tag = doc[entity.sent_index][entity.start_index-1] if entity.start_index > 0 else None, None
    entity.prev_pos = tag

def verify_vocab_match(entity, word):
    pos_word = entity.text.split('_')[0]
    if pos_word != word:
        print('Failed vocab match. '
              'From POS files: "{}". '
              'From relations files: "{}"'.format(pos_word, word))

def get_pos_from_filename(filename):
    sents = []
    with open(filename) as input_file:
        for line in input_file:
            if line.strip() != '':
                sents.append([POS_REGEX.split(x) for x in line.split()])
    return sents

def get_pos_from_all():
    docs = dict()
    for filename in os.listdir(POS_DIR):
        doc_id = filename.strip(POS_SUFFIX)
        docs[doc_id] = get_pos_from_filename(join(POS_DIR, filename))
    return docs

if __name__ == '__main__':
    features = features()

    pos_dict = get_pos_from_all()

    train = get_pairs(TRAIN_GOLD_PATH, pos_dict)
    output_rows(format_train_features(train, features), TRAIN_FEATURE_PATH)

    dev = get_pairs(DEV_GOLD_PATH, pos_dict)
    output_rows(format_test_features(dev, features), DEV_FEATURE_PATH)

    test = get_pairs(TEST_GOLD_PATH, pos_dict)
    output_rows(format_test_features(test, features), TEST_FEATURE_PATH)