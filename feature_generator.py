from model_params import *
from entity_pair import *
import os

class PairFeature:
    def __init__(self, name, extraction_func):
        self.name = name
        self.extraction_func = extraction_func

    def __str__(self):
        return '<{}: {}>'.format(self.__class__.__name__, self.name)

    def extract(self, pairs, index):
        return self.extraction_func(pairs, index)


def get_pairs(filename, pos_dict=None):
    pairs = EntityPair.list_from_filename(filename)
    if pos_dict:
        for p in pairs:
            doc = pos_dict[p.doc_id]
            set_pos_tag(p.entity1, doc)
            set_pos_tag(p.entity2, doc)
    return pairs


def set_pos_tag(entity, doc):
    word, tag = doc[entity.sent_index][entity.start_index]
    # verify_vocab_match(entity, word)
    entity.pos = tag

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
                sents.append([x.split('_') for x in line.split()])
    return sents

def get_pos_from_all():
    docs = dict()
    for filename in os.listdir(POS_DIR):
        doc_id = filename.strip(POS_SUFFIX)
        docs[doc_id] = get_pos_from_filename(join(POS_DIR, filename))
    return docs

# def get_features(pairs, index):
#

if __name__ == '__main__':
    pos_dict = get_pos_from_all()
    train = get_pairs(TRAIN_GOLD_PATH, pos_dict)
    dev = get_pairs(DEV_GOLD_PATH, pos_dict)
    test = get_pairs(TEST_GOLD_PATH, pos_dict)
