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

def training_features():
    features = []

    features.append(PairFeature('relation',
                                lambda pairs, index: pairs[index].relation))

    features.append(PairFeature('entity1_text',
                                lambda pairs, index: pairs[index].entity1.text))

    features.append(PairFeature('entity2_text',
                                lambda pairs, index: pairs[index].entity2.text))

    features.append(PairFeature('entity1_POS',
                                lambda pairs, index: pairs[index].entity1.pos))

    features.append(PairFeature('entity2_POS',
                                lambda pairs, index: pairs[index].entity1.pos))

    return features

def write_feature_file(pairs, features, output_filename):
    with open(output_filename, 'w') as output_file:
        for i in range(len(pairs)):
            output_file.write(format_pair_features(pairs, i, features) + '\n')

def format_pair_features(pairs, index, features):
    return ' '.join(['{}={}'.format(feat.name, feat.extract(pairs, index)) for feat in features])

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

if __name__ == '__main__':
    training_features = training_features()
    testing_features = training_features[1:]

    pos_dict = get_pos_from_all()

    train = get_pairs(TRAIN_GOLD_PATH, pos_dict)
    write_feature_file(train, training_features, TRAIN_FEATURE_PATH)

    dev = get_pairs(DEV_GOLD_PATH, pos_dict)
    write_feature_file(dev, testing_features, DEV_FEATURE_PATH)


    # test = get_pairs(TEST_GOLD_PATH, pos_dict)