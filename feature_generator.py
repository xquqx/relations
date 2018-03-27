from model_params import *
from entity_pair import *
from nltk.stem import WordNetLemmatizer
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
                ('entity1_text_length', lambda p, i: len(p[i].entity1.text)),
                ('entity2_text_length', lambda p, i: len(p[i].entity2.text)),
                ('entity1_num_words', lambda p, i: p[i].entity1.end_index - p[i].entity1.start_index),
                ('entity2_num_words', lambda p, i: p[i].entity2.end_index - p[i].entity2.start_index),
                #('entity1_POS', lambda p, i: p[i].entity1.pos),
                #('entity2_POS', lambda p, i: p[i].entity2.pos),
                #('entity1_type', lambda p, i: p[i].entity1.entity_type),
                #('entity2_type', lambda p, i: p[i].entity2.entity_type),
                #('entity1_type2', lambda p, i: p[i].entity1.entity_type),
                #('entity2_type2', lambda p, i: p[i].entity2.entity_type),
                #('entity_types_pair', lambda p, i: p[i].entity1.entity_type+'-'+p[i].entity2.entity_type),
                #('entity_pos_pair', lambda p, i: p[i].entity1.pos+'-'+p[i].entity2.pos),
                #('distance', lambda p, i: p[i].entity2.start_index - p[i].entity1.end_index),
                #('distance2', lambda p, i: p[i].entity2.start_index - p[i].entity1.end_index),
                #('entity1_in_entity2', lambda p, i: p[i].entity1.text in p[i].entity2.text),
                #('entity2_in_entity1', lambda p, i: p[i].entity2.text in p[i].entity1.text),
                #('shared_words', lambda p, i: '_'.join(word for word in p[i].entity1.text.split('_') if word in p[i].entity2.text.split('_'))),
                #('num_shared_words', lambda p, i: len([word for word in p[i].entity1.text.split('_') if word in p[i].entity2.text.split('_')])),
                #('entities_between_pair', lambda p, i: p[i].entity_distance),
                #('e1_is_country', lambda p, i: p[i].entity2.entity_type if p[i].entity1.is_country else ''),
                #('e2_is_country', lambda p, i: p[i].entity1.entity_type if p[i].entity2.is_country else ''),
                #('between_bow', lambda p, i: p[i].between_entities_bow),
                ]

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

def get_pairs(filename, pos_dict=None, vocab=None, doc_vocab =None, country_list=None, entity_dict=None):
    pairs = EntityPair.list_from_filename(filename)
    if pos_dict:
        for p in pairs:
            doc = pos_dict[p.doc_id]
            set_pos_tag(p.entity1, doc)
            set_pos_tag(p.entity2, doc)
            set_prev_pos_tag(p.entity1, doc)
            set_prev_pos_tag(p.entity2, doc)
    if vocab and doc_vocab:
        print('vocab and doc exist for '+filename)
        for p in pairs:
            set_between_BoW(p, get_bow_vector(p,vocab,doc_vocab))
            if p.between_entities_bow == None:
                print('\tfailed.')
    if entity_dict:
        for p in pairs:
            set_entity_distance(p,entity_dict)
    if country_list:
        for p in pairs:
            set_is_country(p.entity1, country_list)
            set_is_country(p.entity2, country_list)
    return pairs

def get_bow_vector(p, vocab, doc_vocab):
    vector = [0] * len(vocab)
    sent_counter = p.entity1.sent_index
    final_start_index = p.entity1.end_index
    while (sent_counter != p.entity2.sent_index):
        final_start_index = 0
        if sent_counter == p.entity1.sent_index:
            word_counter = p.entity1.end_index
        else:
            word_counter = 0
        try:
            word = doc_vocab[p.doc_id][sent_counter][word_counter]
            try:
                index = vocab.index(word)
                vector[index] = 1
            except:
                pass
        except:
            sent_counter += 1
        word_counter += 1
        
    for i in range(final_start_index, p.entity2.start_index):
        try:
            vector[vocab.index(doc_vocab[p.doc_id][p.entity2.sent_index][i])] = 1
        except:
            pass
    return vector

def set_between_BoW(p, v):
    v_string = ''
    for i in v:
        v_string += str(i)
    p.between_entities_bow = v_string

def set_is_country(entity, country_list):
    if entity.text in country_list:
        entity.is_country = True

def set_entity_distance(pair, entities):
    try:
        pair.entity_distance = entities[pair.doc_id].index((pair.entity2.sent_index,pair.entity2.start_index)) - entities[pair.doc_id].index((pair.entity1.sent_index,pair.entity1.start_index)) - 1
    except:
        pair.entity_distance = 0

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
    vocab = {}
    doc_vocab = {}
    for filename in os.listdir(POS_DIR):
        doc_id = filename.strip(POS_SUFFIX)
        docs[doc_id] = get_pos_from_filename(join(POS_DIR, filename))
        vocab = get_vocab_from_posfile(join(POS_DIR, filename),vocab)
        doc_vocab[doc_id] = get_ordered_word_list(join(POS_DIR, filename))
    
    counter = 1
    length = len(vocab)
    shorter_vocab = vocab.keys()
    while length > 500:
        shorter_vocab = [v for v in vocab.keys() if vocab[v] > counter]
        print(str(counter)+": "+str(len(shorter_vocab)))
        counter += 1
        length = len(shorter_vocab)
    return docs,shorter_vocab,doc_vocab

def get_ordered_word_list(filename):
    wnl = WordNetLemmatizer()
    file_tokens = {}
    with open(filename,encoding='utf8') as input_file:
        sent_count = 0
        for line in input_file:
            word_count = 0
            file_tokens[sent_count] = {}
            if line.strip('\n') != '':
                for w_p in line.strip('\n').split(' '):
                    token = wnl.lemmatize(w_p.split('_')[0].lower().strip('`'))
                    file_tokens[sent_count][word_count] = token
                    word_count += 1
                sent_count += 1
    return file_tokens

def get_vocab_from_posfile(filename,vocab):
    wnl = WordNetLemmatizer()
    with open(filename,encoding='utf8') as input_file:
        for line in input_file:
            if line.strip('\n') != '':
                for w_p in line.strip('\n').split(' '):
                    token = wnl.lemmatize(w_p.split('_')[0].lower().strip('`'))
                    try:
                        vocab[token] += 1
                    except:
                        vocab[token] = 1
    return vocab

def get_ordered_entity_list(filename):
    entities = {}
    with open(filename,encoding='utf8') as input_file:
        for line in input_file:
            if filename.endswith('.raw'):
                try:
                    if (int(line.split()[1]),int(line.split()[2])) not in entities[line.split()[0]]:
                        entities[line.split()[0]].append((int(line.split()[1]),int(line.split()[2])))
                except:
                    entities[line.split()[0]] = [(int(line.split()[1]),int(line.split()[2]))]
            elif filename.endswith('.gold'):
                try:
                    if (int(line.split()[2]),int(line.split()[3])) not in entities[line.split()[1]]:
                        entities[line.split()[1]].append((int(line.split()[2]),int(line.split()[3])))
                except:
                    entities[line.split()[1]] = [(int(line.split()[2]),int(line.split()[3]))]

    return entities

def load_countries():
    countries = []
    with open('country_list.txt',encoding='utf8') as country_list:
        for line in country_list:
            countries.append(line.strip('\n'))
    return countries

if __name__ == '__main__':
    features = features()

    pos_dict,vocab,doc_vocab = get_pos_from_all()
    #country_list = load_countries()

    #entity_dict = get_ordered_entity_list(TRAIN_GOLD_PATH)
    train = get_pairs(TRAIN_GOLD_PATH, pos_dict,vocab,doc_vocab)
    output_rows(format_train_features(train, features), TRAIN_FEATURE_PATH)

    #entity_dict = get_ordered_entity_list(DEV_GOLD_PATH)
    dev = get_pairs(DEV_GOLD_PATH, pos_dict,vocab,doc_vocab)
    output_rows(format_test_features(dev, features), DEV_FEATURE_PATH)

    #entity_dict = get_ordered_entity_list(TEST_GOLD_PATH)
    test = get_pairs(TEST_GOLD_PATH, pos_dict,vocab,doc_vocab)
    output_rows(format_test_features(test, features), TEST_FEATURE_PATH)