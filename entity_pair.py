from model_params import *

class Entity:
    def __init__(self,
                 sent_index,
                 start_index,
                 end_index,
                 entity_type,
                 relation_id,
                 text,
                 pos=None,
                 prev_pos=None):
        self.sent_index = sent_index
        self.start_index = start_index
        self.end_index = end_index
        self.entity_type = entity_type
        self.text = text
        self.pos = pos
        self.prev_pos = prev_pos
        self.is_country = False

class EntityPair:
    def __init__(self,
                 relation,
                 doc_id,
                 entity1,
                 entity2,
                 features=None,
                 entity_distance=None,
                 between_entities_bow=None):
        self.relation = relation
        self.doc_id = doc_id
        self.entity1 = entity1
        self.entity2 = entity2
        self.features = features
        self.entity_distance = entity_distance
        self.between_entities_bow = between_entities_bow

    @classmethod
    def from_text_gold(cls, line):
        (relation, doc_id,
         sent_index1, start_index1, end_index1, entity_type1, relation_id1, word1,
         sent_index2, start_index2, end_index2, entity_type2, relation_id2, word2) \
            = line.split()

        (sent_index1, start_index1, end_index1,
         sent_index2, start_index2, end_index2) = [int(x) for x in (sent_index1, start_index1, end_index1,
                                                                 sent_index2, start_index2, end_index2)]
        return cls(relation,
                   doc_id,
                   Entity(sent_index1, start_index1, end_index1, entity_type1, relation_id1, word1),
                   Entity(sent_index2, start_index2, end_index2, entity_type2, relation_id2, word2))

    @classmethod
    def from_text_raw(cls, line):
        relation = None
        (doc_id,
         sent_index1, start_index1, end_index1, entity_type1, relation_id1, word1,
         sent_index2, start_index2, end_index2, entity_type2, relation_id2, word2)\
            = line.split()

        (sent_index1, start_index1, end_index1,
         sent_index2, start_index2, end_index2) = [int(x) for x in (sent_index1, start_index1, end_index1,
                                                                 sent_index2, start_index2, end_index2)]
        return cls(relation,
                   doc_id,
                   Entity(sent_index1, start_index1, end_index1, entity_type1, relation_id1, word1),
                   Entity(sent_index2, start_index2, end_index2, entity_type2, relation_id2, word2))

    @classmethod
    def list_from_filename(cls, filename):

        def get_lines(filename, process_line_func):
            pairs = []
            with open(filename) as input_file:
                for line in input_file:
                    pairs.append(process_line_func(line))
            return pairs

        suffix = filename.split('.')[-1]
        if suffix == 'gold':
            return get_lines(filename, cls.from_text_gold)
        elif suffix == 'raw':
            return get_lines(filename, cls.from_text_raw)
        else:
            print('Error: Filename does not have valid suffix.')