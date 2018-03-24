from os.path import join

DATA_DIR = 'data'

POS_DIR = join(DATA_DIR,'postagged-files')
POS_SUFFIX = '.head.rel.tokenized.raw.tag'

PARSE_DIR = join(DATA_DIR,'parsed-files')
PARSE_SUFFIX = '.head.rel.tokenized.raw.parse'

GOLD_SUFFIX = 'gold'
RAW_SUFFIX = 'raw'

TRAIN_GOLD_PATH = join(DATA_DIR, 'rel-trainset.gold')

DEV_GOLD_PATH = join(DATA_DIR, 'rel-devset.gold')
DEV_RAW_PATH = join(DATA_DIR, 'rel-devset.raw')

TEST_GOLD_PATH = join(DATA_DIR, 'rel-testset.gold')
TEST_RAW_PATH = join(DATA_DIR, 'rel-testset.raw')

TRAIN_FEATURE_PATH = join('features', 'rel-trainset-features.txt')
DEV_FEATURE_PATH = join('features', 'rel-devset-features.txt')
TEST_FEATURE_PATH = join('features', 'rel-testset-features.txt')

DEV_PREDICTIONS_PATH = join('predictions', 'rel-devset.tagged')
TEST_PREDICTIONS_PATH = join('predictions', 'rel-testset.tagged')