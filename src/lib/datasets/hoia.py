from .dataset import HOIDataset

class HOIA(HOIDataset):
    num_classes = 11
    num_classes_verb = 10
    _valid_ids = list(range(1, 12))
    _valid_ids_verb = list(range(1, 11))
    dataset_tag = 'hoia'
    ann_tag = {'train': 'train_hoia.json', 'test': 'test_hoia.json'}
