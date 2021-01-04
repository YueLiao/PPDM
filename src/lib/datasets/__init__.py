
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .hico import HICO
from .vcoco import VCOCO
from .hoia import HOIA

datasets = {
    'hico': HICO,
    'vcoco': VCOCO,
    'hoia': HOIA
}


def get_dataset(dataset):
    class Dataset(datasets[dataset]):
        pass

    return Dataset
