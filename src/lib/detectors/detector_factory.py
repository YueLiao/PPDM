from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .hoidet import HoidetDetector

detector_factory = {
  'hoidet': HoidetDetector,
}
