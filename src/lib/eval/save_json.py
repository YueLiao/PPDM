import json, os
import numpy as np
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.uint8):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def save_json(annot_file, out_dir, filename):
    with open(os.path.join(out_dir, filename), 'w') as outfile:
        outfile.write(json.dumps(annot_file, cls=MyEncoder))
