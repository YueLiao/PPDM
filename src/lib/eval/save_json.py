import json, os
import numpy as np
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        Default encoder for int or int.

        Args:
            self: (todo): write your description
            obj: (todo): write your description
        """
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
    """
    Save the model ascii file.

    Args:
        annot_file: (str): write your description
        out_dir: (str): write your description
        filename: (str): write your description
    """
    with open(os.path.join(out_dir, filename), 'w') as outfile:
        outfile.write(json.dumps(annot_file, cls=MyEncoder))
