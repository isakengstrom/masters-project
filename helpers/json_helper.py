#import json
import numpy as np
import os


try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json


class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types

    Original code from: https://github.com/mpld3/mpld3/issues/434#issuecomment-340255689

    Then modified by: https://stackoverflow.com/a/49677241

    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_to_json(dic, target_dir):
    """
    A function that saves JSON files.
    """
    dumped = json.dumps(dic, cls=NumpyEncoder)
    file = open(target_dir, 'w')
    json.dump(dumped, file)
    file.close()


def read_from_json(target_dir, use_dumps=False):
    """
    A function that reads JSON files.
    """
    f = open(target_dir, 'r')
    data = json.load(f)
    if use_dumps is True:
        data = json.dumps(data)
    data = json.loads(data)
    f.close()
    return data


def combine_json_files(target_dir, save_dir=None, save_name="combined.json"):
    print(target_dir)
    _, _, json_files = next(os.walk(target_dir))

    if save_dir is None:
        save_dir = os.path.join(target_dir, "combined")
        print(save_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    combined_dict = {}
    for file_name in sorted(json_files):
        print("Adding file:", file_name)
        file_path = os.path.join(target_dir, file_name)
        file_data = read_from_json(file_path)
        combined_dict[file_name] = np.array(file_data)

    write_to_json(combined_dict, save_dir + "/" + save_name)

