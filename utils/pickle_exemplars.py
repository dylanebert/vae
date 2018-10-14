import pickle
import json

def pickle_exemplars(from_path, to_path):
    exemplars = {}
    with open(from_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            exemplars[data['label']] = data['encoding']
    with open(to_path, 'wb+') as f:
        pickle.dump(exemplars, f)

#pickle_exemplars('model/gmc/exemplars_nearest', 'model/gmc/exemplars_nearest.p')
pickle_exemplars('model/gmc/exemplars_random', 'model/gmc/exemplars_random.p')
