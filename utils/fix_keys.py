import json
import os

for filename in os.listdir('model/gmc/predictions'):
    print(filename)
    filepath = os.path.join('model/gmc/predictions', filename)
    with open(filepath, 'r') as f:
        text = f.read()
        new_text = text.replace('nearest_euc', 'predictions_euc')
        with open(os.path.join('model/gmc/predictions_new', filename), 'w+') as g:
            g.write(new_text)
