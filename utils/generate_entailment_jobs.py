import os

bash_str = '#!/bin/bash\ncd /data/nlp/vae\npython3 utils/entailment.py -i REPLACE -o /data/nlp/vae/entailment'

for filepath in os.listdir('data/stratified'):
    path = '/data/nlp/vae/data/stratified/' + filepath
    bash_content = bash_str.replace('REPLACE', path)
    with open(os.path.join('bash', filepath), 'w+') as f:
        f.write(bash_content)
