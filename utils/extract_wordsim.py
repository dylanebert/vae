import os
import csv

with open('/home/dylan/Documents/mmid/combined.csv') as f:
    wordsim_reader = csv.DictReader(f)
    wordsim_dict = {}
    for row in wordsim_reader:
        if row['Word 1'] not in wordsim_dict:
            wordsim_dict[row['Word 1']] = {}
        if row['Word 2'] not in wordsim_dict:
            wordsim_dict[row['Word 2']] = {}
        wordsim_dict[row['Word 1']][row['Word 2']] = float(row['Human (mean)'])
        wordsim_dict[row['Word 2']][row['Word 1']] = float(row['Human (mean)'])

with open('results/mmid100sim') as f:
    with open('results/mmid100sim_extracted', 'w+') as w:
        reader = csv.DictReader(f, skipinitialspace=True)
        writer_fieldnames = reader.fieldnames + ['wordsim']
        writer = csv.DictWriter(w, fieldnames=writer_fieldnames)
        rows = list(reader)
        writer.writeheader()
        for i, row in enumerate(rows):
            if row['class a'] in wordsim_dict and row['class b'] in wordsim_dict[row['class a']]:
                sim = wordsim_dict[row['class a']][row['class b']]
                row['wordsim'] = sim
                writer.writerow(row)
