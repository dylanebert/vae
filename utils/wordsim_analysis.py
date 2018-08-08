import csv
from matplotlib import pyplot as plt
from scipy.stats import pearsonr as r

with open('results/mmid100sim_extracted') as f:
    reader = csv.DictReader(f)

    cossim = []
    wordsim = []
    z_dist = []
    img_dist = []
    kl_div = []

    for row in reader:
        cossim.append(float(row['cossim']))
        wordsim.append(float(row['wordsim']))
        z_dist.append(float(row['z_dist']))
        img_dist.append(float(row['img_dist']))
        kl_div.append(float(row['kl-div']))

    plt.figure(figsize=(8, 8))

    subplot = plt.subplot(2, 2, 1)
    subplot.scatter(wordsim, cossim)
    rval, pval = r(wordsim, cossim)
    plt.title('r: {0:.5f}, p: {1:.5f}'.format(rval, pval))
    plt.xlabel('Human wordsim')
    plt.ylabel('Computed visual similarity (cossim)')

    subplot = plt.subplot(2, 2, 2)
    subplot.scatter(wordsim, z_dist)
    rval, pval = r(wordsim, z_dist)
    plt.title('r: {0:.5f}, p: {1:.5f}'.format(rval, pval))
    plt.xlabel('Human wordsim')
    plt.ylabel('Computed visual similarity (z_dist)')

    subplot = plt.subplot(2, 2, 3)
    subplot.scatter(wordsim, img_dist)
    rval, pval = r(wordsim, img_dist)
    plt.title('r: {0:.5f}, p: {1:.5f}'.format(rval, pval))
    plt.xlabel('Human wordsim')
    plt.ylabel('Computed visual similarity (img_dist)')

    subplot = plt.subplot(2, 2, 4)
    subplot.scatter(wordsim, kl_div)
    rval, pval = r(wordsim, kl_div)
    plt.title('r: {0:.5f}, p: {1:.5f}'.format(rval, pval))
    plt.xlabel('Human wordsim')
    plt.ylabel('Computed visual similarity (kl_div)')

    plt.show()
