def compute_reduced(self):
    if self.config.computed_encodings:
        with open(self.config.encodings_path, 'rb') as f:
            self.encodings = pickle.load(f)
        print('Loaded encodings')
    else:
        self.compute_encodings()
    if self.config.computed_means:
        with open(self.config.means_path, 'rb') as f:
            self.class_means = pickle.load(f)
        print('Loaded means')
    else:
        self.compute_means()
    print('Computing reduced')
    labels = list(self.class_means.keys())
    mean_vals = np.array(list(self.class_means.values()))
    pca = PCA(n_components=2)
    means_reduced = pca.fit_transform(mean_vals)
    encodings_reduced = {}
    n = len(labels)
    for i, label in enumerate(labels):
        if i % 100 == 0:
            print('{0} of {1}'.format(i, n), end='\r')
        encodings_reduced[label] = pca.transform(np.array(list(self.encodings[label]['encodings'])))
    self.means_reduced = dict(zip(labels, means_reduced))
    self.encodings_reduced = encodings_reduced
    with open(self.config.encodings_reduced_path, 'wb+') as f:
        pickle.dump(self.encodings_reduced, f)
    with open(self.config.means_reduced_path, 'wb+') as f:
        pickle.dump(self.means_reduced, f)
    self.config.computed_reduced = True
