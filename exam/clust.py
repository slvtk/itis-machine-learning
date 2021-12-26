import numpy as np
import pandas as pd
import io

from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

df = pd.read_csv(io.BytesIO(open("resources/out.csv", 'rb').read()))
df.append(pd.read_csv(io.BytesIO(open("resources/out1.csv", 'rb').read())))
df.append(pd.read_csv(io.BytesIO(open("resources/out2.csv", 'rb').read())))
df.append(pd.read_csv(io.BytesIO(open("resources/out3.csv", 'rb').read())))

if __name__ == "__main__":
    print('CSV data: ', df)

    vectors = TfidfVectorizer(analyzer='char', ngram_range=(1, 2)).fit_transform(df['vector'].values.astype('U'))
    x_axis = vectors.todense()
    y_axis = TSNE(n_components=2).fit_transform(x_axis)
    plt.scatter(y_axis[:, 0], y_axis[:, 1], cmap=plt.cm.Spectral)
    plt.show()
    model = DBSCAN().fit(vectors)
    labels = model.labels_

    no_clusters = len(np.unique(labels))
    no_noise = np.sum(np.array(labels) == -1, axis=0)
    df['labels'] = labels

    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True

    print('Clusters count: ', len(set(labels)) - (1 if -1 in labels else 0))
