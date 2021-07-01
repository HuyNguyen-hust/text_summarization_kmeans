import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from argparse import ArgumentParser
import plotly.express as px
import joblib
import os
from tqdm import tqdm

from utils import get_content


def get_content_embeddings(model_path, content):
    print('get_content_embeddings...')
    embeddings = []
    model = joblib.load(model_path)
    for sentence in tqdm(content):
        embeddings.append(model.encode(sentence))
    return np.array(embeddings)


def get_clusters(content, content_embeddings, num_sentences, visualize):
    print('clustering...')
    kmeans = KMeans(n_clusters=num_sentences,
                    random_state=0).fit(content_embeddings)
    if visualize == True:
        pca = PCA(n_components=3)
        result = pca.fit_transform(content_embeddings)

        df = pd.DataFrame({
            'sent': content,
            'cluster': kmeans.labels_.astype(str),
            'x': result[:, 0],
            'y': result[:, 1],
            'z': result[:, 2]
        })

        fig = px.scatter_3d(df, x='x', y='y', z='z',
                            color='cluster', hover_name='sent',
                            range_x=[df.x.min()-1, df.x.max()+1],
                            range_y=[df.y.min()-1, df.y.max()+1],
                            range_z=[df.z.min()-1, df.z.max()+1])

        fig.update_traces(hovertemplate='<b>%{hovertext}</b>')
        fig.show()

    print('--------------------------------')
    for label in range(num_sentences):
        print('demo sentences in cluster ', label)
        cnt = 0
        for idx, predicted_label in enumerate(kmeans.labels_):
            if predicted_label == label:
                cnt += 1
                print(content[idx])
                if cnt >= 5:
                    print('...')
                    break
        print('--------------------------------')

    return kmeans


def get_top_K(cluster_result, content, content_embeddings, num_sentences):
    indices_by_cluster = [[] for _ in range(num_sentences)]
    for i in range(len(content_embeddings)):
        indices_by_cluster[cluster_result.labels_[i]].append(i)

    def get_closest_point_to_centroid(centroid_idx):
        min = 1e9
        min_idx = -1
        for point_idx in indices_by_cluster[centroid_idx]:
            distance = np.sum((content_embeddings - cluster_result.cluster_centers_[centroid_idx]) ** 2)
            if distance < min:
                min = distance
                min_idx = point_idx
        return min, min_idx

    k_sentences = []
    for centroid_idx in range(num_sentences):
        _, min_idx = get_closest_point_to_centroid(centroid_idx)
        k_sentences.append(content[min_idx])
    return k_sentences


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = ArgumentParser()
    parser.add_argument('--paper_url', type=str,
                        default=None, help='a paper url')
    parser.add_argument('--model_path', type=str,
                        default='model/paraphrase-xlm-r-multilingual-v1.pkl', help='path to pretrained model')
    parser.add_argument('--k', type=int, default=3,
                        help='number of sentences after summarizing')
    parser.add_argument('--visualize', type=bool, default=False,
                        help='True if you want to visualize sentence embeddings and False if not')

    args = parser.parse_args()
    paper_url = args.paper_url
    model_path = args.model_path
    num_sentences = args.k
    visualize = args.visualize

    content = get_content(paper_url)
    content_embeddings = get_content_embeddings(model_path, content)
    cluster_result = get_clusters(content, content_embeddings,
                 num_sentences=num_sentences, visualize=visualize)
    k_sents = get_top_K(cluster_result, content, content_embeddings, num_sentences)

    print('Result:')
    for sentence in k_sents:
        print(sentence)


