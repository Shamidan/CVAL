import random
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from api.models.model import ClusteringWorkerModel, FrameModel, EmbeddingPredictionModel

INF = 1e12


def k_centroid_greedy(dis_matrix, k):
    n = dis_matrix.shape[0]
    centroids = []
    c = np.random.randint(0, n, (1,))[0]
    centroids.append(c)
    i = 1
    while i < k:
        centroids_diss = dis_matrix[:, centroids].copy()
        centroids_diss = centroids_diss.min(axis=1)
        centroids_diss[centroids] = -1
        new_c = np.argmax(centroids_diss)
        centroids.append(new_c)
        i += 1
    return centroids


def kmeans(dis_matrix, k, n_iter=100):
    n = dis_matrix.shape[0]
    centroids = k_centroid_greedy(dis_matrix, k)
    data_indices = np.arange(n)
    assign_dis_records = []
    for _ in range(n_iter):
        centroid_dis = dis_matrix[:, centroids]
        cluster_assign = np.argmin(centroid_dis, axis=1)
        assign_dis = centroid_dis.min(axis=1).sum()
        assign_dis_records.append(assign_dis)

        new_centroids = []
        for i in range(k):
            cluster_i = data_indices[cluster_assign == i]
            #print(data_indices[cluster_assign == i])
            assert len(cluster_i) >= 1
            dis_mat_i = dis_matrix[cluster_i][:, cluster_i]
            new_centroid_i = cluster_i[np.argmin(dis_mat_i.sum(axis=1))]
            new_centroids.append(new_centroid_i)
        centroids = np.array(new_centroids)
    return centroids.tolist()


@torch.no_grad()
def get_img_score_distance_matrix_slow(
        all_labels,
        all_scores,
        all_feats,
        score_thr=0.,
        same_label=True
):
    n_images = all_labels.size(0)
    n_dets = all_labels.size(1)
    feat_dim = all_feats.size(-1)
    dets_indices = torch.arange(n_dets).to(device=all_feats.device)

    all_feats = F.normalize(all_feats, p=2, dim=-1)
    all_feats_t = all_feats.transpose(1, 2)

    all_score_valid = (all_scores > score_thr).to(dtype=all_feats.dtype)
    all_score_valid_t = all_score_valid[:, :, None].transpose(1, 2)
    all_labels_t = all_labels[:, :, None].transpose(1, 2)

    distances = []
    for i in range(n_images):
        # torch.cuda.empty_cache()

        labels_i = all_labels[i].clone()  # [n_dets]
        labels_i[(labels_i == -2)] = -3
        scores_valid_i = all_score_valid[i]  # [n_dets]
        scores_i = all_scores[i]  # [n_dets]
        feats_i = all_feats[i]  # [n_dets, feat_dim]

        feat_distances_i = -1 * torch.matmul(feats_i.view(1, n_dets, feat_dim),
                                             all_feats_t) + 1  # [n_images, n_dets, n_dets]
        feat_distances_i[i, dets_indices, dets_indices] = 0  # force diag to 0, avoid numerical unstable
        score_valid = torch.matmul(scores_valid_i.view(1, n_dets, 1), all_score_valid_t)  # [n_images, n_dets, n_dets]

        if same_label:
            labels_i = labels_i[:, None].repeat(1, n_dets)  # [n_dets, n_dets]
            label_valid = (labels_i.view(1, n_dets, n_dets) == all_labels_t).to(dtype=all_feats.dtype)
        else:
            label_valid = torch.ones_like(score_valid)

        label_invalid = (1 - label_valid).to(dtype=torch.bool)
        score_invalid = (1 - score_valid).to(dtype=torch.bool)

        feat_distances_i[label_invalid] = 2.
        feat_distances_i[score_invalid] = INF

        feat_distances_i = feat_distances_i.min(dim=-1)[0]  # [n_images, n_dets]

        norm = (score_valid.max(dim=-1)[0] * scores_i[None, :]).sum(dim=-1) + 0.00001
        '''
            Potential BUG:
            If no box > score_thr in both images, the algorithm fails. But this is unlikely to happen
            '''
        # feat_distances_i[feat_distances_i > 2] = 0.

        feat_distances_i2 = feat_distances_i * scores_i[None, :]
        feat_distances_i3 = feat_distances_i2.sum(dim=-1) / norm
        distances.append(feat_distances_i3.cpu())

    feat_distance = torch.stack(distances, dim=0)
    feat_distance = 0.5 * (feat_distance + feat_distance.transpose(0, 1))
    return feat_distance


def return_files(input_data: ClusteringWorkerModel):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    n_dets = max([len(i.predictions) for i in input_data.frames])
    feat_dim = max([max([len(j.embedding) for j in i.predictions]) for i in input_data.frames])
    n_images = len(input_data.frames)
    embeddings = [[[j.embedding, j.score, j.category_id] for j in i.predictions] for i in input_data.frames]
    names = [x.frame_id for x in input_data.frames]
    all_labels_np = np.ones((n_images, n_dets)) * (-2)
    all_scores_np = np.ones((n_images, n_dets)) * 0
    all_feats_np = np.zeros((n_images, n_dets, feat_dim))
    for i in range(n_images):
        list_a = embeddings[i]
        for j, a in enumerate(list_a):
            all_labels_np[i, j] = a[2]
            all_scores_np[i, j] = a[1]
            #print(a[0])
            all_feats_np[i, j] = a[0]
    all_labels = torch.tensor(all_labels_np).to(device)
    all_scores = torch.tensor(all_scores_np).to(device)
    all_feats = torch.tensor(all_feats_np).to(device)
    k = input_data.num_of_samples
    img_dist_mat = get_img_score_distance_matrix_slow(all_labels, all_scores, all_feats, score_thr=0.05)
    centr = sorted(kmeans(img_dist_mat.numpy(), k))
    return list(sorted([names[i] for i in centr]))


if __name__ == "__main__":
    emb_sz = 500
    r = ClusteringWorkerModel(
        num_of_samples=1,
        frames=[
            FrameModel(
                frame_id=uuid.uuid4().hex,
                predictions=[EmbeddingPredictionModel(embedding=[.5, .3, .2]*10, category_id=1, score=.5)]
            ),
            FrameModel(
                frame_id=uuid.uuid4().hex,
                predictions=[EmbeddingPredictionModel(
                    embedding=map(
                        lambda x: random.random(), range(emb_sz)
                                  ), category_id=1, score=.5)]
            ),
            FrameModel(
                frame_id=uuid.uuid4().hex,
                predictions=[EmbeddingPredictionModel(embedding=[.5, .3, .2]*10, category_id=1, score=.5)]
            )
        ]
    )
    print(r.json())
    # out_id, out_name = return_files(r)
    # print(out_name, out_id)
