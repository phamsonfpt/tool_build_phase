import torch
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

class FCDLatentExtractor:
    def __init__(self, jitter_eps=1e-5):
        self.jitter_eps = jitter_eps

    def hungarian_query_alignment(self, anchor_boxes, expert_boxes):
        cost_matrix = torch.cdist(anchor_boxes, expert_boxes, p=1).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    def compute_mahalanobis_variance(self, matched_latent_vectors):
        X = matched_latent_vectors
        mu = torch.mean(X, dim=0)
        X_centered = X - mu
        cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)
        identity = torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
        cov_matrix_stable = cov_matrix + self.jitter_eps * identity
        inv_cov_matrix = torch.linalg.inv(cov_matrix_stable)
        mahalanobis_distances = []
        for i in range(X.shape[0]):
            vec = X_centered[i].unsqueeze(0)
            dist_sq = vec @ inv_cov_matrix @ vec.T
            dist = torch.sqrt(torch.clamp(dist_sq, min=1e-9))
            mahalanobis_distances.append(dist.item())
        variance = np.var(mahalanobis_distances)
        return variance

    def extract_uncertainty(self, anchor_outputs, list_of_expert_outputs):
        start_time = time.time()
        uncertainty_scores = []
        anchor_boxes = anchor_outputs['boxes']
        for i in range(len(anchor_boxes)):
            query_latents_across_experts = []
            for expert_idx, expert_out in enumerate(list_of_expert_outputs):
                expert_boxes = expert_out['boxes']
                expert_latents = expert_out['latent_queries']
                anchor_idx, expert_matched_idx = self.hungarian_query_alignment(anchor_boxes, expert_boxes)
                match_pos = np.where(anchor_idx == i)[0]
                if len(match_pos) > 0:
                    matched_latent = expert_latents[expert_matched_idx[match_pos[0]]]
                    query_latents_across_experts.append(matched_latent)
                else:
                    query_latents_across_experts.append(torch.zeros_like(expert_latents[0]))
            query_tensor = torch.stack(query_latents_across_experts)
            fcd_score = self.compute_mahalanobis_variance(query_tensor)
            uncertainty_scores.append(fcd_score)
        end_time = time.time()
        print(f"[FCD Plugin] Calculated for {len(anchor_boxes)} objects. Time: {(end_time - start_time)*1000:.2f} ms")
        return uncertainty_scores
