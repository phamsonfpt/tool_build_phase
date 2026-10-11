import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class OutputVarianceExtractor:
    def __init__(self, lambda_entropy=1.0, eps=1e-8):
        """
        Plugin 3 (Output Disagreement / Output Variance):
        Kết hợp 2 thành phần:
        1. Spatial Variance (Phương sai Tọa độ Bounding Box): Sự rung lắc hình học giữa 5 chuyên gia.
        2. Shannon Entropy (Entropy Nhãn Lớp): Sự bất đồng ngữ nghĩa giữa 5 chuyên gia.
        
        Score = Spatial_Variance + lambda_entropy * Shannon_Entropy
        """
        self.lambda_entropy = lambda_entropy
        self.eps = eps

    def hungarian_query_alignment(self, anchor_boxes, expert_boxes):
        """Gióng hàng các Box giữa Anchor và Expert bằng khoảng cách L1."""
        cost_matrix = torch.cdist(anchor_boxes, expert_boxes, p=1).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    def compute_shannon_entropy(self, matched_probs_across_experts):
        """
        Tính Shannon Entropy trên phân phối xác suất trung bình của các chuyên gia:
        p_bar = (1/M) * sum(p_k)
        H(p_bar) = - sum(p_bar * log(p_bar + eps))
        """
        mean_prob = torch.mean(matched_probs_across_experts, dim=0)
        entropy = -torch.sum(mean_prob * torch.log(mean_prob + self.eps)).item()
        return entropy

    def extract_uncertainty(self, anchor_outputs, list_of_expert_outputs):
        """
        anchor_outputs: dict chứa 'boxes' [N, 4] và 'logits' [N, C]
        list_of_expert_outputs: list gồm 5 dict, mỗi dict chứa 'boxes' và 'logits'
        """
        uncertainty_scores = []
        anchor_boxes = anchor_outputs['boxes']
        
        for i in range(len(anchor_boxes)):
            boxes_across_experts = []
            probs_across_experts = []
            
            for expert_out in list_of_expert_outputs:
                expert_boxes = expert_out['boxes']
                expert_logits = expert_out['logits']
                expert_probs = F.softmax(expert_logits, dim=-1)
                
                anchor_idx, expert_matched_idx = self.hungarian_query_alignment(anchor_boxes, expert_boxes)
                match_pos = np.where(anchor_idx == i)[0]
                
                if len(match_pos) > 0:
                    matched_idx = expert_matched_idx[match_pos[0]]
                    boxes_across_experts.append(expert_boxes[matched_idx])
                    probs_across_experts.append(expert_probs[matched_idx])
                else:
                    # Chuyên gia không tìm thấy vật thể -> gán box 0 và phân phối đều (Max Entropy)
                    boxes_across_experts.append(torch.zeros_like(expert_boxes[0]))
                    num_classes = expert_probs.shape[-1]
                    uniform_prob = torch.full_like(expert_probs[0], 1.0 / num_classes)
                    probs_across_experts.append(uniform_prob)
            
            # 1. Tính Spatial Variance của 4 tọa độ (x, y, w, h)
            boxes_tensor = torch.stack(boxes_across_experts) # [5, 4]
            spatial_var = torch.var(boxes_tensor, dim=0).sum().item()
            
            # 2. Tính Shannon Entropy của phân phối nhãn trung bình
            probs_tensor = torch.stack(probs_across_experts) # [5, Num_Classes]
            shannon_ent = self.compute_shannon_entropy(probs_tensor)
            
            # 3. Kết hợp 2 thành phần
            total_uncertainty = spatial_var + (self.lambda_entropy * shannon_ent)
            uncertainty_scores.append(total_uncertainty)
            
        return uncertainty_scores
