import torch
import torch.nn.functional as F

class MinConfExtractor:
    def __init__(self):
        """
        Plugin 1: Min-Confidence (Baseline Truyền thống).
        Sử dụng trực tiếp xác suất Softmax.
        Độ bất định = 1 - Max(Softmax)
        """
        pass

    def extract_uncertainty(self, model_outputs):
        """
        model_outputs: dictionary chứa 'logits' từ 1 mô hình duy nhất (Base Model).
        """
        logits = model_outputs['logits'] # [Batch, Num_Queries, Num_Classes]
        
        # Áp dụng Softmax để lấy xác suất
        probs = F.softmax(logits, dim=-1)
        
        # Lấy xác suất cao nhất của mỗi Bounding Box
        max_probs, _ = torch.max(probs, dim=-1)
        
        # Độ bất định = 1 - Xác suất cao nhất
        uncertainty = 1.0 - max_probs
        
        # Trả về list các điểm bất định cho từng ảnh trong batch
        return uncertainty.cpu().tolist()
