import torch
import torch.nn.functional as F
import time

class EDL_VacuityExtractor:
    def __init__(self, num_classes=80):
        self.num_classes = num_classes

    def get_evidence(self, logits):
        return F.relu(logits)

    def extract_uncertainty(self, model_outputs):
        start_time = time.time()
        logits = model_outputs['logits'] 
        uncertainty_scores = []
        for batch_idx in range(logits.shape[0]):
            box_uncertainties = []
            for query_idx in range(logits.shape[1]):
                query_logits = logits[batch_idx, query_idx, :]
                evidence = self.get_evidence(query_logits)
                alpha = evidence + 1.0
                S = torch.sum(alpha)
                vacuity = self.num_classes / S
                box_uncertainties.append(vacuity.item())
            uncertainty_scores.append(box_uncertainties)
        end_time = time.time()
        print(f"[EDL Plugin] Calculated Vacuity for {logits.shape[1]} queries. Time: {(end_time - start_time)*1000:.2f} ms")
        return uncertainty_scores
