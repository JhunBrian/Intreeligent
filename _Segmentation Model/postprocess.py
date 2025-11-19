import torch

class PredictionProcessor:
    """
    Only accepts one instance
    """
    def __init__(self, mask_threshold=0.5, score_threshold=0.5):
        self.mask_threshold = mask_threshold
        self.score_threshold = score_threshold

    def __call__(self, prediction):
        # Handle empty predictions
        if not prediction or "masks" not in prediction or "scores" not in prediction:
            return {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
                "scores": torch.empty((0,), dtype=torch.float32),
                "masks": torch.empty((0, 1, 1), dtype=torch.uint8)  # single 2D mask per instance
            }

        # Filter by score threshold
        keep = prediction["scores"] >= self.score_threshold
        prediction = {k: v[keep] for k, v in prediction.items()}

        # Threshold masks and convert to uint8
        if "masks" in prediction and len(prediction["masks"]) > 0:
            # originally: (N,1,H,W) -> convert to (N,H,W)
            prediction["masks"] = ((prediction["masks"] > self.mask_threshold)
                                    .to(torch.uint8)
                                    .squeeze(1))  # remove channel dim

        return prediction