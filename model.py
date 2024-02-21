import torch
from torch import nn, Tensor
from torch.nn import functional as F
from config import Classes, SequenceClasses

class AdaptiveSpatialAvgPool(nn.AdaptiveAvgPool2d):
    """Applies a 2D adaptive average pooling over a 5D input featuremap
    """
    def forward(self, input: Tensor) -> Tensor:
        assert input.ndim == 5
        b, c, t = input.shape[:3]
        output = F.adaptive_avg_pool2d(input.flatten(0, 1), self.output_size)
        out_shape = (b, c, t) + output.shape[2:]
        return output.view(out_shape)


class HeuristicFindTopNPostprocessing(nn.Module): 
    """
    """
    def __init__(
        self,
        output_len: int = None,
        pred_conversion: bool = False,
        conf_thres: float = None,
        vote_conf_by_count = False,
        select_by_conf = False,
    ):
        super().__init__()
        self.output_len = output_len
        self.pred_conversion = pred_conversion
        self.conf_thres = conf_thres
        self.vote_conf_by_count = vote_conf_by_count
        self.select_by_conf = select_by_conf

    def forward(self, x: Tensor) -> Tensor:
        # softmax
        x = F.softmax(x, -1)
        # b, t, c => b, t
        confidence, preds = x.max(-1)
        # b, t => b, output_len
        results = []
        for conf, pred in zip(confidence, preds):
            if self.conf_thres is not None:
                # filter less confident predictions
                index = conf > self.conf_thres
                conf = conf[index]
                pred = pred[index]
            # get unique predictions
            unique_pred, counts = torch.unique_consecutive(pred, return_counts=True)
            selected_indices = torch.cat([counts.new_zeros(1), torch.cumsum(counts[:-1], 0)])
            if self.output_len is not None:
                # voted confidence by count
                voted_conf = conf[selected_indices] * counts if self.vote_conf_by_count else conf[selected_indices]
                if self.select_by_conf:
                    k = min(unique_pred.shape[0], self.output_len)
                    top_index = torch.topk(voted_conf, k)[1]
                    unique_pred = unique_pred[top_index]
                else:
                    unique_pred = unique_pred[:self.output_len]
                # pad if needed
                if unique_pred.shape[0] < self.output_len:
                    pad_length = self.output_len - unique_pred.shape[0]
                    unique_pred = torch.cat(
                        [
                            unique_pred,
                            unique_pred.new_zeros(pad_length),
                        ]
                    )
            if self.pred_conversion:
                unique_pred = self._pred_conversion(unique_pred)
            else:
                unique_pred = unique_pred.tolist()
            results.append(unique_pred)
        ret = x.new_tensor(results)
        return ret

    @staticmethod
    def _pred_conversion(ordered_ped):
        ordered_ped = ordered_ped.tolist()
        pred_str = [Classes(pred).name for pred in ordered_ped]
        pred_str = "_".join(pred_str)
        if pred_str in SequenceClasses.names():
            return SequenceClasses[pred_str].value
        else:
            # work around for un-supported combinations
            return 0

    @staticmethod
    def _find_order(x, values):
        ret = []
        for v in values:
            ret.append((x == v).nonzero()[0])
        return torch.cat(ret)


class SequenceModel(nn.Module):
    def __init__(self, model: nn.Module, post_processor: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.avgpool = AdaptiveSpatialAvgPool(1)
        self.post_processor = post_processor
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.avgpool(x)
        # b, c, t, 1, 1 => b, t, c
        x = x.flatten(2).permute(0, 2, 1)
        # Flatten the layer to fc
        x = self.model.fc(x)
        x = self.post_processor(x)

        return x
