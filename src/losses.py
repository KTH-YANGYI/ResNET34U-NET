import torch
import torch.nn as nn
import torch.nn.functional as F


def get_primary_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


def prepare_logits_and_target(logits,target):
    """
    logits :[batch_size, logit, height,width]
    maybe targets:[batchsize, height,width]
    目的就是使logits和target形状一致，如果targets没有通道维，就补一个通道维
    """
    if logits.ndim!=4:
        raise ValueError(
            f"logits的形状为4维,但当前是{tuple(logits.shape)}"
        )
    if target.ndim ==3:
        target =target.unsqueeze(1)

    if target.ndim !=4:
        raise ValueError(
            f"targets的形状应为4维,但当前是{tuple(target.shape)}"
        )

    target =target.float()

    if logits.shape !=target.shape:
        raise ValueError(
            f"logits 和 target 的形状必须一致，"
            f"但当前 logits={tuple(logits.shape)}, target={tuple(target.shape)}"
        )
        
    return logits,target

class DiceLoss(nn.Module):
    """
    计算diceloss，预测区域和真值区域的重叠程度，重叠越大loss越小
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self,logits,target):
        logits,target = prepare_logits_and_target(logits,target)

        probs = torch.sigmoid(logits)
        
        dims=(1,2,3)
        intersection = (probs*target).sum(dim=dims)

        probs_sum = probs.sum(dim=dims)
        target_sum = target.sum(dim=dims)

        dice = (2.0*intersection+self.eps)/(probs_sum+target_sum+self.eps)
        loss = 1.0-dice
        return loss.mean()
    
class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=None,
        normal_fp_loss_weight=0.0,
        normal_fp_topk_ratio=1.0,
        deep_supervision_weight=0.0,
        deep_supervision_decay=0.5,
        boundary_aux_weight=0.0,
        boundary_width=3,
        eps=1e-6,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.normal_fp_loss_weight = float(normal_fp_loss_weight)
        self.normal_fp_topk_ratio = float(normal_fp_topk_ratio)
        self.deep_supervision_weight = float(deep_supervision_weight)
        self.deep_supervision_decay = float(deep_supervision_decay)
        self.boundary_aux_weight = float(boundary_aux_weight)
        self.boundary_width = max(1, int(boundary_width))

        self.dice_loss = DiceLoss(eps=eps)
        
        self.use_pos_weight = pos_weight is not None

        if self.use_pos_weight:
            self.register_buffer(
                "pos_weight",
                torch.tensor(float(pos_weight), dtype=torch.float32),
            )            
        else:
            # 如果 pos_weight，我们也注册一个占位张量
            self.register_buffer(
                "pos_weight",
                torch.tensor(1.0, dtype=torch.float32),
            )

    def segmentation_loss(self, logits, target, include_normal_fp=True):
        logits, target = prepare_logits_and_target(logits, target)
        if self.use_pos_weight:
            bce = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=self.pos_weight,
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits,
                target,
            )
        # 计算 Dice 部分
        dice = self.dice_loss(logits, target)

        # 加权相加
        total_loss = self.bce_weight * bce + self.dice_weight * dice

        if include_normal_fp and self.normal_fp_loss_weight > 0.0:
            empty_target = target.sum(dim=(1, 2, 3)) <= 0.0
            if bool(empty_target.any()):
                normal_probs = torch.sigmoid(logits[empty_target]).flatten(start_dim=1)
                topk_ratio = min(max(self.normal_fp_topk_ratio, 0.0), 1.0)
                if topk_ratio <= 0.0:
                    normal_fp_loss = normal_probs.mean()
                else:
                    topk_count = max(1, int(round(normal_probs.shape[1] * topk_ratio)))
                    normal_fp_loss = normal_probs.topk(k=topk_count, dim=1).values.mean()
                total_loss = total_loss + self.normal_fp_loss_weight * normal_fp_loss

        return total_loss

    def boundary_target(self, target):
        _, target = prepare_logits_and_target(target, target)
        kernel_size = self.boundary_width
        padding = kernel_size // 2
        dilated = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-target, kernel_size=kernel_size, stride=1, padding=padding)
        return (dilated - eroded > 0.0).float()

    def forward(self,logits,target):
        """
            总损失 = bce_weight * BCE + dice_weight * Dice
        """
        if isinstance(logits, dict):
            output = logits
            main_logits = output["logits"]
        else:
            output = {}
            main_logits = logits

        total_loss = self.segmentation_loss(main_logits, target, include_normal_fp=True)

        aux_logits = output.get("aux_logits", [])
        if self.deep_supervision_weight > 0.0 and len(aux_logits) > 0:
            aux_loss = 0.0
            total_weight = 0.0
            for index, aux_logit in enumerate(aux_logits):
                weight = self.deep_supervision_decay ** index
                aux_loss = aux_loss + weight * self.segmentation_loss(aux_logit, target, include_normal_fp=False)
                total_weight += weight
            if total_weight > 0.0:
                total_loss = total_loss + self.deep_supervision_weight * (aux_loss / total_weight)

        boundary_logits = output.get("boundary_logits")
        if self.boundary_aux_weight > 0.0 and boundary_logits is not None:
            _, prepared_target = prepare_logits_and_target(main_logits, target)
            boundary_target = self.boundary_target(prepared_target)
            boundary_logits, boundary_target = prepare_logits_and_target(boundary_logits, boundary_target)
            boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
            total_loss = total_loss + self.boundary_aux_weight * boundary_loss

        return total_loss
