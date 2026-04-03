import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, eps=1e-6):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

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

    def forward(self,logits,target):
        """
            总损失 = bce_weight * BCE + dice_weight * Dice
        """
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

        return total_loss


