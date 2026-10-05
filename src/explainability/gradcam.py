"""
gradcam.py  (place at MedVqa/src/explainability/gradcam.py)

Grad-CAM (Selvaraju et al., 2017) for the two-input Med-VQA model.

Paper (Sec. III-D): Grad-CAM uses the gradients of a target output class flowing into the final
convolutional layer to produce a localisation map over the input image. The authors used the
pytorch-grad-cam package, modified for the two inputs (image + question).

Here the same algorithm is implemented directly:
    A_k  = activations of the target layer          [B, C, h, w]
    a_k  = spatial mean of d(score_class)/dA_k      (one weight per channel)
    CAM  = ReLU( sum_k a_k * A_k )                  -> bilinear upsample to the image size
    CAM is min-max normalised per image to [0, 1].
Target layer: image_encoder.features[28], the 13th (last) conv layer of VGG-16 (before its ReLU).
Target class: the predicted class by default (what the paper inspects), or a chosen class.
The question tokens are passed through the model unchanged; only the image branch is explained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def prepare_model_for_gradcam(model: nn.Module) -> nn.Module:
    """eval mode + no in-place ReLUs (in-place ops corrupt activation/gradient hooks). Weights unchanged."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return model


def last_conv_layer(model: nn.Module) -> nn.Module:
    layer = model.image_encoder.features[28]
    assert isinstance(layer, nn.Conv2d), "features[28] is not a Conv2d: unexpected VGG-16 layout"
    return layer


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def remove(self):
        self._handle.remove()

    def __call__(self, image, input_ids, attention_mask, target_class=None):
        """
        image [B,3,H,W], input_ids/attention_mask [B,L], target_class None (= predicted) or LongTensor [B].
        Returns: cam [B,H,W] in [0,1], raw_max [B] (max of the CAM before normalisation),
                 logits [B,num_classes] (detached), target_class [B].
        """
        self.model.zero_grad(set_to_none=True)
        self.activations = self.gradients = None
        image = image.clone().requires_grad_(True)       # guarantees a gradient path to the conv layer

        logits = self.model(image, input_ids, attention_mask)
        if target_class is None:
            target_class = logits.argmax(dim=1)
        score = logits.gather(1, target_class.view(-1, 1)).sum()
        score.backward()

        A = self.activations.detach()                     # [B,C,h,w]
        G = self.gradients.detach()
        weights = G.mean(dim=(2, 3), keepdim=True)        # [B,C,1,1]
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[:, 0]

        raw_max = cam.amax(dim=(1, 2))
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-7)
        return cam.detach(), raw_max.detach(), logits.detach(), target_class.detach()