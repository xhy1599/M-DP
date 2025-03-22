import torch
import torch.nn as nn
import torchvision
from typing import Callable




def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = nn.Identity()
    return resnet

def replace_submodules(root_module: nn.Module,
                       predicate: Callable[[nn.Module], bool],
                       func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(root_module: nn.Module,
                       features_per_group: int = 16) -> nn.Module:
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


class ResNet50Encoder(nn.Module):
    """
    Custom ResNet50 encoder with additional fully connected layers
    to output embeddings of desired dimensionality.
    """
    
    def __init__(self, encoding_size=512, context_size=3,pretrained=True, use_gn=True):
        """
        Initialize a ResNet50 model with custom output dimension.
        
        Args:
            encoding_size (int): Size of the output encoding vector
            pretrained (bool): Whether to use pretrained weights
            use_gn (bool): Whether to replace BatchNorm with GroupNorm
        """
        super().__init__()
        
        # Get weights based on pretrained parameter
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        
        # Initialize the base ResNet50 model without the final FC layer
        self.backbone = get_resnet('resnet50', weights=weights)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=3*(context_size+1),  
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        
        # Replace BatchNorm with GroupNorm if requested
        if use_gn:
            self.backbone = replace_bn_with_gn(self.backbone)
        
        # Add a new fully connected layer to map from ResNet50's output (2048) to desired encoding size
        self.fc = nn.Sequential(
            nn.Linear(2048, encoding_size),
            #nn.ReLU(inplace=True)
        )
    
    def forward(self, obs_img, input_goal_mask=None):
        """
        Forward pass through the network.
        
        Args:
            obs_img (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, encoding_size)
        """
        assert obs_img.shape[2] == 224, f"input image height is not 224"
        assert obs_img.shape[3] == 224, f"input image wideth is not 224"
        # Extract features from the backbone
        features = self.backbone(obs_img)
        
        # Pass through the final FC layer
        encoding = self.fc(features)
        
        return encoding

# Example usage:
# model = ResNet50Encoder(encoding_size=512)
# dummy_input = torch.randn(1, 9, 224, 224)
# output = model(dummy_input)
# print(output.shape)  # Should print torch.Size([1, 512])