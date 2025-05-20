
# model_architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
REFLECTIVITY_MAX = 70.0
TIME_HORIZON = 12
NUM_FILTERS = 64

class SpatiotemporalUNet(nn.Module):
    def __init__(self, input_channels):
        super(SpatiotemporalUNet, self).__init__()
        
        # Encoder
        self.c1_conv1 = nn.Conv2d(input_channels, NUM_FILTERS, 3, padding='same')
        self.c1_bn1 = nn.BatchNorm2d(NUM_FILTERS)
        self.c1_conv2 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, 3, padding='same')
        self.c1_bn2 = nn.BatchNorm2d(NUM_FILTERS)
        self.p1_pool = nn.MaxPool2d(2, 2)
        
        self.c2_conv1 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS * 2, 3, padding='same')
        self.c2_bn1 = nn.BatchNorm2d(NUM_FILTERS * 2)
        self.c2_conv2 = nn.Conv2d(NUM_FILTERS * 2, NUM_FILTERS * 2, 3, padding='same')
        self.c2_bn2 = nn.BatchNorm2d(NUM_FILTERS * 2)
        self.p2_pool = nn.MaxPool2d(2, 2)
        
        # Bridge
        self.bn_conv1 = nn.Conv2d(NUM_FILTERS * 2, NUM_FILTERS * 4, 3, padding='same')
        self.bn_bn1 = nn.BatchNorm2d(NUM_FILTERS * 4)
        self.bn_conv2 = nn.Conv2d(NUM_FILTERS * 4, NUM_FILTERS * 4, 3, padding='same')
        self.bn_bn2 = nn.BatchNorm2d(NUM_FILTERS * 4)
        
        # Decoder (using bilinear upsampling instead of transposed convolutions)
        self.u1_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.u1_conv = nn.Conv2d(NUM_FILTERS * 4, NUM_FILTERS * 2, 3, padding='same')
        self.u1_bn = nn.BatchNorm2d(NUM_FILTERS * 2)
        self.c3_conv1 = nn.Conv2d(NUM_FILTERS * 4, NUM_FILTERS * 2, 3, padding='same')
        self.c3_bn1 = nn.BatchNorm2d(NUM_FILTERS * 2)
        self.c3_conv2 = nn.Conv2d(NUM_FILTERS * 2, NUM_FILTERS * 2, 3, padding='same')
        self.c3_bn2 = nn.BatchNorm2d(NUM_FILTERS * 2)
        
        self.u2_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.u2_conv = nn.Conv2d(NUM_FILTERS * 2, NUM_FILTERS, 3, padding='same')
        self.u2_bn = nn.BatchNorm2d(NUM_FILTERS)
        self.c4_conv1 = nn.Conv2d(NUM_FILTERS * 2, NUM_FILTERS, 3, padding='same')
        self.c4_bn1 = nn.BatchNorm2d(NUM_FILTERS)
        self.c4_conv2 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, 3, padding='same')
        self.c4_bn2 = nn.BatchNorm2d(NUM_FILTERS)
        
        # High reflectivity attention module
        self.hr_attention = HighReflectivityAttention(NUM_FILTERS)
        
        # Output layer
        self.out_conv = nn.Conv2d(NUM_FILTERS, TIME_HORIZON, 1, padding='same')
        
    def forward(self, x):
        # Encoder path with batch norm and LeakyReLU
        # Changed LeakyReLU slope from 0.1 to 0.2 throughout the network
        c1 = F.leaky_relu(self.c1_bn1(self.c1_conv1(x)), 0.2)
        c1 = F.leaky_relu(self.c1_bn2(self.c1_conv2(c1)), 0.2)
        p1 = self.p1_pool(c1)
        
        c2 = F.leaky_relu(self.c2_bn1(self.c2_conv1(p1)), 0.2)
        c2 = F.leaky_relu(self.c2_bn2(self.c2_conv2(c2)), 0.2)
        p2 = self.p2_pool(c2)
        
        # Bridge
        bn = F.leaky_relu(self.bn_bn1(self.bn_conv1(p2)), 0.2)
        bn = F.leaky_relu(self.bn_bn2(self.bn_conv2(bn)), 0.2)
        
        # Decoder path with proper size handling
        u1 = self.u1_bn(self.u1_conv(self.u1_up(bn)))
        
        # Handle size mismatch for skip connection
        diffY = c2.size()[2] - u1.size()[2]
        diffX = c2.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        u1 = torch.cat([u1, c2], dim=1)
        c3 = F.leaky_relu(self.c3_bn1(self.c3_conv1(u1)), 0.2)
        c3 = F.leaky_relu(self.c3_bn2(self.c3_conv2(c3)), 0.2)
        
        u2 = self.u2_bn(self.u2_conv(self.u2_up(c3)))
        
        # Handle size mismatch for second skip connection
        diffY = c1.size()[2] - u2.size()[2]
        diffX = c1.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        u2 = torch.cat([u2, c1], dim=1)
        c4 = F.leaky_relu(self.c4_bn1(self.c4_conv1(u2)), 0.2)
        c4 = F.leaky_relu(self.c4_bn2(self.c4_conv2(c4)), 0.2)
        
        # Apply high reflectivity attention
        c4 = self.hr_attention(c4)
        
        # Output layer
        outputs = self.out_conv(c4)
        return outputs

class HighReflectivityAttention(nn.Module):
    """Module designed to enhance high reflectivity features"""
    def __init__(self, channels):
        super(HighReflectivityAttention, self).__init__() 
        self.conv = nn.Conv2d(channels, channels, 3, padding='same')
        self.bn = nn.BatchNorm2d(channels)
        self.gate = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        # Residual connection
        residual = x
        
        # Create attention map with modified activation (LeakyReLU 0.2 instead of 0.1)
        attention = F.leaky_relu(self.bn(self.conv(x)), 0.2)
        attention = torch.sigmoid(self.gate(attention))
        
        # Apply attention with residual connection and a small constant scaling factor of 1.2
        return residual + 1.2 * attention * x

def create_spatiotemporal_unet(input_channels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SpatiotemporalUNet(input_channels).to(device)
    return model

def weighted_mse(y_true, y_pred):
    """
    Enhanced weighted MSE loss function that gives progressively higher weights
    to higher reflectivity values.
    """
    # Denormalize to actual reflectivity values (dBZ)
    y_true_denorm = y_true * (REFLECTIVITY_MAX / 3) + (REFLECTIVITY_MAX / 2)
    
    # Base weight = 1.0
    weights = torch.ones_like(y_true_denorm)
    
    # Apply in descending order to prevent overwriting higher weights
    # Modified weights to give even more emphasis to high reflectivity areas
    weights = torch.where(y_true_denorm > 40, torch.tensor(20.0, device=y_true.device), weights)  # Increased from 16.0
    weights = torch.where((y_true_denorm > 30) & (y_true_denorm <= 40), torch.tensor(10.0, device=y_true.device), weights)  # Increased from 8.0
    weights = torch.where((y_true_denorm > 20) & (y_true_denorm <= 30), torch.tensor(4.0, device=y_true.device), weights)  # Kept same
    weights = torch.where((y_true_denorm > 10) & (y_true_denorm <= 20), torch.tensor(2.0, device=y_true.device), weights)  # Kept same
    
    # Calculate weighted MSE
    error = torch.square(y_true - y_pred)
    return torch.mean(weights * error)
