import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BASE
from data.dataset import SpectroDataset

# every class in this file uses the same dataset:
class CNN_BASE(BASE):
    def __init__(self):
        super(CNN_BASE, self).__init__()
    @classmethod
    def supported_dataset(cls):
        return SpectroDataset 


####################################################
########### BASELINE MODEL FROM SCRATCH ############
####################################################
class CNN(CNN_BASE):
    def __init__(self, input_channels=1, H=128, W=32, num_classes=30):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_conv = nn.Dropout2d(p=0.2)

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128 , kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256 , kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512 , kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # after pooling
        final_h = H // 32
        final_w = W // 32

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * final_h * final_w, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)



        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout_conv(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout_conv(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)

        return x









####################################################
###########        LeNet5 Model         ############
####################################################
# models from : https://www.geeksforgeeks.org/machine-learning/convolutional-neural-network-cnn-architectures/
class LeNet5(CNN_BASE):
    def __init__(self, input_channels=1, H=128, W=32, num_classes=30):
        # Call the parent class's init method
        super(LeNet5, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5, stride=1)
        
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Calculate actual dimensions after convolutions and pooling
        # After conv1 (kernel=5): H -> H-4, W -> W-4 (no padding)
        # After pool1 (kernel=2, stride=2): H -> (H-4)//2, W -> (W-4)//2
        # After conv2 (kernel=5): H -> (H-4)//2 - 4, W -> (W-4)//2 - 4
        # After pool2 (kernel=2, stride=2): H -> ((H-4)//2 - 4)//2, W -> ((W-4)//2 - 4)//2
        
        # For H=128, W=32:
        # After conv1: 124 x 28
        # After pool1: 62 x 14
        # After conv2: 58 x 10
        # After pool2: 29 x 5
        
        h_after_conv1 = H - 4
        w_after_conv1 = W - 4
        h_after_pool1 = h_after_conv1 // 2
        w_after_pool1 = w_after_conv1 // 2
        h_after_conv2 = h_after_pool1 - 4
        w_after_conv2 = w_after_pool1 - 4
        final_h = h_after_conv2 // 2
        final_w = w_after_conv2 // 2
        
        # First Fully Connected Layer
        self.fc1 = nn.Linear(in_features=16 * final_h * final_w, out_features=120)
        
        # Second Fully Connected Layer
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        
        # Output Layer
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        
        # Store final dimensions for forward pass
        self.final_h = final_h
        self.final_w = final_w
    
    def forward(self, x):
        # Pass the input through the first convolutional layer and activation function
        x = self.pool(F.relu(self.conv1(x)))
        
        # Pass the output of the first layer through 
        # the second convolutional layer and activation function
        x = self.pool(F.relu(self.conv2(x)))
        
        # Reshape the output to be passed through the fully connected layers
        x = x.view(-1, 16 * self.final_h * self.final_w)
        
        # Pass the output through the first fully connected layer and activation function
        x = F.relu(self.fc1(x))
        
        # Pass the output of the first fully connected layer through 
        # the second fully connected layer and activation function
        x = F.relu(self.fc2(x))
        
        # Pass the output of the second fully connected layer through the output layer
        x = self.fc3(x)
        
        # Return the final output
        return x











#####################################################
###########        AlexNet Model         ############
#####################################################
class AlexNet(CNN_BASE):
    def __init__(self, num_classes=1000):
        # Call the parent class's init method to initialize the base class
        super(AlexNet, self).__init__()
        
        # First Convolutional Layer with 11x11 filters, stride of 4, and 2 padding
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        
        # Max Pooling Layer with a kernel size of 3 and stride of 2
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second Convolutional Layer with 5x5 filters and 2 padding
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        
        # Third Convolutional Layer with 3x3 filters and 1 padding
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        
        # Fourth Convolutional Layer with 3x3 filters and 1 padding
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        
        # Fifth Convolutional Layer with 3x3 filters and 1 padding
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        
        # First Fully Connected Layer with 4096 output features
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        
        # Second Fully Connected Layer with 4096 output features
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        
        # Output Layer with `num_classes` output features
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        # Pass the input through the first convolutional layer and ReLU activation function
        x = self.pool(F.relu(self.conv1(x)))
        
        # Pass the output of the first layer through 
        # the second convolutional layer and ReLU activation function
        x = self.pool(F.relu(self.conv2(x)))
        
        # Pass the output of the second layer through 
        # the third convolutional layer and ReLU activation function
        x = F.relu(self.conv3(x))
        
        # Pass the output of the third layer through 
        # the fourth convolutional layer and ReLU activation function
        x = F.relu(self.conv4(x))
        
        # Pass the output of the fourth layer through 
        # the fifth convolutional layer and ReLU activation function
        x = self.pool(F.relu(self.conv5(x)))
        
        # Reshape the output to be passed through the fully connected layers
        x = x.view(-1, 256 * 6 * 6)
        
        # Pass the output through the first fully connected layer and activation function
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)    
        
        # Pass the output of the first fully connected layer through 
        # the second fully connected layer and activation function
        x = F.relu(self.fc2(x))
        
        # Pass the output of the second fully connected layer through the output layer
        x = self.fc3(x)
        
        # Return the final output
        return x





###################################################
###########        VGG16 Model         ############
###################################################
# from: https://www.geeksforgeeks.org/computer-vision/vgg-16-cnn-model/
class VGG16(CNN_BASE):
    def __init__(self, input_channels=1, H=128, W=32, num_classes=30):
        # Call the parent class's init method
        super(VGG16, self).__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate dimensions after all pooling layers
        # After 5 pooling layers with stride 2: H // 32, W // 32
        final_h = H // 32  # 128 // 32 = 4
        final_w = W // 32  # 32 // 32 = 1
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=512 * final_h * final_w, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        
        # Block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)
        
        # Block 5
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layer 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Fully Connected Layer 2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output Layer
        x = self.fc3(x)
        
        # Return the final output
        return x






####################################################
###########       ResNet18 Model        ############
####################################################
#from: https://www.geeksforgeeks.org/deep-learning/resnet18-from-scratch-using-pytorch/
class BasicBlock(CNN_BASE):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample layer for skip connection if dimensions change
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsample to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection
        out += identity
        out = F.relu(out)
        
        return out


class ResNet18(CNN_BASE):
    def __init__(self, input_channels=1, H=128, W=32, num_classes=30):
        super(ResNet18, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers (2 blocks each)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        
        # Adaptive pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(512 * BasicBlock.expansion , num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        # Create downsample layer if stride != 1 or channels change
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        # First block (may have stride and downsample)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks (stride=1, no downsample)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x




######################################################
###########     EfficientNet-B0 Model     ############
######################################################

#from : https://www.geeksforgeeks.org/computer-vision/efficientnet-architecture/
class Swish(CNN_BASE):
    """Swish activation function (x * sigmoid(x))"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(CNN_BASE):
    """Squeeze-and-Excitation block"""
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            Swish(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class MBConvBlock(CNN_BASE):
    """Mobile Inverted Residual Bottleneck Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, reduction=4):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expansion phase
        hidden_dim = in_channels * expand_ratio
        self.expand = expand_ratio != 1
        
        if self.expand:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            )
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                     stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        )
        
        # Squeeze-and-Excitation
        self.se = SEBlock(hidden_dim, reduction)
        
        # Projection phase
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand:
            x = self.expand_conv(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Squeeze-and-Excitation
        x = self.se(x)
        
        # Projection
        x = self.project_conv(x)
        
        # Skip connection
        if self.use_residual:
            x = x + identity
        
        return x


class EfficientNetB0(CNN_BASE):
    def __init__(self, input_channels=1, H=128, W=32, num_classes=30):
        super(EfficientNetB0, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        
        # MBConv blocks configuration for EfficientNet-B0
        # [expand_ratio, channels, repeats, stride, kernel_size]
        blocks_config = [
            [1, 16, 1, 1, 3],   # Stage 1
            [6, 24, 2, 2, 3],   # Stage 2
            [6, 40, 2, 2, 5],   # Stage 3
            [6, 80, 3, 2, 3],   # Stage 4
            [6, 112, 3, 1, 5],  # Stage 5
            [6, 192, 4, 2, 5],  # Stage 6
            [6, 320, 1, 1, 3],  # Stage 7
        ]
        
        # Build MBConv blocks
        self.blocks = nn.ModuleList([])
        in_channels = 32
        
        for expand_ratio, out_channels, repeats, stride, kernel_size in blocks_config:
            for i in range(repeats):
                # Only first block in each stage uses the specified stride
                s = stride if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(in_channels, out_channels, kernel_size, s, expand_ratio)
                )
                in_channels = out_channels
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish()
        )
        
        # Adaptive pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # MBConv blocks
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = self.head(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Dropout and classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x






######################################################
###########     MobileNetV2 Model     ############
######################################################
class InvertedResidual(CNN_BASE):
    """Inverted Residual Block (Bottleneck)"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion (1x1 conv)
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution (3x3)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise projection (1x1 conv) - Linear bottleneck (no activation)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(CNN_BASE):
    def __init__(self, input_channels=1, H=128, W=32, num_classes=30, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        
        # Initial convolution
        input_channel = int(32 * width_mult)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted residual blocks configuration
        # t: expansion factor, c: output channels, n: number of blocks, s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],   # 64x16
            [6, 24, 2, 2],   # 32x8
            [6, 32, 3, 2],   # 16x4
            [6, 64, 4, 2],   # 8x2
            [6, 96, 3, 1],   # 8x2
            [6, 160, 3, 2],  # 4x1
            [6, 320, 1, 1],  # 4x1
        ]
        
        # Build inverted residual blocks
        features = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        self.features = nn.Sequential(*features)
        
        # Final convolution layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        
        # Calculate final dimensions after all pooling/stride operations
        # Initial stride=2, then 5 stride=2 operations in blocks = 2^6 = 64 total reduction
        final_h = H // 64
        final_w = W // 64
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
