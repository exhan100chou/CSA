import torch
import torch.nn as nn
import copy
from .modules import Resblock, MaskAttentionSampler    
#try:
#    from torch.hub import load_state_dict_from_url  # noqa: 401
#except ImportError:
#    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401


__all__ = ['DTJSCC','ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    "resnet50": 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
def client_update_sgd(model_encode, model_decode, dataloader_client, optimizer_en, optimizer_de, criterion, mod, args, local_epochs=1):
    """
    Perform local training for a client and return gradients for server aggregation.
    
    Args:
    - model_encode: Client's encoder model.
    - model_decode: Client's decoder model.
    - dataloader_client: DataLoader for the client's data.
    - optimizer_en: Optimizer for the encoder model (AdamW).
    - optimizer_de: Optimizer for the decoder model (AdamW).
    - criterion: Loss function.
    - mod: Modulation object.
    - args: Arguments.
    - local_epochs: Number of local epochs for client training.
    
    Returns:
    - Best encoder model state.
    - Best decoder model state.
    - Best Top-1 accuracy.
    - Encoder gradients.
    - Decoder gradients.
    """
    model_encode.train()
    model_decode.train()

    best_acc1 = 0.0
    for epoch in range(local_epochs):
        for imgs, labs in dataloader_client:
            imgs = imgs.to(args.device if torch.cuda.is_available() else "cpu")
            labs = labs.to(args.device if torch.cuda.is_available() else "cpu")
            
            # Forward pass
            en_X, former_shape, dist = model_encode(imgs, mod=mod)
            features = model_decode(en_X, former_shape)
            outs = model_decode.head(features)
            
            loss, _, _ = criterion(dist, outs, labs)

            # Backpropagation
            optimizer_en.zero_grad()
            optimizer_de.zero_grad()
            loss.backward()
            
            # Collect gradients before updating the client models
            encode_grads = {name: param.grad.clone() for name, param in model_encode.named_parameters() if param.requires_grad}
            decode_grads = {name: param.grad.clone() for name, param in model_decode.named_parameters() if param.requires_grad}
            
            # Update the client models using AdamW
            optimizer_en.step()
            optimizer_de.step()

        # Evaluate the model on the client data
        acc1, _, _ = eval_test(dataloader_client, model_encode, model_decode, mod, args)
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_model_encode_state = copy.deepcopy(model_encode.state_dict())
            best_model_decode_state = copy.deepcopy(model_decode.state_dict())
            
    return best_model_encode_state, best_model_decode_state, best_acc1, encode_grads, decode_grads


def server_aggregate_sgd(global_model, client_gradients, lr):
    """
    Aggregates gradients from clients and updates the global model using SGD.
    
    Args:
    - global_model: The global model whose parameters will be updated.
    - client_gradients: List of client gradients (state_dicts).
    - lr: Learning rate for the global model's SGD optimizer.

    Returns:
    - Updated global model.
    """
    # Initialize the global model gradients to zero
    global_gradients = {key: torch.zeros_like(param) for key, param in global_model.state_dict().items()}

    # Sum the client gradients for each parameter
    for client_gradient in client_gradients:
        for key in global_gradients:
            # Skip BatchNorm running statistics (running_mean, running_var) and other non-gradient parameters
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                continue
            # Aggregate gradients from clients
            # Check if the key exists in client gradients
            if key in client_gradient:
                global_gradients[key] += client_gradient[key]  # Aggregate gradients
            else:
                print(f"Warning: Key '{key}' not found in client gradients.")

    # Perform an SGD step on the global model
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if param.requires_grad:
                param -= lr * global_gradients[name]  # SGD update step

    return global_model
def server_aggregate_avg1(global_model, client_models, client_accs):
    """
    Aggregates the model weights from different clients and updates the global model.
    Args:
    - global_model: The global model that will be updated.
    - client_models: A list of state_dicts from clients.
    """
    # Normalize accuracies to use as weights for aggregation
    acc_sum = sum(client_accs)
    if acc_sum == 0:
        acc_sum = 1  # Prevent division by zero

    weights = [acc / acc_sum for acc in client_accs]    
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i][key].float() * weights[i] for i in range(len(client_models))], 0).sum(0)

    global_model.load_state_dict(global_dict)
    return global_model
def server_aggregate_avg(global_model_encode, global_model_decode, client_models, client_accs):
    """
    Aggregates the model weights from different clients based on their Top-1 accuracy.
    Args:
    - global_model_encode: The global encoder model to be updated.
    - global_model_decode: The global decoder model to be updated.
    - client_models: A list of tuples containing client models (encode and decode state dicts).
    - client_accs: A list of Top-1 accuracies for the respective client models.
    """

    # Normalize accuracies to use as weights for aggregation
    acc_sum = sum(client_accs)
    if acc_sum == 0:
        acc_sum = 1  # Prevent division by zero

    weights = [acc / acc_sum for acc in client_accs]

    global_encode_dict = global_model_encode.state_dict()
    global_decode_dict = global_model_decode.state_dict()

    # Average the client model parameters weighted by accuracy
    for key in global_encode_dict.keys():
        global_encode_dict[key] = torch.stack([client_models[i][0][key].float() * weights[i] for i in range(len(client_models))], 0).sum(0)

    for key in global_decode_dict.keys():
        global_decode_dict[key] = torch.stack([client_models[i][1][key].float() * weights[i] for i in range(len(client_models))], 0).sum(0)

    # Load the aggregated weights into the global models
    global_model_encode.load_state_dict(global_encode_dict)
    global_model_decode.load_state_dict(global_decode_dict)

    return global_model_encode, global_model_decode
def server_aggregate(global_model_encode, global_model_decode, client_models, client_accs):
    """
    Server selects the client with the highest Top-1 accuracy for aggregation.
    Args:
    - global_model_encode: Global encoder model.
    - global_model_decode: Global decoder model.
    - client_models: List of tuples (model_encode_state, model_decode_state) for each client.
    - client_accs: List of Top-1 accuracy scores for each client.
    
    Returns:
    - Global models updated with the best client's parameters.
    """
    # Find the index of the client with the highest Top-1 accuracy
    best_client_idx = client_accs.index(max(client_accs))
    
    # Get the state dicts (parameters) of the best client's models
    best_encode_state, best_decode_state = client_models[best_client_idx]
    
    # Load the best client's state dict into the global model
    global_model_encode.load_state_dict(best_encode_state)
    global_model_decode.load_state_dict(best_decode_state)
    
    print(f"Selected client {best_client_idx} with the highest Top-1 accuracy for model aggregation.")
    
    return global_model_encode, global_model_decode

class DTJSCC(nn.Module):
    def __init__(self, in_channels, latent_channels, out_classes, num_embeddings=400):
        super().__init__()
        self.latent_d = latent_channels
        self.prep = nn.Sequential(
                    nn.Conv2d(in_channels, latent_channels//8,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//8),
                    nn.ReLU()
                    )
        self.layer1 = nn.Sequential(
                    nn.Conv2d(latent_channels//8,latent_channels//4, kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//4),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )
        self.layer2 = nn.Sequential(
                    nn.Conv2d(latent_channels//4,latent_channels//2,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels//2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                    )
        self.layer3 = nn.Sequential(
                    nn.Conv2d(latent_channels//2,latent_channels,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(latent_channels),
                    nn.ReLU(),
                    # nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode = False)
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )
        self.encoder = nn.Sequential(
            self.prep,                    # 64x32x32
            self.layer1,                  # 128x16x16
            Resblock(latent_channels//4), # 128x16x16
            self.layer2,                  # 256x8x8
            self.layer3,                  # 512x4x4
            # Resblock(latent_channels),    # 512x4x4
            Resblock(latent_channels)     # 512x4x4
        )
        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)
        self.decoder = nn.Sequential(
            Resblock(latent_channels),
            Resblock(latent_channels),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),      # 512x1x1
            nn.Flatten(),                 # 512
            nn.Linear(latent_channels, out_classes)
        )

    def encode(self, X):
        en_X = self.encoder(X)
        former_shape = en_X.shape
        en_X = en_X.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_d)
        return en_X, former_shape
    
    def decode(self, features, former_shape):
        b, c , h, w = former_shape
        features = features.view(b, h, w, c)
        features = features.permute(0,3,1,2).contiguous()
        tilde_X = self.decoder (features)
        return tilde_X 

    def forward(self, X, mod=None):
        out, former_shape = self.encode(X)
        out, dist = self.sampler(out, mod=mod)
        tilde_X = self.decode(out, former_shape)

        return tilde_X, dist   
import torch.nn as nn

# Encoder module
class DTJSCC_encode(nn.Module):
    def __init__(self, in_channels, latent_channels, num_embeddings=400):
        super().__init__()

        self.feature_num = latent_channels
        self.latent_d = latent_channels
        # Encoder part
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels//8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//8),
            nn.ReLU()
        )
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(latent_channels//8, latent_channels//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(latent_channels//4, latent_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(latent_channels//2, latent_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.encoder = nn.Sequential(
            self.prep,                    # 64x32x32
            self.layer1,                  # 128x16x16
            Resblock(latent_channels//4), # 128x16x16
            self.layer2,                  # 256x8x8
            self.layer3,                  # 512x4x4
            Resblock(latent_channels)     # 512x4x4
        )

        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)

    def encode(self, X):
        en_X = self.encoder(X)
        former_shape = en_X.shape
        en_X = en_X.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_d)
        return en_X, former_shape

    def forward(self, X, mod=None):
        encoded, former_shape = self.encode(X)
        encoded, dist = self.sampler(encoded, mod=mod)
        return encoded, former_shape, dist


# Decoder module
class DTJSCC_decode(nn.Module):
    def __init__(self, latent_channels, out_classes):
        super().__init__()
        # Decoder part
        self.decoder = nn.Sequential(
            Resblock(latent_channels),
            Resblock(latent_channels),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),      # 512x1x1
            nn.Flatten()                 # 512
            #nn.Linear(latent_channels, out_classes)
        )
        self.head = nn.Linear(latent_channels , out_classes)   
    def decode(self, features, former_shape):
        b, c, h, w = former_shape
        features = features.view(b, h, w, c)
        features = features.permute(0, 3, 1, 2).contiguous()
        features = self.decoder(features)
        return features
    
    def forward(self, features, former_shape):
        return self.decode(features, former_shape)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.feature_num = 512 * block.expansion
        self.num_classes = num_classes

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512 * block.expansion, num_classes)
       # self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        # x = self.head(features)
        # x = torch.nn.functional.linear(features, weight=self.head.weight.detach(), bias=self.head.bias.detach())

        return features

    def forward(self, x):
        return self._forward_impl(x)
    

def _resnet(arch, block, layers, pretrained, progress, pretrained_dir, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        state_dict = torch.load(pretrained_dir, map_location='cpu')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, pretrained_dir=None, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, pretrained_dir,
                   **kwargs)


def resnet34(pretrained=False, progress=True, pretrained_dir=None, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, pretrained_dir,
                   **kwargs)


def resnet50(pretrained=False, progress=True, pretrained_dir=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, pretrained_dir,
                   **kwargs)


def resnet101(pretrained=False, progress=True, pretrained_dir=None, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, pretrained_dir,
                   **kwargs)


def resnet152(pretrained=False, progress=True, pretrained_dir=None, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, pretrained_dir,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
