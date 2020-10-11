import torch

def get_kernel (kernel_name="haar", reverse=False):

    if kernel_name == 'haar':
        t1 = torch.tensor([0.7071067, 0.7071067])
        t2 = torch.tensor([-0.7071067, 0.7071067])
    elif kernel_name == 'db2':
        t1 = torch.tensor([-0.1294095, 0.22414386, 0.8365163, 0.4829629])
        t2 = torch.tensor([-0.4829629, 0.8365163, -0.22414386, -0.1294095])
    elif kernel_name == 'db3':
        t1 = torch.tensor([0.0352262, -0.0854412, -0.1350110, 0.45987750, 0.8068915, 0.3326705])
        t2 = torch.tensor([-0.3326705, 0.8068915, -0.45987750, -0.1350110, 0.0854412, 0.0352262])
    elif kernel_name == 'sym2':
        t1 = torch.tensor([-0.129409, 0.2241438, 0.8365163, 0.4829629])
        t2 = torch.tensor([-0.4829629, 0.8365163, -0.2241438, -0.129409])
    else:
        raise ValueError(f"Invalid kernel {kernel_name}")

    t1 = t1.reshape(-1, 1)
    t2 = t2.reshape(-1, 1)
    if reverse:
        t1 = t1[::-1]
        t2 = t2[::-1]

    kernel = torch.zeros ( [4, t1.shape[0], t1.shape[0]] )
    kernel[0, :, :] = t1 @ t1.T
    kernel[1] = t1 @ t2.T
    kernel[2] = t2 @ t1.T
    kernel[3] = t2 @ t2.T

    return kernel


class dwt (torch.nn.Module):

    def __init__(self, kernel_name='haar'):
        super(dwt, self).__init__()
        self.model, self.kernel_shape = self.init_depthwise(get_kernel(kernel_name).clone().detach())
        self.freeze_layers()

    def init_depthwise(self, kernel):
        kernel_size = tuple(kernel.shape[1:])
        depthwise = torch.nn.Conv2d(1, 4, kernel_size=kernel_size, stride=(2,2),
                                    padding=int((kernel_size[0]-2)/2), groups=1, bias=False)
        depthwise.weight = torch.nn.Parameter(kernel.unsqueeze(1))
        return depthwise, depthwise.weight.shape

    def freeze_layers(self):
        self.model.weight.requires_grad = False

    def forward(self, x):
        tensor_shape = x.shape
        x = self.model(x.reshape(-1, 1, tensor_shape[-2], tensor_shape[-1]))
        return x.reshape(tensor_shape[0], int(tensor_shape[1]*4), int(x.shape[-2]), int(x.shape[-1]))


class idwt (torch.nn.Module):

    def __init__(self, kernel_name='haar'):
        super(idwt, self).__init__()
        self.model, self.kernel_shape = self.init_depthwise(get_kernel(kernel_name, reverse=False).clone().detach())
        self.freeze_layers()

    def init_depthwise(self, kernel):
        kernel_size = tuple(kernel.shape[1:])
        depthwise = torch.nn.ConvTranspose2d(4, 1, kernel_size=kernel_size, stride=(2,2),
                                             padding=int((kernel_size[0]-2)/2), groups=1, bias=False)
        depthwise.weight = torch.nn.Parameter(kernel.unsqueeze(1))
        return depthwise, depthwise.weight.shape

    def freeze_layers(self):
        self.model.weight.requires_grad = False

    def forward(self, x):

        if x.shape[1] % 4 != 0:
            raise ValueError(f'Number of channels is {x.shape[1]}, but it must be divisible by 4')

        tensor_shape = x.shape
        x = x.reshape(-1, 4, tensor_shape[-2], tensor_shape[-1])
        x = self.model(x)
        return x.reshape(tensor_shape[0], -1, int(x.shape[-2]), int(x.shape[-1]))















