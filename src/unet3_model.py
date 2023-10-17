import torch
from torch import nn

class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels, normF=nn.BatchNorm2d):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels = 16, kernel_size = 3,padding=1),                         
                        nn.ReLU(),
                        normF(16),
                        nn.MaxPool2d(2,2),                                                              
                        nn.Conv2d(16,32,3,padding=1),                           
                        nn.ReLU(),
                        normF(32),
                        nn.Conv2d(32,64,3,padding=1),                                                    
                        nn.ReLU(),
                        normF(64),
                        nn.MaxPool2d(2,2),                                                               
                        nn.Conv2d(64,128,3,padding=1),                          
                        nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(128,64,3,stride = 2,padding = 1, output_padding = 1),            
                        nn.ReLU(),
                        normF(64),
                        nn.ConvTranspose2d(64,16,3,1,1),                                                
                        nn.ReLU(),
                        normF(16),
                        nn.ConvTranspose2d(in_channels=16,out_channels=in_channels,kernel_size = 3, stride = 2,padding = 1, output_padding = 1)                     
        )

                
    def forward(self,x):
        encoded_img = self.encoder(x)
        reconstructed_img = self.decoder(encoded_img)
        return reconstructed_img


    
    
class DownSampleLeadTime(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_dropout, use_batchnorm, pool_fn = nn.MaxPool2d, normF=nn.BatchNorm2d):
        super(DownSampleLeadTime, self).__init__()
        self.pool = pool_fn(2)
        self.conv1 = ConvBNActTime(in_channels, time_dim, out_channels, use_dropout=use_dropout, use_batchnorm=use_batchnorm, normF=normF)
        self.conv2 = ConvBNActTime(out_channels, time_dim, out_channels, use_dropout=use_dropout, use_batchnorm=use_batchnorm, normF=normF)
        
    def forward(self, x, t):
        return self.conv2(self.conv1(self.pool(x), t), t)

    
class UpSampleLeadTime(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_dropout, use_batchnorm, normF=nn.BatchNorm2d):
        super(UpSampleLeadTime, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv1 = ConvBNActTime(out_channels + out_channels, time_dim, out_channels, use_dropout=use_dropout, use_batchnorm=use_batchnorm, normF=normF)
        self.conv2 = ConvBNActTime(out_channels, time_dim, out_channels, use_dropout=use_dropout, use_batchnorm=use_batchnorm, normF=normF)
        
    def forward(self, x, snapshot, t):
        x = self.upsample(x)
        x = torch.cat((snapshot, x), dim=-3)
        x = self.conv2(self.conv1(x, t), t)
        return x
    
class ConvBNActTime(nn.Module):
    def __init__(self, img_dim, time_dim, out_channels, use_batchnorm = False, use_dropout = False, act_fn = nn.ReLU, normF = nn.BatchNorm2d):
        super(ConvBNActTime, self).__init__()
        self.img_conv = nn.Conv2d(img_dim, out_channels, padding=1, kernel_size=3)
        self.time_conv = nn.Conv2d(time_dim, out_channels, padding=1, kernel_size=3, bias=False)
        
        self.bn = normF(out_channels)
        self.act = act_fn()
        
    def forward(self, img, time=None):
        x = self.img_conv(img)
        if time is not None:
            f_time = self.time_conv.weight[:, time].transpose(0, 1).sum(dim=[-2,-1], keepdim=True)
            x += f_time
        x = self.bn(x)
        return self.act(x)
    

    
class UNetV2LeadTime(nn.Module):
    def __init__(self, initial_channels = 32, out_channels = 1, img_dim = 7, time_dim = 0, use_dropout = False, use_batchnorm = False, use_batchnorm_at_first = False, normF = nn.BatchNorm2d):
        super(UNetV2LeadTime, self).__init__()
        self.initial_conv = ConvBNActTime(img_dim, time_dim, initial_channels, use_dropout = use_dropout, use_batchnorm = use_batchnorm_at_first, normF = normF)
        self.second_conv = ConvBNActTime(initial_channels, time_dim, initial_channels, use_dropout = use_dropout, use_batchnorm = use_batchnorm_at_first, normF = normF)
        self.down_layers = nn.ModuleList([DownSampleLeadTime((initial_channels << i),
                                                     (initial_channels << (i+1)),
                                                             time_dim,
                                                     use_dropout = use_dropout,
                                                     use_batchnorm = use_batchnorm, normF = normF) for i in range(5)])
        
        self.up_layers = nn.ModuleList([UpSampleLeadTime((initial_channels << (5-i)),
                                                 (initial_channels << (4-i)),
                                                         time_dim,
                                                 use_dropout = use_dropout,
                                                 use_batchnorm = use_batchnorm, normF = normF) for i in range(5)])
        
        
    def forward(self, img, time = None):
        x = self.second_conv(self.initial_conv(img, time))
        self.snapshots = []
        for i, layer in enumerate(self.down_layers):
            self.snapshots.append(x)
            x = layer(x, time)
        for i, layer in enumerate(self.up_layers):
            x = layer(x, self.snapshots[-(i+1)], time)
        del self.snapshots
        return x
    


class DenoisingMultiUNet(nn.Module):
    def __init__(self, initial_channels = 32, out_channels = 1, img_dim = 7, time_dim = 0, use_dropout = False, use_batchnorm = False, use_batchnorm_at_first = False, resolution=1.):
        super(DenoisingMultiUNet, self).__init__()
        if resolution == 1.:
            _normF = nn.BatchNorm2d
        elif resolution == 0.5:
            _normF = nn.InstanceNorm2d
        
        self.unet = UNetV2LeadTime(img_dim = img_dim, time_dim = time_dim, initial_channels = initial_channels, out_channels = out_channels, use_dropout = False, use_batchnorm = False, use_batchnorm_at_first = False, normF = _normF)
        self.last_conv = nn.Conv2d(initial_channels, out_channels, padding=1, kernel_size=3)
        self.denosing_convs = ConvAutoEncoder(img_dim, normF = _normF)
        self.apply(self.weight_init)
        self.n_classes = out_channels
        
    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, img, time):
        img = self.denosing_convs(img)
        x = self.unet(img, time)
        return self.last_conv(x).permute(0, 2, 3, 1)#, x

    