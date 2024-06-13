
import torch
from torch import nn
from torchvision import transforms


transforms = nn.Sequential(transforms.Normalize(
    mean=[0.48235, 0.45882, 0.40784],
    std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
    ))

class SegNet(nn.Module):
    def __init__(self, output_size=32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # ENCODER LAYERS (Vgg-16)

        self.enc_conv_0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_conv_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # max_pooling -> skip_connection
        self.enc_conv_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # max_pooling -> skip_connection
        self.enc_conv_4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_conv_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.enc_conv_6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # max_pooling -> skip_connection
        self.enc_conv_7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc_conv_8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_conv_9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # max_pooling -> skip_connection
        self.enc_conv_10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_conv_11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_conv_12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # max_pooling


        # DECODER LAYERS

        # upsampling
        self.dec_conv_0 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_conv_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_conv_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # concatenation -> upsampling
        self.dec_conv_3 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec_conv_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_conv_5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        # concatenation -> upsampling
        self.dec_conv_6 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_conv_7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_conv_8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # concatenation -> upsampling
        self.dec_conv_9 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv_10 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # concatenation -> upsampling
        self.dec_conv_11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv_12 = nn.Conv2d(64, output_size, kernel_size=3, padding=1)


        # OTHER LAYERS
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.batch_norm32 = nn.BatchNorm2d(32)
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.batch_norm256 = nn.BatchNorm2d(256)
        self.batch_norm512 = nn.BatchNorm2d(512)

    def conv_block(self, x, *args):
        """
            Does sucessives convolutions with relu activation and batch normalization
            for every convolution in arguments
        """
        for conv_layer in args:
            x = conv_layer(x)
            x = self.relu(x)
            if x.shape[1] == 32:
                x = self.batch_norm32(x)
            elif x.shape[1] == 64:
                x = self.batch_norm64(x)
            elif x.shape[1] == 128:
                x = self.batch_norm128(x)
            elif x.shape[1] == 256:
                x = self.batch_norm256(x)
            elif x.shape[1] == 512:
                x = self.batch_norm512(x)
        
        return x

    def forward(self, x):
        
        # ENCODING

        skip_0 = self.conv_block(x, self.enc_conv_0, self.enc_conv_1)
        skip_0 = self.pool(skip_0)

        skip_1 = self.conv_block(skip_0, self.enc_conv_2, self.enc_conv_3)
        skip_1 = self.pool(skip_1)

        skip_2 = self.conv_block(skip_1, self.enc_conv_4, self.enc_conv_5, self.enc_conv_6)
        skip_2 = self.pool(skip_2)
        
        skip_3 = self.conv_block(skip_2, self.enc_conv_7, self.enc_conv_8, self.enc_conv_9)
        skip_3 = self.pool(skip_3)
        
        bottleneck = self.conv_block(skip_3, self.enc_conv_10, self.enc_conv_11, self.enc_conv_12)
        bottleneck = self.pool(bottleneck)


        # DECODING

        upsampled = self.upsampling(bottleneck)
        upsampled = self.conv_block(upsampled, self.dec_conv_0, self.dec_conv_1, self.dec_conv_2)

        upsampled = torch.cat((upsampled, skip_3), dim=1)
        upsampled = self.upsampling(upsampled)
        upsampled = self.conv_block(upsampled, self.dec_conv_3, self.dec_conv_4, self.dec_conv_5)

        upsampled = torch.cat((upsampled, skip_2), dim=1)
        upsampled = self.upsampling(upsampled)
        upsampled = self.conv_block(upsampled, self.dec_conv_6, self.dec_conv_7, self.dec_conv_8)

        upsampled = torch.cat((upsampled, skip_1), dim=1)
        upsampled = self.upsampling(upsampled)
        upsampled = self.conv_block(upsampled, self.dec_conv_9, self.dec_conv_10)

        upsampled = torch.cat((upsampled, skip_0), dim=1)
        upsampled = self.upsampling(upsampled)
        upsampled = self.conv_block(upsampled, self.dec_conv_11, self.dec_conv_12)

        return self.softmax(upsampled)
    

if __name__ == "__main__":
    model = SegNet()
    x = torch.rand((1, 3, 224, 224))
    print(x.shape)
    print(model(x).shape)
    print(torch.sum(model(x), dim=1))