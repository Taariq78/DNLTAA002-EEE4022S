class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

#'''unet_start'''
#        '''encoder'''
        self.conv_0=nn.Conv2d(2,16,kernel_size=(4,4), stride= (2,2), padding=(1,1))
        self.conv_1=nn.Conv2d(16,32,kernel_size=(4,4), stride= (2,2), padding=(1,1))
        self.conv_2=nn.Conv2d(32,32,kernel_size=(4,4), stride= (2,2), padding=(1,1))
        self.conv_3=nn.Conv2d(32,32,kernel_size=(4,4), stride= (2,2), padding=(1,1))

        self.activation=nn.LeakyReLU(negative_slope=0.2)
        

#        '''decoder'''
        self.conv_0_dec=nn.Conv2d(32,32,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv_1_dec=nn.Conv2d(64,32,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv_2_dec=nn.Conv2d(64,32,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv_3_dec=nn.Conv2d(48,32,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv_4_dec=nn.Conv2d(32,32,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv_5_dec=nn.Conv2d(34,16,kernel_size=(3,3),stride=(1,1),padding=(1,1))

#        '''upsample'''
        self.upsample=nn.Upsample(scale_factor=2.0, mode='nearest')
#'''unet_end'''

        self.conv_vm2=nn.Conv2d(16,16,kernel_size=(3,3),stride=(1,1),padding=(1,1))

        self.conv_flow=nn.Conv2d(16,2,kernel_size=(3,3),stride=(1,1),padding=(1,1))

        self.spatial_transform = SpatialTransformer((256,256))

    def Unet(self, image_pair):
#          '''encode'''
      x1=self.conv_0(image_pair)
      x1=self.activation(x1)

      x2=self.conv_1(x1)
      x2=self.activation(x2)

      x3=self.conv_2(x2)
      x3=self.activation(x3)

      x4=self.conv_3(x3)
      x4=self.activation(x4)

#          '''decode'''
      y=self.conv_0_dec(x4)
      y=self.activation(y)
      y=self.upsample(y)
      y=torch.cat([y,x3],1)

      y=self.conv_1_dec(y)
      y=self.activation(y)
      y=self.upsample(y)
      y=torch.cat([y,x2],1)

      y=self.conv_2_dec(y)
      y=self.activation(y)
      y=self.upsample(y)
      y=torch.cat([y,x1],1)

      y=self.conv_3_dec(y)
      y=self.activation(y)          
      y=self.conv_4_dec(y)
      y=self.activation(y)

      y=self.upsample(y)
      y=torch.cat([y,image_pair],1)
      y=self.conv_5_dec(y)
      y=self.activation(y)

      y=self.conv_vm2(y)
      y=self.activation(y)

      return y

    def forward(self,mov,fix):
      z = torch.cat([mov, fix], dim=1)
      z = self.Unet(z)
      #print(z.shape)
      flow = self.conv_flow(z)
      #print(flow.shape)

      z=self.spatial_transform(mov,flow)   

      return z, flow  



model=Net().to(device)
