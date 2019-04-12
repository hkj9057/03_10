import torch

"""
CODE 설명
cnn 모델을 class형으로 만듬
O=W-K+2PS+1
where 0 is the output height/length, W is the input height/length, K is filter size, P is the padding, and S is the stride.
"""
batch_size = 2
class Cnn_Model(torch.nn.Module):

    def __init__(self):

        super(Cnn_Model,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,20,kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(20,2,kernel_size=3)
        #이 부분을 고쳐줘야 한다.
        self.conv3 = torch.nn.Linear(302,2)


    def forward(self, x):

        x = self.conv1(x)

        print("conv1 ",x.size())
        x = self.relu(x)
        x = self.conv2(x)

        #print("conv2 ".x.size())
        x = self.relu(x)
        x = x.view(batch_size,-1)
        x = self.conv3(x)

        #print("conv3 ", x)

        # print("last x size ".x.size())

        x = x.view(batch_size,2)

        return x
