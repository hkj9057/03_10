import torch
from my_dataset import NkDataSet
from deep3_03_17 import Cnn_Model

batch_size = 2
def train(my_dataset_loader,model,criterion,optimizer,epoch):

    model.train()

    for i, data in enumerate(my_dataset_loader, 0):
        # Forward pass: Compute predicted y by passing x to the model

        # fc 구조 이기 때문에 일렬로 피는 작업이 필요하다.
        images, label = data

        #그냥 images를 하면 에러가 난다. 데이터 shape이 일치하지 않아서
        y_pred = model(images)

        #Compute and print loss
        loss = criterion(y_pred, label)

        #print(epoch, loss.item())

        #Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(my_dataset_loader, model, criterion, epoch):
    model.eval()
    for i, data in enumerate(my_dataset_loader, 0):
        #Forwad pass: Compute predicted y by passing x to the model

        # fc 구조 이기 때문에 일렬로 쫙피는 작업이 필요하다.
        images, label = data

        # 그냥 images를 하면 에러가 난다. 데이터 shape이 일치하지 않아서
        y_pred = model(images)

        # Compute and print loss
        loss = criterion(y_pred, label)

        for i in range((batch_size)):

            if(label[i].item() == 0 ):
                print("고양이, epcoh : %f, loss: %f"%(epoch,loss.item()))

            else:
                print("강아, epcoh : %f, loss: %f" % (epoch, loss.item()))
       # print(epoch, loss.item())

#Data Load
csv_path = './test.csv'

custom_dataset = NkDataSet(csv_path)

my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size=2,
                                                shuffle=True,
                                                num_workers=1)

t_csv_path = './test2.csv'

t_custom_dataset = NkDataSet(t_csv_path)

t_my_dataset_loader = torch.utils.data.DataLoader(dataset=t_custom_dataset,
                                                  batch_size=2,
                                                  shuffle=False,
                                                  num_workers=1)


#test_data_set 만들어 줘야 한다.
#Model_Load
#imput , hidden, output size


model = Cnn_Model()

#CrossEntropyLoss 를 사용
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for epoch in range(500):

    print('epoch', epoch)

    train(my_dataset_loader,model,criterion,optimizer,epoch)
    test(t_my_dataset_loader,model,criterion,epoch)