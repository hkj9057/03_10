from torch.utils.data.dataset import Dataset
import torch
from deep3_03_17 import Cnn_Model
from my_dataset import NkDataSet


#Data Load
csv_path = './test.csv'

custom_dataset = NkDataSet(csv_path)

my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size=2,
                                                shuffle=True,
                                                num_workers=1)

model = Cnn_Model()

#CrossEntropyLoss를 사용
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


for t in range(500):

    for i,data in enumerate(my_dataset_loader,0):
        # Forward pass: Compute predicted y by passing x to the model

        #fc 구조 이기 때문에 일렬로 쫙피는 작업이 필요하다.

        images,label = data


        #그냥 images를 하면 에러가 난다. 데이터 shape이 일치하지 않아서
        y_pred = model(images)

        # Compute and print loss
        loss = criterion(y_pred,label.long())

        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()