import torch
import torchvision
import numpy as np
from tqdm import trange
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Subset
from sklearn.model_selection import train_test_split
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

face = np.load("faces.npy")
ages = np.load("ages.npy")
face = np.expand_dims(np.array(face), axis=1)
face = face.repeat(3, 1)/225
face = torch.tensor(face).to(device)
ages = torch.tensor(ages).to(device)
faces = torch.zeros(7500,3,224,224).to(device)
transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToPILImage(),
                                torchvision.transforms.Resize((224,224)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
                              ])

for i in range(len(ages)):
    faces[i] = transform(face[i])

class FaceLoader(torch.utils.data.Dataset):
    def __init__(self, faces, transform=None, time_num=1):
        super(FaceLoader, self).__init__()
        self.faces = faces
        self.transform = transform
        self.time_num = time_num

    def __len__(self):
        return self.faces.size(0)

    def __fetitem__(self, item):
        # for i in range(len(self.faces))
        if self.transofrm:
            face = self.transform(self.faces[item])
        return face
        
class AgeLoader(torch.utils.data.Dataset):
    def __init__(self, ages, time_num=1):
        super(AgeLoader, self).__init__()

        self.ages = ages
        self.time_num = time_num

    def __len__(self):
        return self.ages.size(0)

    def __fetitem__(self, item):
        age = self.ages[item]
        return age


face = FaceLoader(faces=faces, transform=transform)
age = AgeLoader(ages=ages)

def split_tensor(tensor, ratio):
    length = len(tensor)
    test_num = int(length*ratio)
    train, test = random_split(tensor, lengths=[length-test_num,test_num],generator=torch.Generator().manual_seed(42))
    return train, test

N = len(ages)
test_ratio = 0.2
valid_ratio = 0.2
test_N = int(N*test_ratio)

face_train, face_test = split_tensor(faces, test_ratio)
age_train, age_test = split_tensor(ages, test_ratio)

face_train, face_valid = split_tensor(face_train, valid_ratio)
age_train, age_valid = split_tensor(age_train, valid_ratio)

batchTrain = 3
batchValid = 2
batchTest = 2
params_train = {'batch_size':batchTrain,"shuffle":True,'num_workers':0}
params_valid = {'batch_size':batchValid,"shuffle":False,'num_workers':0}
params_test = {'batch_size':batchTest,"shuffle":False,'num_workers':0}

train_faces = DataLoader(face_train, **params_train)
train_ages = DataLoader(age_train, **params_train)

valid_faces = DataLoader(face_valid, **params_valid)
valid_ages = DataLoader(age_valid, **params_valid)

test_faces = DataLoader(face_test, **params_test)
test_ages = DataLoader(age_test, **params_test)


resnet = torchvision.models.resnet50(weights='DEFAULT').to(device)
#resnet.fc = nn.Linear(2048,1).to(device)
# print(resnet)

class FFNNnet(nn.Module):
  def __init__(self):
    super(FFNNnet,self).__init__()
    self.fc = nn.Linear(2048,1)

    self._initialize_weight()
  def _initialize_weight(self):
    nn.init.xavier_normal_(self.fc.weight)
    nn.init.normal_(self.fc.bias)
 
  def forward(self,x):
    relu = nn.ReLU()
    x = relu(self.fc(x))

    return x

ffnn = FFNNnet().to(device)

learning_rate = 0.001
beta = (0.995,0.998)
eps = 1e-10
resnet.fc = ffnn
optimizer = torch.optim.Adam(ffnn.parameters(), lr=learning_rate, betas=beta, eps=eps)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

criterion = RMSELoss()

train_loss = []
valid_loss = []
best_loss = 1000
Epoch = 100

debug_step = 10

#########################################################
###########                  Training
#########################################################

for epoch in trange(Epoch):
    resnet.train()
    n = 0
    dynamic_loss = 0
    loss_train = 0
    for i, (faces_, ages_) in enumerate(zip(train_faces,train_ages),0):
        faces = faces_.to(dtype=torch.float32).to(device)
        ages = ages_.to(dtype=torch.float32).to(device)
        ages = ages.squeeze()
        output = resnet(faces).squeeze()
        # output2 = ffnn(output1)
        loss = criterion(output,ages)
        
        k = faces.shape[0]
        n += k
        dynamic_loss += torch.pow(loss,2)* k

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i > debug_step:
        #   break
    loss_train = torch.sqrt(dynamic_loss/n).item()
    train_loss.append(loss_train)

    n = 0
    dynamic_loss_v = 0
    loss_val = 0
    resnet.eval()
    with torch.no_grad():
        for i_val, (faces_v_, ages_v_) in enumerate(zip(valid_faces,valid_ages),0):
            faces_v = faces_v_.to(dtype=torch.float32).to(device)
            ages_v = ages_v_.to(dtype=torch.float32).to(device)
            output_v = resnet(faces_v).squeeze()
            # output_v2 = ffnn(output_v1)
            loss_v = criterion(output_v,ages_v).squeeze()
            k = faces_v.shape[0]
            n += k
            dynamic_loss_v += torch.pow(loss_v,2)*k
            loss_val = torch.sqrt(dynamic_loss_v/n).item()
            valid_loss.append(loss_val)

    print("Epoch: ", epoch)
    print("Training_loss: {:.2f}, Validation_loss: {:.2f}".format(loss_train, loss_val))

    if loss_val < best_loss:
        best_loss = loss_val
        best_model = copy.deepcopy(ffnn)
        torch.save(best_model.state_dict(),'HW5_Q2_ffnn.pt')



#######################################################################
###########                  Testing                        ###########
################################################################################################
###########  ffor testing, uncomment the testing coded and comment the training code  ##########
################################################################################################

# def test(test_faces, test_ages, model):
#     model.eval()
#     n = 0
#     dynamic_loss_t = 0
#     for i_test, (faces_t_, ages_t_) in enumerate(zip(test_faces,test_ages),0):
#         faces_t = faces_t_.to(dtype=torch.float32).to(device)
#         ages_t = ages_t_.to(dtype=torch.float32).to(device)
#         output_t = model(faces_t).squeeze()
#         # output_t2 = model(output_t1)
#         loss_t = criterion(output_t,ages_t).squeeze()

#         k = faces_t.shape[0]
#         n += k
#         dynamic_loss_t += torch.pow(loss_t,2)*k
    
#     test_loss = torch.sqrt(dynamic_loss_t/n).item()
#     return test_loss

# ffnn.load_state_dict(torch.load('HW5_Q2_ffnn1.pt'))
# resnet.fc = ffnn
# test_loss = test(test_faces, test_ages, resnet)
# print("Testing RMSE loss: {:.2f}".format(test_loss))