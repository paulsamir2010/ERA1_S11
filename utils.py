import numpy as np
import matplotlib.pyplot as plt
import torch
import random
#from dataset import train_mean, train_std
from torchvision.transforms import Normalize

from torchvision import datasets
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def wrong_predictions(model, test_loader, device):
    wrong_images=[]
    wrong_label=[]
    correct_label=[]
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)        
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])
            wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))
        print(f'Total wrong predictions are {len(wrong_predictions)}')

    return wrong_predictions

def plot_misclassified(wrong_predictions, mean, std, num_img):
    fig = plt.figure(figsize=(15,12))
    fig.tight_layout()
    for i, (img, pred, correct) in enumerate(wrong_predictions[:num_img]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        for j in range(img.shape[0]):
            img[j] = (img[j]*std[j])+mean[j]

        img = np.transpose(img, (1, 2, 0)) 
        ax = fig.add_subplot(5, 5, i+1)
        fig.subplots_adjust(hspace=.5)
        ax.axis('off')
        #class_names,_ = get_classes()

        ax.set_title(f'\nActual : {classes[target.item()]}\nPredicted : {classes[pred.item()]}',fontsize=10)  
        ax.imshow(img)  

    plt.show()

# Calculate mean and standard deviation for training dataset
train_data = datasets.CIFAR10('./data', download=True, train=True)

# use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
print(x.shape)
# calculate the mean and std along the (0, 1) axes
train_mean = np.mean(x, axis=(0, 1))/255
train_std = np.std(x, axis=(0, 1))/255

def get_incorrect_preds(model, test_dataloader):
  incorrect_examples = []
  pred_wrong = []
  true_wrong = []

  model.eval()
  for data,target in test_dataloader:
    data , target = data.cuda(), target.cuda()
    output = model(data)
    _, preds = torch.max(output,1)
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()
    preds = np.reshape(preds,(len(preds),1))
    target = np.reshape(target,(len(preds),1))
    data = data.cpu().numpy()
    for i in range(len(preds)):
        if(preds[i]!=target[i]):
            pred_wrong.append(preds[i])
            true_wrong.append(target[i])
            incorrect_examples.append(data[i])

  return true_wrong, incorrect_examples, pred_wrong

def plot_incorrect_preds(true,ima,pred,n_figures = 10):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']
    
    denorm = Normalize((-train_mean / train_std).tolist(), (1.0 / train_std).tolist())
    print('Classes in order Actual and Predicted')
    n_row = int(n_figures/5)
    fig,axes = plt.subplots(figsize=(12, 3), nrows = n_row, ncols=5)
    plt.subplots_adjust(hspace=1)
    for ax in axes.flatten():
        a = random.randint(0,len(true)-1)
        image,correct,wrong = ima[a],true[a],pred[a]
        image = torch.from_numpy(image)
        image = denorm(image)*255
        image = image.permute(2, 1, 0) # from NHWC to NCHW
        correct = int(correct)
        wrong = int(wrong)
        image = image.squeeze().numpy().astype(np.uint8)
        im = ax.imshow(image) #, interpolation='nearest')
        ax.set_title(f'A: {labels[correct]} , P: {labels[wrong]}', fontsize = 8)
        ax.axis('off')
    plt.show()
    
def plot_sample_imgs(train_loader,n_figures = 40):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']
    ima, targets = next(iter(train_loader))
    denorm = Normalize((-train_mean / train_std).tolist(), (1.0 / train_std).tolist())
    n_row = int(n_figures/10)
    fig,axes = plt.subplots(figsize=(10, 3), nrows = n_row, ncols=10)
    plt.subplots_adjust(hspace=1)
    for ax in axes.flatten():
        a = random.randint(0,len(ima)-1)
        image, target = ima[a], targets[a]
#         image = torch.from_numpy(image)
        image = denorm(image)*255
        image = image.permute(2, 1, 0) # from NHWC to NCHW
        image = image.squeeze().numpy().astype(np.uint8)
        im = ax.imshow(image) #, interpolation='nearest')
        ax.set_title(f'{labels[target]}', fontsize = 8)
        ax.axis('off')
    plt.show()