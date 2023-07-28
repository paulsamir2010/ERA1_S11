


def test11(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    tst_corr = 0
    test_correct = 0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

  
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

                        # Tally the number of correct predictions
            predicted = torch.max(output.data, 1)[1] 
            tst_corr += (predicted == target).sum()
            
        loss = criterion(output, target)
        #test_losses.append(loss)
        #test_correct.append(tst_corr)

        #test_loss /= len(test_loader.dataset)
        #test_losses.append(test_loss)

        print('\nTest set:  Accuracy: {}/{} ({:.2f}%)\n'.format(
         tst_corr, len(test_loader.dataset),
        100. * tst_corr / len(test_loader.dataset)))
    
    #test_acc.append(100. * tst_corr / len(test_loader.dataset))