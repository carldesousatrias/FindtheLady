# This is a sample Python script.
from utils import *
from Architectures import *
from NNWmethods.UCHI import Uchi_tools


def training(net, trainloader, optimizer, criterion, watermarking_dict=None):
    '''
    :param watermarking_dict: dictionary with all watermarking elements
    :return: the different losses ( global loss, task loss, watermark loss)
    '''
    running_loss = 0
    running_loss_nn = 0
    running_loss_watermark = 0
    for i, data in enumerate(trainloader, 0):
        # split data into the image and its label
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        if inputs.size()[1] == 1:
            inputs.squeeze_(1)
            inputs = torch.stack([inputs, inputs, inputs], 1)
        # initialise the optimiser
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)
        # backward
        loss_nn = criterion(outputs, labels)
        # watermark
        loss = loss_nn

        loss.backward()
        # update the optimizer
        optimizer.step()

        # loss
        running_loss += loss.item()
        running_loss_nn += loss_nn.item()
    return running_loss, running_loss_nn, 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # (optional) Reproducibility/Repeatability
    # torch.manual_seed(0)
    # np.random.seed(0)
    save='vgg16_Uchi'

    # initialisation
    num_class=10
    network = tv.models.vgg16()
    network=network.to(device)
    print_net(network)
    criterion = nn.CrossEntropyLoss()
    num_epochs=10
    batch_size=128
    learning_rate = 0.01
    trainset, testset, inference_transform = CIFAR10_dataset()
    trainloader, testloader = dataloader(trainset,testset,batch_size)

    optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=.1)

    ### for watermarking add the section at the bottom of "NNWmethods/UCHI.py"

    # training
    epoch,loss_nn=0,10

    while epoch<num_epochs:
        network.train()
        print(" Starting epoch " + str(epoch + 1) + '...')
        prev_loss_nn=loss_nn
        loss,loss_nn,loss_w=training(network, trainloader, optimizer, criterion) #tools.training for watermarking
        scheduler.step()
        loss = (loss * batch_size / len(trainloader.dataset))
        loss_nn = (loss_nn * batch_size / len(trainloader.dataset))
        loss_w = (loss_w * batch_size / len(trainloader.dataset))
        print(' loss  : %.5f   - loss_wm: %.5f, loss_nn: %.5f  ' % (loss,loss_w,loss_nn))
        epoch+=1

        # (optional) save
    torch.save({
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save + '.pth')

    checkpoint = torch.load(save+".pth", map_location=torch.device('cpu'))
    network.load_state_dict(checkpoint["model_state_dict"])

    # Small report
    print('Finished Training:')
    print('loss  : %.5f ; task loss  : %.5f, watermark loss  : %.5f  ' % (loss,loss_nn,loss_w))
    print('Validation error : %.2f' %fulltest(network,testloader))

