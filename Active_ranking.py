import torch
from torch.nn.utils import prune
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from utils import *
from Architectures import *
import matplotlib
font = {'family':'sans serif','sans-serif':['Helvetica'],
    'size': 18}
matplotlib.get_cachedir()
matplotlib.rc('font', **font)
matplotlib.rc('text',**{"usetex": True})

def print_net(net):
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())

def calcul_diff(original,retrieved):
    res=0
    for i in range(min(len(retrieved),len(original))):
        if original[i]!=retrieved[i]:res+=1
    return res

def adding_noise(net, power, module_name):
    '''add gausian noise to the parameter of the network'''
    if power==0:
        return net
    for name, parameters in net.named_parameters():
        if module_name in name:
            print("noise added")
            calcul = nn.utils.parameters_to_vector(parameters)
            sigma = torch.std(calcul, unbiased=False).item()
            noise = torch.normal(mean=0, std=power*sigma, size=parameters.size(),device=device)
            parameters.data += noise
    return net

def hooked_net(net, weight_name):
    hook = small_net(net, weight_name)
    new_model = nn.Sequential(*list(net.children())[0][:hook + 4]).to(device)
    return new_model

def get_canonical_t(smallnet,input):
    outputs = smallnet(input).squeeze(0)
    flat_weight = outputs.sum(dim=(1, 2))  # thing to rank
    sorted_weight, indices = torch.sort(flat_weight, dim=0, descending=True)
    return sorted_weight, indices

def get_loss(smallnet, input,  gamma=1):

    #investigate by testing with only looking the gap between two neurons

    outputs = smallnet(input).squeeze(0)
    N=outputs.size()[0]
    flat_weight = outputs.sum(dim=(1, 2)).unsqueeze(1) # thing to rank
    transpose_flat_weight = torch.transpose(flat_weight, 0, 1)
    deltas = torch.abs(flat_weight-transpose_flat_weight)
    tril_deltas=torch.tril(deltas,-1) #####
    mean=torch.sum(tril_deltas)/(N*(N-1)/2)
    std=torch.sum(torch.tril(torch.pow(deltas-mean,2),-1))/(N*(N-1)/2)
    # inf_tensor=torch.
    deltas[deltas==0]=np.inf
    return torch.sum(torch.tril(- deltas , -1)),mean, std, deltas

def small_net(net, weights_name):
    '''find the position of the layer with the name_w in net.modules'''
    i = 1
    for name, parameters in net.named_parameters():
        i += 1
        if weights_name in name:
            return i
    return "error"

def show_histo_dist(sorted_weight):
    np_sorted_weight = sorted_weight.cpu().detach().numpy()
    vparameters = np_sorted_weight.flatten()
    np_arr = vparameters
    print(norm.fit(np_arr))
    plt.figure()
    plt.hist(np_arr, bins = 10)


if __name__ == '__main__':
    torch.manual_seed(0)
    save = 'vgg16'
    weight_name = "features.17"
    print(device)

    torch.cuda.empty_cache()

    fullnet = vgg16()
    checkpoint = torch.load(save + ".pth", map_location=torch.device('cpu'))
    fullnet.load_state_dict(checkpoint["model_state_dict"])
    fullnet.eval()
    for param in fullnet.parameters():
        param.requires_grad = False
    net=hooked_net(fullnet, weight_name)
    lr=1
    trigger_input = torch.rand((1, 3, 32,32), requires_grad=True, device=device)
    trigger_input.requires_grad = True

    gamma=0
    optimizer=torch.optim.SGD([trigger_input],lr=lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20000,gamma=0.1)

    mini_list=[]
    maxi_list=[]
    loss_list=[]
    for i in tqdm(range(30000)):
        optimizer.zero_grad()
        loss,mean,std,deltas = get_loss(net, trigger_input,gamma)

        loss.backward()

        loss_list.append(loss.clone().detach().item())
        optimizer.step()
        scheduler.step()
        mini_list.append(torch.min(deltas).clone().detach().item())
        maxi_list.append(torch.max(deltas).clone().detach().item())


    weight_sorted,canonical_order=get_canonical_t(net, trigger_input)

    trainset, testset, _ = CIFAR10_dataset()

    trainloader, testloader = dataloader(trainset, testset, 100)
    criterion_t = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer_t = optim.SGD(net.parameters(), lr=learning_rate, momentum=.9, weight_decay=1e-4)
    power_list=np.arange(0,10)
    res=[]
    for power in power_list:
        fullnet.load_state_dict(checkpoint["model_state_dict"])
        adding_noise(fullnet,power,weight_name)
        lastnet = hooked_net(fullnet, weight_name).to(device)
        _,utimate_canonical = get_canonical_t(lastnet, trigger_input)
        res.append(calcul_diff(canonical_order,utimate_canonical)/len(canonical_order)*100)
        print("(neuron) differences between permut and retrieved permut=", calcul_diff(canonical_order,utimate_canonical))
    plt.figure()
    plt.plot(power_list,res)
    plt.xlabel("number of fine-tuning epochs")
    plt.ylabel("percentage of permuted neurons")
    plt.show()
