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

def quantization(net,num_bits):
    def fake_quantization(x, num_bits):
        qmax = 2. ** num_bits - 1.
        min_val, max_val = torch.min(x), torch.max(x)
        scale = qmax / (max_val - min_val)
        x_q = (x - min_val) * scale
        x_q.clamp_(0, qmax).round_()  # clamp = min(max(x,min_value),max_value)
        x_q.byte()
        x_f_q = x_q.float() / scale + min_val
        return x_f_q

    with torch.no_grad():
        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                tensor = module.weight
                tensor_q = fake_quantization(tensor, num_bits)
                module.weight = nn.Parameter(tensor_q)

def print_net(net):
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())

def calcul_diff(original,retrieved):
    res=0
    for i in range(min(len(retrieved),len(original))):
        if original[i]!=retrieved[i]:res+=1
    return res

def hooked_layer(net,module_name):
    for name, parameters in net.named_parameters():
        if module_name in name:
            if "weight" in name:
                return parameters.data.detach().clone()

def get_canonical(net,module_name):
    layer=hooked_layer(net,module_name)
    flat_weight = torch.sum(layer,dim=(1, 2, 3))  # thing to rank
    sorted_weight, indices = torch.sort(flat_weight, dim=0, descending=True)
    return sorted_weight, indices


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
    save = 'vgg16_Uchi'
    module_name = "features.17.w"

    torch.cuda.empty_cache()

    net = tv.models.vgg16().to(device)

    checkpoint = torch.load(save + ".pth", map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    for param in net.parameters():
        param.requires_grad = False

    ylim = (0, 105)
    _,canonical_order= get_canonical(net,module_name)


    trainset, testset, _ = CIFAR10_dataset()
    # # blackbox method here
    #
    trainloader, testloader = dataloader(trainset, testset, 100)
    criterion_t = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer_t = optim.SGD(net.parameters(), lr=learning_rate, momentum=.9, weight_decay=1e-4)
    q_list=[int(i) for i in range(2,17)]
    res=[]
    test=[]
    for q in q_list:
        print("doing q=",q)
        net.load_state_dict(checkpoint["model_state_dict"])
        quantization(net,q)
        _,utimate_canonical = get_canonical(net,module_name)
        res.append(100-calcul_diff(canonical_order,utimate_canonical)/len(canonical_order)*100)
        test.append(fulltest(net,testloader))
        print("(neuron) differences between permut and retrieved permut=", calcul_diff(canonical_order, utimate_canonical))

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    plt.xticks(q_list, q_list)
    ax1.set_xlabel(r'$B$')
    ax1.set_ylabel(r'$\Psi$', color=color)
    plt.ylim(ylim)
    ax1.plot(q_list, res, marker='.', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel(r'$\it err \% $', color=color)  # we already handled the x-label with ax1
    plt.ylim(ylim)
    ax2.plot(q_list, test, marker='x',linestyle='--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.savefig('fig_method_0.pdf')
    plt.show()
