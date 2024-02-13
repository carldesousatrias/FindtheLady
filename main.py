from torch.nn.utils import prune
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from utils import *
from Architectures import *
from Permutation import permutation
from NNWmethods.UCHI import Uchi_tools

############ Attacks
def adding_noise(net, power, module_name):
    '''add gausian noise to the parameter of the network'''
    if power==0:
        return net
    for name, parameters in net.named_parameters():
        if module_name in name:
            print("noise added")
            calcul = nn.utils.parameters_to_vector(parameters)
            sigma = torch.std(calcul, unbiased=False).item()
            noise = torch.normal(mean=0, std=power*sigma, size=parameters.size())
            parameters.data += noise
    return net

def pruning(net,proportion, module_name):
    '''prune the aimed layer'''
    for name, module in net.named_modules():
        if module_name[:-2] in name:
            print("pruned")
            prune.l1_unstructured(module, name='weight',amount=proportion)
            prune.remove(module,'weight')
    return net

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


########################### Permutation
def hooked_layer(net,module_name):
    ''' return the layer of the network'''
    for name, parameters in net.named_parameters():
        if module_name in name:
            if "weight" in name:
                return parameters.data.detach().clone()

def retrieved_layer_n(net,module_name):
    '''prepare the layer for the neuron depermutation'''
    layer = hooked_layer(net,module_name)
    N = layer.size()[0]
    layer_N = torch.reshape(layer, (N, -1))
    layer_N = torch.nn.functional.normalize(layer_N, p=2, dim=1)
    return layer_N

def permutation_tensor(layer,perm_N):
    '''
    permute the layer using the permutation matrix (on the neuron and/or the channel)
    :param layer:
    :param perm_N:
    :param perm_C:
    :return:
    '''
    return layer[perm_N]

############## Our method
def depermutation(layer,module_name,org_net):
    '''resynchronize the layer using the original network
    :param layer: original layer
    :param module_name: module name
    :param org_net: original network
    :return: resynchronized layer, depermutation matrix
    '''
    N, C = layer.size()[0],layer.size()[1]
    #### neuron depermutation
    org_layer_N = retrieved_layer_n(org_net,module_name)
    layer_N = torch.reshape(layer, (N, -1))
    layer_N = torch.nn.functional.normalize(layer_N, p=2, dim=1)
    corr_matrix_N = torch.mm(org_layer_N, layer_N.t())
    # plt.matshow(corr_matrix_N, cmap=plt.cm.Blues)
    # plt.colorbar()
    perm_N=torch.argmax(corr_matrix_N,dim=0)
    layer=layer[torch.argmax(corr_matrix_N, dim=1),:,:,:]
    return layer, perm_N

if __name__ == '__main__':
    dict_res = {"retrieve_list": [], "testlist": []}
    name="vgg16_cifar10_noise"
    save="vgg16_Uchi"
    module_name = "features.17.w"
    next_module_name = "features.19.w"
    trainset, testset, _ = CIFAR10_dataset()
    trainloader, testloader = dataloader(trainset, testset, batch_size=100)
    os.makedirs('res', exist_ok=True)
    ### for NNW results, uncomment next line but the watermarking dictionnary should be loaded
    # tools, watermarking_dict= Uchi_tools(), np.load(save+".npy",allow_pickle=True).item()

    for power in range(1,10):
        retrieve_res=[]
        fulltest_res=[]
        print("doing..", power)

        ###############################org net extraction############################################
        org_net = tv.models.vgg16()
        checkpoint = torch.load(save + ".pth", map_location=torch.device('cpu'))
        org_net.load_state_dict(checkpoint["model_state_dict"])
        org_net.eval()
        # print_net(our_net)
        ######################### attacked net extraction and permutation/depermutation ####################################################
        new_net = tv.models.vgg16()
        new_net.load_state_dict(checkpoint["model_state_dict"])

        adding_noise(new_net, power ,module_name)
        # to do fine-tuning load another weights to "new_net"
        # quantization(new_net, 4)
        # pruning(new_net, power/10,module_name)

        ######
        new_net.eval()
        new_layer=hooked_layer(new_net, module_name)

        ###### applying permutation
        permut_N = torch.randperm(new_layer.size()[0])
        permutation(new_net, permut_N, module_name, next_module_name) ## to the model
        new_layer = permutation_tensor(new_layer,permut_N)

        ###### applying permutation
        new_layer, perm_N = depermutation(new_layer,module_name,org_net)
        permutation(new_net, perm_N, module_name, next_module_name)
        ### for NNW results, uncomment next line but the watermarking dictionnary should be loaded
        # print(tools.detection(new_net.to(device), watermarking_dict))
        print("(neuron) differences between permut and retrieved permut=", calcul_diff(perm_N, permut_N))
        new_net.to(device)
        retrieve_res.append(100-calcul_diff(perm_N, permut_N)/len(permut_N)*100)
        new_net.to(device)
        test=fulltest(new_net, testloader)
        print(test)
        fulltest_res.append(test)
    dict_res['retrieve_list'].append(retrieve_res)
    dict_res['testlist'].append(fulltest_res)

    df = pd.DataFrame.from_dict(dict_res)
    df.to_csv('res/' + name + '.csv',
              index=True, header=True)
