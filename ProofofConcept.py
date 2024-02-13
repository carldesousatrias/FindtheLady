from utils import *
from Architectures import *
import matplotlib
import matplotlib.pyplot as plt
font = {'family':'sans serif','sans-serif':['Helvetica'],
    'size': 18}
matplotlib.get_cachedir()
matplotlib.rc('font', **font)
matplotlib.rc('text',**{"usetex": True})

def calcul_diff(original,retrieved):
    ''' calculate the difference between the original and the retrieved order'''
    res=0
    for i in range(min(len(retrieved),len(original))):
        if original[i]!=retrieved[i]:res+=1
    return res

def show_net(net):
    '''print the network'''
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())


def hooked_net(net, weight_name):
    '''return the network from the first layer to the aimed layer '''
    hook = small_net(net, weight_name)
    new_model = nn.Sequential(*list(net.children())[0][:hook]).to(device)
    return new_model

def hooked_layer_wo(net, weight_name):
    '''return the layer of the network, without activation function'''
    hook = small_net(net, weight_name)
    test=list(net.children())[0]

    new_model = nn.Sequential(*list(net.children())[0][hook -2:hook-1]).to(device)
    return new_model

def hooked_layer(net, weight_name):
    '''return the layer of the network, with activation function'''
    hook = small_net(net, weight_name)
    test=list(net.children())[0]

    new_model = nn.Sequential(*list(net.children())[0][hook -2:hook]).to(device)
    return new_model

def find_layer(net,module_name):
    '''return the weights tensor of the layer'''
    for name, parameters in net.named_parameters():
        if module_name in name:
            if "weight" in name:
                return parameters.data.detach().clone()

def small_net(net, weights_name):
    '''find the position of the layer with the name_w in net.modules'''
    i = 1
    for name, parameters in net.named_parameters():
        i += 1
        if weights_name in name:
            return i + 4 #4 number of maxpool_layer
    return "error"


def new_y(y):
    '''return the similarity matrix for all outputs'''
    y=y.squeeze(0)
    new_y=torch.reshape(y,(y.size()[0],-1))
    norm_y=new_y.norm(p=2,dim=1).unsqueeze(1)
    norm_y.clamp_(min=1e-12)
    norm_2=torch.mm(norm_y,norm_y.t())
    sim=torch.mm(new_y,new_y.t())/norm_2
    select_sim=torch.tril(sim,-1)
    return select_sim

if __name__ == '__main__':


    # for i in range(1000):
    i=2
    torch.manual_seed(i)
    weights_folder = 'vgg16_Uchi.pth'
    module_name = "features.28.w"
    print("seed",i)
    dataset_name = 'CIFAR10'
    trainset, testset , _ = CIFAR10_dataset()
    # blackbox method here

    trainloader, testloader = dataloader(trainset, testset, 100)

    fullnet = tv.models.vgg16().to(device)
    parameters = find_layer(fullnet, module_name)
    # show_net(fullnet)
    checkpoint = torch.load(weights_folder, map_location=torch.device('cpu'))
    fullnet.load_state_dict(checkpoint["model_state_dict"])
    fullnet.eval()
    for param in fullnet.parameters():
        param.requires_grad = False
    net2 = hooked_net(fullnet, "features.26.w")
    # show_net(net2)
    nb_images=[]
    for i in range(5):
        nb_images.append(trainloader.dataset[i][0].unsqueeze(0).to(device))


    net=hooked_net(fullnet, module_name)
    # show_net(net)

    final_delta = None
    for img in nb_images:
        result = net(img)
        result = new_y(result)
        final_delta = result if final_delta is None else final_delta + result

    # find 2 neurons similar:
    print(final_delta.max().item())
    i=torch.argmax(final_delta)//512
    j=torch.argmax(final_delta)%512
    print(i.item(),j.item())

    # find the 2 neurons cosine similarity
    layer_N = torch.reshape(parameters, (parameters.size()[0], -1))
    layer_N = torch.nn.functional.normalize(layer_N, p=2, dim=1)
    corr_matrix_N = torch.mm(layer_N, layer_N.t())
    print(corr_matrix_N[i][j].item())


    # extract the 2 neurons' behaviour
    layer = hooked_layer(fullnet,module_name)
    layer_wo=hooked_layer_wo(fullnet,module_name)

    y_i=[]
    y_j=[]
    y_ij=[]
    y_ij_wo=[]
    for nb in range(len(nb_images)-1):
        img_a=nb_images[nb]
        img_b=nb_images[nb+1]
        y_a=net2(img_a)
        y_b=net2(img_b)
        for weight in range(101):
            interpolation=torch.lerp(y_a,y_b,weight/100)
            output=layer(interpolation).squeeze()
            # output_sim=cosinedist(output[i],output[j])
            output_i= output[i].reshape(-1).unsqueeze(0)
            output_j= output[j].reshape(-1).unsqueeze(1)
            cosdist=torch.mm(output_i,output_j)/(torch.norm(output_i)*torch.norm(output_j))
            y_ij.append(((cosdist[0][0]*10**3)/10**3).round().item())
            output = layer_wo(interpolation).squeeze()
            # output_sim=cosinedist(output[i],output[j])
            output_i = output[i].reshape(-1).unsqueeze(0)
            output_j = output[j].reshape(-1).unsqueeze(1)
            cosdist = torch.mm(output_i, output_j) / (torch.norm(output_i) * torch.norm(output_j))
            y_ij_wo.append(cosdist[0][0].item())
            y_i.append(output[i].norm())
            y_j.append(output[j].norm())
            # print(output[i],output[j])

    #plot it
    x=[i for i in range(len(y_ij))]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(r'$y_{l-1}$')
    ax1.set_ylabel(r'$S_C$')
    ax1.plot(x, y_ij, color=color,label=r'$\varphi\left(\left< \boldmath{w}_{l}, (\boldmath{y}_{l-1,-})^T\right> \right)$')
    ax1.plot(x, y_ij_wo, color='tab:blue', linestyle='--', label=r'$\left< \boldmath{w}_{l}, (\boldmath{y}_{l-1,-})^T\right>$')
    ax1.tick_params(axis='y')
    test=[1,2,3,4]
    plt.xticks(np.arange(0, len(y_ij),step=100),[1,2,3,4,5])

    plt.legend(loc='center',fontsize=14,bbox_to_anchor=(.8, -.21))
    plt.tight_layout()
    color = 'tab:blue'



    plt.savefig('fig_hope.pdf')
    plt.show()









