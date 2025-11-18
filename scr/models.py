from scr.module import *

class GCN(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GCNConv(nin, nout))
        else:
            self.layers.append(GCNConv(nin, nhid)) 
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv(nhid, nhid)) 
            self.layers.append(GCNConv(nhid, nout))  
        self.dropout = dropout
        self.initialize()

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)
    
    def forward_distill(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1), x
    
class SGC(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, cached=False, dropout=0):
        super().__init__()
        self.layers = SGConv(nin, nout, nlayers, cached=cached) 
        self.H_val =None
        self.H_test=None
        self.dropout = dropout
        self.initialize()

    def initialize(self):
        self.layers.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1) 
    
    def MLP(self, H):
        x = self.layers.lin(H)
        return F.log_softmax(x, dim=1)    
           



def get_NICE_mask(args, dim, orientation=True):
    mask = np.zeros(dim)
    mask[::2] = 1.
    if orientation:
        mask = 1. - mask  # flip mask orientation
    mask = torch.tensor(mask).requires_grad_(False)
    mask = mask.to(args.device)
    return mask.float()
    

class CouplingLayer(nn.Module):
    """
  Implementation of the additive coupling layer from section 3.2 of the NICE
  paper.
  """

    def __init__(self, data_dim, hidden_dim, mask, num_layers=4):
        super().__init__()
        self.mask = mask
        modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LeakyReLU(0.2)) 
        modules.append(nn.Linear(hidden_dim, data_dim))

        self.m = nn.Sequential(*modules)

    def forward(self, x, invert=False):
        if not invert:
            x1, x2 = self.mask * x, (1. - self.mask) * x
            y1, y2 = x1, x2 + (self.m(x1) * (1. - self.mask))
            return y1 + y2

        y1, y2 = self.mask * x, (1. - self.mask) * x
        x1, x2 = y1, y2 - (self.m(y1) * (1. - self.mask))
        return x1 + x2 
    




class NICE_OT_improve(nn.Module):
    def __init__(self, args, nin, nhid, nout, num_coupling_layers, m_layers):
        super().__init__()

        self.masks = [self._get_mask(nin, orientation=(i % 2 == 0)).to(args.device)
                 for i in range(num_coupling_layers)]

        self.coupling_layers = nn.ModuleList([CouplingLayer(data_dim=nin,
                                                            hidden_dim=nhid,
                                                            mask=self.masks[i], num_layers=m_layers)
                                              for i in range(num_coupling_layers)])
        self.classifier = Linear(nin, nout, bias=False)
        self.slide_cla = self.accumulate(args.budget_cla)

    def accumulate(self, budget_cla):
        # // accumulate the budget
        slide_cla = [0]
        for i in range(len(budget_cla)):
            slide_cla.append(slide_cla[-1]+budget_cla[i])
        return slide_cla


    def forward(self, x):
        z = x
        for _, coupling_layer in enumerate(self.coupling_layers):
            z = coupling_layer(z)
        return z

    def inverse(self, z):
        x = z
        for _, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
            x = coupling_layer(x, invert=True)
        return x
    
    def predict(self, x):
        w = self.classifier.weight
        w = F.normalize(w, dim=1, p=2)
        x = F.normalize(x, dim=1, p=2)
        x = x@w.t()
        # x = self.classifier(x)
        return x
    
    def _get_mask(self, dim, orientation=True):
        mask = np.zeros(dim)
        mask[::2] = 1.
        if orientation:
            mask = 1. - mask  # flip mask orientation
        mask = torch.tensor(mask).requires_grad_(False)
        return mask.float()
    
