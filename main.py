from scr.models import *
from scr.utils import *
from scr.module import *
from scr.dataloader import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="cora", help='cora, citeseer, arxiv, flickr, reddit')
parser.add_argument('--ratio', type=float, default= 0.026)
# cora 0.026 citeseer 0.018  arxiv 0.0025 flickr 0.005  reddit 0.001
parser.add_argument('--raw_data_dir', type=str, default="/home/uqxgao4/data/")
parser.add_argument('--result_path', type=str, default="./results")
parser.add_argument('--cond_folder', type=str, default="./cond_graph/")
parser.add_argument('--config_folder', type=str, default="./config/")
parser.add_argument('--epoch', type=int, default=600)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_dim', type=int, default=256)
parser.add_argument('--test_gnn', type=str, default="GCN")
parser.add_argument('--test_gnn_idx', type=int, default=0)

parser.add_argument('--conv_depth', type=int, default=2, help= 'number of conv depth of the original graph')
parser.add_argument('--repeat', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.8)
parser.add_argument('--NICE_layer', type=int, default=2)
parser.add_argument('--NICE_transform_layer', type=int, default=2)

parser.add_argument('--sigma', type=float, default=0.5, help='weight for losses')                   
parser.add_argument('--delta', type=float, default=10, help='weight for losses')                   
parser.add_argument('--temperature', type=float, default=1, help='temperature for ce loss')     
parser.add_argument('--ot_epochs', type=float, default=60, help='epochs for ot')                 
parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon for ot')                
parser.add_argument('--lr_ot', type=float, default=0.0005, help='lr for ot')                   
parser.add_argument('--weight_decay_ot', type=float, default=5e-4, help='weight_decay for ot')  
parser.add_argument('--update_iter', type=float, default=4, help='iteration for update ot')      
parser.add_argument('--warmup', type=float, default=30, help='iteration for warmup')             
parser.add_argument('--noise_level', type=float, default=0.1, help='iteration for warmup')       

parser.add_argument('--generate_adj', type=int, default=0, help='generate the condensed graph')
parser.add_argument('--adj_T', type=float, default=0.95, help='threshold for condensed graph')
parser.add_argument('--alpha', type=float, default=1, help='weight for smoothness')             
args = parser.parse_args()

args = create_folder(args)
args = device_setting(args)
seed_everything(args.seed)

## load hyper parameters
args = load_hyperparams(args)

## load dataset
datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)

graph_file = args.folder+f'{args.dataset_name}_{args.ratio}.pt' if args.generate_adj == 1 else args.folder+f'{args.dataset_name}_noadj_{args.ratio}.pt'
# if os.path.exists(graph_file):
#     graph = torch.load(graph_file, map_location= args.device)
# else:

## condensation
begin = time.time()
args, label_cond = generate_labels_syn(args, data)

H = conv_graph(args, data)
H, cls_mask = mask_gen(args, H, data)
slide_cla, class_table = accumulate(args.budget_cla)  

model =  NICE_OT_improve(args, H.shape[1], args.n_dim, len(label_cond), num_coupling_layers = args.NICE_layer, m_layers = args.NICE_transform_layer).to(args.device)
cross_entropy = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_ot, weight_decay=args.weight_decay_ot)

for epoch in range(int(args.ot_epochs)): 
    if epoch == 0:
        Y, M, partition_mask = gen_label(args, data, H, model, cls_mask, slide_cla, 'kmeans', class_table)
    elif epoch >= args.warmup and (epoch-args.warmup) % args.update_iter ==0 :   
        Y, M, partition_mask = gen_label(args, data, H, model, cls_mask, slide_cla, 'OT', class_table)

    model.train()
    Z = model(H)
    output = model.predict(Z)
    loss = cross_entropy(output[partition_mask]/args.temperature, Y[partition_mask])

    H_aug, _ = mask_feature(H, p = args.noise_level)
    output_aug = model.predict(model(H_aug))
    target = torch.softmax(output.detach(), dim=-1)
    pred = F.log_softmax(output_aug, dim=-1)
    loss += F.kl_div(pred, target) * args.delta

    C = torch.spmm(M, Z.to(args.device))
    P = F.normalize(C, dim=-1, p=2)
    loss += ((P @ P.t()) - torch.eye(len(P)).to(args.device)).pow(2).mean() * args.sigma

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    norms = torch.norm(model.classifier.weight.data, dim=1).max().item()
    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, proto_norm: {norms:.4f}')
    if epoch > 0:
        model = update_classifier(args, model, H, M)
model.eval()
Z = model(H)
C=torch.spmm(M, Z.to(args.device))
h = model.inverse(C).detach()


## generate graph
if args.generate_adj == 1:
    a = get_adj(h, args.adj_T)
    x = get_feature(a, h, args.alpha)
    graph = Data(x=x, y=label_cond, edge_index=a.nonzero().t(), edge_attr=a[a.nonzero()[:,0], a.nonzero()[:,1]], train_mask=torch.ones(len(x), dtype=torch.bool))
    if a.nonzero().shape[0] == 0:
        print('No edges in the condensed graph')
        exit(0) 
else:
    graph = Data(x=h, y=label_cond, edge_index=torch.eye(len(h)).nonzero().t(), edge_attr=torch.ones(len(h)), train_mask=torch.ones(len(h), dtype=torch.bool))
args.cond_time = time.time()-begin
print('Condensation time:',  f'{args.cond_time:.3f}', 's')
print('#edges:', a.nonzero().shape[0]) if args.generate_adj == 1 else print('No adj')
print('#training labels:', data.train_mask.sum().item())
torch.save(graph, graph_file)


# model training
graph=graph.to(args.device)
acc= []
for repeat in range(args.repeat): 
    model = GCN(data.num_features, args.n_dim, args.num_class, 2, args.dropout).to(args.device)
    args.test_gnn = model.__class__.__name__
    acc.append(model_training(model, args, data, graph, data_val, data_test))

result_record(args, acc)
print(f'ratio: {args.ratio}')
print(f'Average accuracy: {np.mean(acc):.4f} ± {np.std(acc):.4f}')
