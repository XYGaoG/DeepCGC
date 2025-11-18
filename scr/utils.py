from scr.module import *
from scr.models import *


def load_hyperparams(args):
    """
    Load hyperparameters from a YAML file and update the provided args.

    Parameters:
    - args (Namespace): Argument namespace containing 'dataset_name', 'ratio', and 'generate_adj'.

    Returns:
    - Namespace: Updated args with the loaded hyperparameters.
    """
    dataset_name, label_ratio, generate_adj = args.dataset_name, args.ratio, args.generate_adj
    yaml_path = os.path.join(args.config_folder, f"{dataset_name}.yaml")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    if dataset_name not in config:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.")

    ratio_key = f"ratio:{label_ratio}"
    if ratio_key not in config[dataset_name]:
        raise ValueError(f"Label ratio '{label_ratio}' not found for dataset '{dataset_name}'.")

    generate_adj_key = f"generate_adj:{generate_adj}"
    if generate_adj_key not in config[dataset_name][ratio_key]:
        raise ValueError(f"generate_adj={generate_adj} not defined for dataset '{dataset_name}' and ratio {label_ratio}.")

    # Get the configuration
    hyperparams = config[dataset_name][ratio_key][generate_adj_key]

    # Update args with the hyperparameters
    for key, value in hyperparams.items():
        setattr(args, key, value)

    # Return the updated args
    return args


def generate_labels_syn(args, data):
    reduction_rate = ratio_transfer(args)

    from collections import Counter
    counter = Counter(np.array(data.y[data.train_mask].cpu()))
    num_class_dict = {}
    n = len(data.y[data.train_mask])

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    sum_ = 0
    for ix, (c, num) in enumerate(sorted_counter):
        if ix == len(sorted_counter) - 1:
            num_class_dict[c] = int(n * reduction_rate) - sum_
        else:
            num_class_dict[c] = max(int(num * reduction_rate), 1)
            sum_ += num_class_dict[c]

    num_class = np.zeros(len(num_class_dict), dtype=int)
    for i in range(len(num_class_dict)):
        num_class[i] = num_class_dict[i]

    labels_syn = []
    for i in range(args.num_class):
        labels_syn += [i] * num_class[i]
    labels_syn = torch.tensor(labels_syn).long()

    args.budget = sum(num_class)
    args.budget_cla = num_class

    return args, labels_syn


def device_setting(args):
    if args.gpu != -1:
        args.device='cuda'
    else:
        args.device='cpu'  
    torch.cuda.set_device(args.gpu)
    return args


def create_folder(args):
    args.folder =  args.cond_folder + f'{args.dataset_name}/'
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    return args


def ratio_transfer(args):
    if args.dataset_name == 'cora':
        if args.ratio == 0.013:
            return 0.25
        if args.ratio == 0.026:
            return 0.5
        if args.ratio == 0.052:
            return 1
    
    elif args.dataset_name == 'citeseer':
        if args.ratio == 0.009:
            return 0.25
        if args.ratio == 0.018:
            return 0.5
        if args.ratio == 0.036:
            return 1
    
    elif args.dataset_name == 'arxiv':
        if args.ratio == 0.0005:
            return 0.001
        if args.ratio == 0.0025:
            return 0.005
        if args.ratio == 0.005:
            return 0.01

    else:
        return args.ratio
    


def propagate_assignment(args, M, adj_norm):
    adj = adj_norm.cpu()
    C_filter = M.to_dense().cpu().t()
    M = M.cpu().t()
    for _ in range(2):
        C_filter = 0.2 * torch.spmm(adj, C_filter) + M
    C_filter = F.normalize(C_filter, p=2, dim=1)


    indices = C_filter.nonzero(as_tuple=False).T  # shape: [2, num_nonzero]

    # Get corresponding values
    values = C_filter[tuple(indices)]

    # Create sparse COO tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=C_filter.shape)
    sparse_tensor = sparse_tensor.coalesce() 


    return sparse_tensor.to(args.device).t()


def conv_graph(args, data):
    adj_norm = normalize_adj_sparse(data).to(data.x.device)
    H0 = data.x
    H1 = torch.spmm(adj_norm, H0)
    H2 = torch.spmm(adj_norm, H1)
    H_out = [H0, H1, H2]
    return H_out


def aug_feature_dropout(input_feat, drop_percent=0.5):
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0
    return aug_input_feat

def conv_graph_multi(args, data):
    if args.kernel == "gcn":
        adj_norm = normalize_adj_sparse(data).to(data.x.device)
        H0 = data.x
        H1 = torch.spmm(adj_norm, H0)
        H2 = torch.spmm(adj_norm, H1)
        return H0, H1, H2


def normalize_adj_sparse(data):

    try:
        mx = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index.cpu().numpy()), shape=(data.x.shape[0], data.x.shape[0]))
        if type(mx) is not sp.lil.lil_matrix:
            mx = mx.tolil()
        if mx[0, 0] == 0 :
            mx = mx + sp.eye(mx.shape[0])
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1/2).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        mx = mx.dot(r_mat_inv)

        sparse_mx = mx.tocoo().astype(np.float32)
        sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
        sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
        sparseconcat=torch.cat((sparserow, sparsecol),1)
        sparsedata=torch.FloatTensor(sparse_mx.data)
        adj = torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

    except Exception as e:
        adj = normalize_sparse_tensor(data.edge_index)
    return adj


def mask_gen(args, H, data):
    if args.dataset_name in ['cora', 'citeseer']:
        feat = sum(H)/len(H)
        H = H[2]
        Y_L = torch.nn.functional.one_hot(data.y[data.train_mask], args.num_class).float()
        model = torch.linalg.lstsq(feat[data.train_mask.cpu()].cpu(), Y_L.cpu(), driver="gelss")[0]
        logit = (H @ model.to(args.device)).softmax(dim=-1)
        y_pred = logit.argmax(1)
        conf_pred = logit[np.arange(len(logit)), y_pred].cpu()
        topk_indices = torch.topk(conf_pred, k = int(5*data.train_mask.sum().item())).indices
        y_pred[data.train_mask] = data.y[data.train_mask]
        train_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
        train_mask[topk_indices] = True
        train_mask[data.train_mask] = True
        train_mask[data.val_mask] = False
        train_mask[data.test_mask] = False
        cls_mask = []
        for cls in range(args.num_class):
            cls_mask.append((y_pred == cls) & train_mask)
    else:
        H = H[2]
        train_mask = data.train_mask
        cls_mask = []
        for cls in range(args.num_class):
            cls_mask.append((data.y == cls) & train_mask)
    return  H, cls_mask




def sample_k_nodes_per_label(label, visible_nodes, k, num_class):
    ref_node_idx = [
        (label[visible_nodes] == lbl).nonzero().view(-1) for lbl in range(num_class)
    ]
    sampled_indices = [
        label_indices[torch.randperm(len(label_indices))[:k]]
        for label_indices in ref_node_idx
    ]
    return visible_nodes[torch.cat(sampled_indices)]


def add_self_loops_(edge_index, edge_weight=None, fill_value=1, num_nodes=None):

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def normalize_sparse_tensor(adj, fill_value=1):
    """Normalize sparse tensor. Need to import torch_scatter
    """
    row, col, _ = adj.coo()
    edge_index = torch.stack([row, col])
    num_nodes= adj.sizes()[0]
    edge_weight = torch.ones(adj.nnz()).to(adj.device())

    edge_index, edge_weight = add_self_loops_(
	edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    shape = adj.sizes()
    return torch.sparse.FloatTensor(edge_index, values, shape)






def data_assessment(args, data, model, feat):

    if args.aug_ratio >0:
        
        H_augs = feat[1][data.train_mask]
        y = data.y[data.train_mask]

        logits_aug = H_augs @ model.to(args.device)
        logits_aug = logits_aug.softmax(dim=-1)
        y_pred = logits_aug.argmax(1)
        report = classification_report(y.cpu().numpy(), y_pred.cpu().numpy(), output_dict=True, zero_division=0)
        acc = [float(metrics['f1-score']) for label, metrics in report.items() if label.isdigit()]
        weights = 1 - np.array(acc)
        probabilities = weights[y.cpu()]+1e-20
        pro = probabilities/sum(probabilities)
        idx = np.random.choice(len(pro), size=int(len(H_augs)*args.aug_ratio), p=pro)

        #process aug
        H_aug = H_augs[idx]
        y_aug = y[idx]
        logits_aug = logits_aug[idx]
        y_pred_aug = logits_aug.argmax(1)
        conf_pred_aug = logits_aug[np.arange(len(logits_aug)), y_aug].cpu()
        conf_pred_aug[y_pred_aug!=y_aug]=torch.min(conf_pred_aug)

        H = feat[2][data.train_mask]
        logits = H @ model.to(args.device)
        logits = logits.softmax(dim=-1)
        y_pred = logits.argmax(1)
        conf_pred = logits[np.arange(len(logits)), y].cpu()
        conf_pred[y_pred!=y]=torch.min(conf_pred)

        H_all = torch.concat([H, H_aug])
        y_all = torch.concat([y, y_aug])
        conf_all = torch.concat([conf_pred, conf_pred_aug])

    else:
        y = data.y[data.train_mask]
        H = feat[2][data.train_mask]
        logits = H @ model.to(args.device)
        logits = logits.softmax(dim=-1)
        y_pred = logits.argmax(1)
        conf_pred = logits[np.arange(len(logits)), y].cpu()
        conf_pred[y_pred!=y]=torch.min(conf_pred)

        H_all = H
        y_all = y
        conf_all = conf_pred

    return H_all, y_all, conf_all



def clustering_fast(H, num_center, method):

    if num_center == len(H):
        return np.arange(num_center)

    if method == 'kmeans':
        kmeans = faiss.Kmeans(int(H.shape[1]), int(num_center), gpu=False)
        kmeans.cp.min_points_per_centroid = 1
        kmeans.train(H.astype('float32'))
        _, I = kmeans.index.search(H.astype('float32'), 1)
        cluster_labels = I.flatten()

    return cluster_labels


def get_adj(h, adj_T):
    h = F.normalize(h, dim=1)
    a = torch.mm(h, h.t())
    a = (a>adj_T).float()
    adj = a
    return adj


def get_feature(a, h, alpha):
    lap = laplacian_tensor(a)  
    a_norm = normalize_adj_tensor(a)
    a_conv = a_norm
    for _ in range(2):
        a_conv = torch.mm(a_conv, a_norm)
    aTa = torch.mm(a_conv, a_conv.t())
    aTh = alpha*torch.mm(a_conv.t(), h)
    k = (aTa+lap)
    x = torch.linalg.solve(k, aTh)
    return x 


def laplacian_tensor(adj):

    r_inv = adj.sum(1).flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv  + 1e-10
    r_inv = r_inv.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    L_norm = r_mat_inv @ L @ r_mat_inv
    return L_norm


def normalize_adj_tensor(adj):
    device = adj.device

    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx


def model_training(model, args, data, graph, data_val=None, data_test=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = test_acc = 0
    begin_time = time.time()
    for epoch in range(1, int(args.epoch)+1):
        if epoch == args.epoch // 2:
            lr = args.lr*0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        model.train()
        output = model(graph)
        loss = F.nll_loss(output[graph.train_mask], graph.y[graph.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.dataset_name in ['flickr', 'reddit']:
            # if epoch%50 == 1:
            train_acc, val_acc, tmp_test_acc = test_inductive(args, model, data_val, data_test)
        else:
            train_acc, val_acc, tmp_test_acc = test(model, data.to(args.device))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if epoch%100 == 0 :
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Train: {train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
    print()
    end_time = time.time()
    train_time = end_time - begin_time
    print(f"Training time on condensed graph: {train_time:.3f}s")
    return test_acc



def feature_processing(args, data_val, data_test, k):

    model = SGC(data_val.num_features, None, int(data_val.y.max()+1), k, cached=True).to(args.device)
    model(data_val)
    feat_val = model.layers._cached_x
    model = SGC(data_val.num_features, None, int(data_val.y.max()+1), k, cached=True).to(args.device)
    model(data_test)
    feat_test = model.layers._cached_x
    return feat_val, feat_test


def test_inductive(args, model, data_val, data_test, k=2):
    if model.__class__.__name__ == 'SGC' or model.__class__.__name__ == 'SGC2':
        if model.H_val == None:
            model.H_val,  model.H_test = feature_processing(args, data_val, data_test, k)
        with torch.no_grad():
            model.eval()
            accs = []
            accs.append(0)

            out = model.MLP(model.H_val)
            pred = out.argmax(1)
            acc = pred.eq(data_val.y).sum().item() / len(data_val.y)
            accs.append(acc)      

            out = model.MLP(model.H_test)
            pred = out.argmax(1)
            acc = pred.eq(data_test.y).sum().item() / len(data_test.y)
            accs.append(acc)                
    else:
        with torch.no_grad():
            model.eval()
            accs = []
            accs.append(0)
            for data in [data_val, data_test]:
                out = model(data)
                pred = out.argmax(1)
                acc = pred.eq(data.y).sum().item() / len(data.y)
                accs.append(acc)
    return accs


def test(model, data):
    with torch.no_grad():
        model.eval()
        out, accs = model(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = out[mask].argmax(1)
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs



def result_record(args, ALL_ACCs):
    result_path_file = args.result_path + f"{args.dataset_name}.csv" if args.generate_adj == 1 else args.result_path + f"{args.dataset_name}_noadj.csv"
    ALL_ACC = [np.mean(ALL_ACCs, axis=0)*100, np.std(ALL_ACCs, axis=0, ddof=1)*100] if len(ALL_ACCs) > 1 else [ALL_ACCs[0]*100, 0]
    with open(result_path_file ,'a+',newline='')as f:
        writer = csv.writer(f)
        writer.writerow( [
        "dataset_name:" f"{args.dataset_name}",
        "generate_adj:" f"{args.generate_adj}",
        "ratio:" f"{args.ratio}",
        "cond_time:" f'{args.cond_time:.3f}s',  
        "test_GNN:" f"{args.test_gnn}", 
        "epoch:" f"{args.epoch}",
        "lr:" f"{args.lr}",
        "weight_decay:" f"{args.weight_decay}",
        "dropout:" f"{args.dropout}",
        "adj_T:" f"{args.adj_T}",
        "alpha:" f"{args.alpha}",
        "sigma:" f"{args.sigma}",
        "delta:" f"{args.delta}",
        "temperature:" f"{args.temperature}",
        "ot_epochs:" f"{args.ot_epochs}",
        "epsilon:" f"{args.epsilon}",
        "lr_ot:" f"{args.lr_ot}",
        "weight_decay_ot:" f"{args.weight_decay_ot}",
        "update_iter:" f"{args.update_iter}",
        "warmup:" f"{args.warmup}",
        "noise_level:" f"{args.noise_level}",
        "ave:" f"{ALL_ACC[0]:.1f}±{ALL_ACC[1]:.1f}",
        "acc:" f"{ALL_ACC[0]:.1f}"
        ])
        



def assignment_generation(H, y, args, train_mask=None):
    budgets = args.budget_cla
    idx = torch.arange(len(H))
    indices = torch.LongTensor().new_empty((0, 2))
    values = torch.FloatTensor() 
    row = 0
    buf = 0
    assignments = []
    id  = []
    var=0
    for cls in range(args.num_class):
        if train_mask is not None:
            cls_mask = (y == cls) & train_mask
        else:
            cls_mask = (y == cls)
        H_cls = H[cls_mask].cpu().detach().numpy()
        cluster_labels = clustering_fast(H_cls, budgets[cls], args.clustering)

        labels = cluster_labels + buf
        buf += budgets[cls]
        idx_ = idx[cls_mask]
        assignments = np.concatenate([assignments, labels])
        id = np.concatenate([id, idx_])

        a, b = torch.unique(torch.tensor(cluster_labels), return_counts=True)
        if len(a) >1:
            var+=torch.std(b.float()).item()
            if cls==28:
                std_28 = torch.std(b.float()).item()

        for center_idx in range(budgets[cls]):
            center_mask = torch.from_numpy(cluster_labels == center_idx)
            idx_center = idx[cls_mask][center_mask]
            if len(idx_center) > 0:
                val = (1.0 / len(idx_center) )*torch.ones(len(idx_center))            
                cls_indices = torch.full((len(idx_center), 1), row, dtype=torch.long)
                combined_indices = torch.cat([cls_indices, idx_center.unsqueeze(1)], dim=1)
                indices = torch.cat([indices, combined_indices], dim=0)
                values = torch.cat([values, val])
            row+=1
    size = torch.Size([args.budget, len(H)])
    mapping_norm = torch.sparse_coo_tensor(indices.t(), values, size)
    c = torch.spmm(mapping_norm.to(args.device), H.to(args.device))
    return id.astype(np.int64), torch.tensor(assignments, dtype=torch.int64).to(args.device), c, mapping_norm, [var, std_28]





def assignment_generation_ssl(H, args):
    cluster_labels = clustering_fast(H.cpu().detach().numpy(), args.ssl_num_cluster, args.clustering)
    return torch.tensor(cluster_labels, dtype=torch.int64).to(args.device)








def assignment_random(H, y, args, train_mask=None):
    budgets = args.budget_cla
    idx = torch.arange(len(H))
    indices = torch.LongTensor().new_empty((0, 2))
    values = torch.FloatTensor() 
    row = 0
    buf = 0
    assignments = []
    id  = []
    for cls in range(args.num_class):
        if train_mask is not None:
            cls_mask = (y == cls) & train_mask
        else:
            cls_mask = (y == cls)
        H_cls = H[cls_mask].cpu().detach().numpy()
        cluster_labels = clustering_fast(H_cls, budgets[cls], args.clustering)
        np.random.shuffle(cluster_labels)
        labels = cluster_labels + buf
        buf += budgets[cls]
        idx_ = idx[cls_mask]
        assignments = np.concatenate([assignments, labels])
        id = np.concatenate([id, idx_])


        for center_idx in range(budgets[cls]):
            center_mask = torch.from_numpy(cluster_labels == center_idx)
            idx_center = idx[cls_mask][center_mask]
            if len(idx_center) > 0:
                val = (1.0 / len(idx_center) )*torch.ones(len(idx_center))            
                cls_indices = torch.full((len(idx_center), 1), row, dtype=torch.long)
                combined_indices = torch.cat([cls_indices, idx_center.unsqueeze(1)], dim=1)
                indices = torch.cat([indices, combined_indices], dim=0)
                values = torch.cat([values, val])
            row+=1
    size = torch.Size([args.budget, len(H)])
    mapping_norm = torch.sparse_coo_tensor(indices.t(), values, size)
    c = torch.spmm(mapping_norm.to(args.device), H.to(args.device))
    return id.astype(np.int64), torch.tensor(assignments, dtype=torch.int64).to(args.device), c, mapping_norm


@torch.no_grad()
def initial_classifier(model, input, cls_mask, args):
    model.eval()
    Z = model(input)
    budget_cla = args.budget_cla
    means = []

    for i in range(args.num_class):
        H = Z[cls_mask[i]]
        n_clusters = budget_cla[i]
        num_samples = len(H)
        cluster_assignments = torch.randint(0, n_clusters, (num_samples,))
        for i in range(n_clusters):
            cluster_members = H[cluster_assignments == i]
            if cluster_members.shape[0] == 0:
                mean = torch.zeros(H.shape[1], dtype=H.dtype, device=H.device)
            else:
                mean = cluster_members.mean(dim=0)
                mean = mean / mean.norm(p=2) 
            means.append(mean)
    data = torch.stack(means)
    model.classifier.weight.data = data
    return model



def accumulate(budget_cla):
    # // accumulate the budget
    slide_cla = [0]
    for i in range(len(budget_cla)):
        slide_cla.append(slide_cla[-1]+budget_cla[i])
    class_table = torch.tensor(np.repeat(np.arange(len(budget_cla)), budget_cla))

    return slide_cla, class_table



def preprocess_features(npdata, pca=10):
    """Preprocess an array of features.
    Args:
    npdata (np.array N * ndim): features to preprocess
    pca (int): dim of output
    Returns:
    np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


@torch.no_grad()
def gen_label(args, data, H, model, cls_mask, slide_cla, method, class_table):
    partition_mask = data.train_mask

    if method == 'kmeans':
        buf = 0
        model.eval()
        Z = model(H) 
        idx = torch.arange(len(H))
        indices = torch.LongTensor().new_empty((0, 2))
        values = torch.FloatTensor() 
        row = 0
        Y = torch.ones(len(H), dtype=torch.long).to(args.device)

        for cls in range(args.num_class):
            cluster_labels = clustering_fast(Z[cls_mask[cls]].cpu().detach().numpy(), args.budget_cla[cls], 'kmeans')
            hard_target = torch.tensor(cluster_labels).to(args.device)
            global_target = buf+hard_target.to(args.device)
            Y[cls_mask[cls]] = global_target
            buf += args.budget_cla[cls]

            for center_idx in range(args.budget_cla[cls]):
                center_mask = torch.from_numpy(cluster_labels == center_idx)
                idx_center = idx[cls_mask[cls]][center_mask]
                if len(idx_center) > 0:
                    val = (1/len(idx_center)) * torch.ones(len(idx_center))   
                    # val = torch.ones(len(idx_center))   

                    cls_indices = torch.full((len(idx_center), 1), row, dtype=torch.long)
                    combined_indices = torch.cat([cls_indices, idx_center.unsqueeze(1)], dim=1)
                    indices = torch.cat([indices, combined_indices], dim=0)
                    values = torch.cat([values, val])
                row+=1
        size = torch.Size([args.budget, len(H)])
        M = torch.sparse_coo_tensor(indices.t(), values, size).to(args.device)
    elif method == 'OT':
        buf = 0
        model.eval()
        Z = model(H) 
        output = model.predict(Z) 

        idx = torch.arange(len(H))
        indices = torch.LongTensor().new_empty((0, 2))
        values = torch.FloatTensor() 
        row = 0
        
        Y = torch.ones(len(H), dtype=torch.long).to(args.device)

        for cls in range(args.num_class):
            source = output[cls_mask[cls], slide_cla[cls]:slide_cla[cls+1]]
            soft_target = sinkhorn(source, sinkhorn_iterations=3, epsilon=args.epsilon)
            if soft_target.isnan().sum() > 0 or source.isnan().sum() > 0:
                print('cls is', cls, soft_target.isnan().sum(), source.isnan().sum())
                exit(0)            
            hard_target = torch.argmax(soft_target, dim=1)
            a, b = torch.unique(hard_target, return_counts=True)
            if len(a) != args.budget_cla[cls]:
                print(f'{cls}th ({args.budget_cla[cls]}) class: {len(a)}, {b.cpu().numpy()}')
                exit(0)  
            global_target = buf+hard_target.to(args.device)
            Y[cls_mask[cls]] = global_target
            buf += args.budget_cla[cls]

            for center_idx in range(args.budget_cla[cls]):
                center_mask = torch.from_numpy(hard_target.cpu().numpy() == center_idx)
                idx_center = idx[cls_mask[cls]][center_mask]
                if len(idx_center) > 0:
                    val = (1/len(idx_center)) * torch.ones(len(idx_center))   
                    # val = torch.ones(len(idx_center))  
                    cls_indices = torch.full((len(idx_center), 1), row, dtype=torch.long)
                    combined_indices = torch.cat([cls_indices, idx_center.unsqueeze(1)], dim=1)
                    indices = torch.cat([indices, combined_indices], dim=0)
                    values = torch.cat([values, val])
                row+=1
        size = torch.Size([args.budget, len(H)])
        M = torch.sparse_coo_tensor(indices.t(), values, size).to(args.device)


    return Y, M, partition_mask


@torch.no_grad()
def sinkhorn(out, sinkhorn_iterations=3, epsilon=0.03):     
    Q = torch.exp(out / epsilon).t()  
    B, K = Q.shape[1], Q.shape[0]  
    Q /= Q.sum()
    for _ in range(sinkhorn_iterations):
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= K
        
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= B
    Q *= B  
    return Q.t()


@torch.no_grad()
def update_classifier(args, model, H, M):
    model.eval()
    Z = model(H)
    C=torch.spmm(M, Z.to(args.device))
    model.classifier.weight.data.copy_(C)
    return model




