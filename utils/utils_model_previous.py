import torch
from torch.nn.modules.loss import _Loss
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from e3nn.o3 import Irrep, Irreps, spherical_harmonics, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import Gate, FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from utils.utils_plot import generate_dataframe, plot_bands, plot_gphonons
torch.autograd.set_detect_anomaly(True)

class BandLoss(_Loss):
    def __init__(self, size_average = None, reduce = None, reduction: str = 'mean') -> None:
        super(BandLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.sum(torch.pow(torch.abs(input - target)/torch.max(torch.abs(target)), 2)) \
               /torch.numel(target)

class BandLoss_MAE_Norm(_Loss):
    def __init__(self, size_average = None, reduce = None, reduction: str = 'mean') -> None:
        super(BandLoss_MAE_Norm, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.sum(torch.abs(input - target)/torch.max(torch.abs(target))) \
               /torch.numel(target)

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x

class GraphConvolution(torch.nn.Module):
    def __init__(self,
                 irreps_in,
                 irreps_node_attr,
                 irreps_edge_attr,
                 irreps_out,
                 number_of_basis,
                 radial_layers,
                 radial_neurons):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_node_attr = Irreps(irreps_node_attr)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_out = Irreps(irreps_out)

        self.linear_input = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)
        self.linear_mask = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)
        
        irreps_mid = []
        instructions = []
        for i, (mul, irrep_in) in enumerate(self.irreps_in):
            for j, (_, irrep_edge_attr) in enumerate(self.irreps_edge_attr):
                for irrep_mid in irrep_in * irrep_edge_attr:
                    if irrep_mid in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, irrep_mid))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for (i_1, i_2, i_out, mode, train) in instructions]

        self.tensor_edge = TensorProduct(self.irreps_in,
                                         self.irreps_edge_attr,
                                         irreps_mid,
                                         instructions,
                                         internal_weights=False,
                                         shared_weights=False)
        
        self.edge2weight = FullyConnectedNet([number_of_basis] + radial_layers * [radial_neurons] + [self.tensor_edge.weight_numel], torch.nn.functional.silu)
        self.linear_output = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)

    def forward(self,
                node_input,
                node_attr,
                node_deg,
                edge_src,
                edge_dst,
                edge_attr,
                edge_length_embedded,
                numb, n):

        node_input_features = self.linear_input(node_input, node_attr)
        node_features = torch.div(node_input_features, torch.pow(node_deg, 0.5))

        node_mask = self.linear_mask(node_input, node_attr)

        edge_weight = self.edge2weight(edge_length_embedded)
        edge_features = self.tensor_edge(node_features[edge_src], edge_attr, edge_weight)

        node_features = scatter(edge_features, edge_dst, dim = 0, dim_size = node_features.shape[0])
        node_features = torch.div(node_features, torch.pow(node_deg, 0.5))

        node_output_features = self.linear_output(node_features, node_attr)

        node_output = node_output_features

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        mask = self.linear_mask.output_mask
        c_x = (1 - mask) + c_x * mask
        return c_s * node_mask + c_x * node_output

class GraphHamiltonianConvolution(GraphConvolution):
    def __init__(self, 
                 irreps_in, 
                 irreps_node_attr, 
                 irreps_edge_attr, 
                 irreps_out, 
                 number_of_basis, 
                 radial_layers, 
                 radial_neurons):
        super().__init__(irreps_in, 
                         irreps_node_attr, 
                         irreps_edge_attr, 
                         irreps_out, 
                         number_of_basis, 
                         radial_layers, 
                         radial_neurons)
        tr = 3 ** -0.5
        tw = 2 ** -0.5
        self.irrep2tens = torch.tensor([[    tr,  0,   0,   0,      tr,  0,  0,   0,     tr],
                                        [     0,  0,   0,   0,       0, tw,  0, -tw,      0],
                                        [     0,  0, -tw,   0,       0,  0, tw,   0,      0],
                                        [     0, tw,   0, -tw,       0,  0,  0,   0,      0],
                                        [     0,  0,  tw,   0,       0,  0, tw,   0,      0],
                                        [     0, tw,   0,  tw,       0,  0,  0,   0,      0],
                                        [-tw*tr,  0,   0,   0, 2*tw*tr,  0,  0,   0, -tw*tr],
                                        [     0,  0,   0,   0,       0, tw,  0,  tw,      0],
                                        [   -tw,  0,   0,   0,       0,  0,  0,   0,     tw]], dtype = torch.complex128)

    @staticmethod
    def glue(blocks, numb, n):
        return torch.cat(torch.cat(list(blocks), dim = 1).t().chunk(n*numb), dim = 1).t().reshape((n, 3*numb, 3*numb))

    def forward(self,
                node_input,
                node_attr,
                node_deg,
                edge_src,
                edge_dst,
                edge_attr,
                edge_length_embedded,
                numb, n):
        node_output = super().forward(node_input,
                                        node_attr,
                                        node_deg,
                                        edge_src,
                                        edge_dst,
                                        edge_attr,
                                        edge_length_embedded,
                                        numb, n)
        output = node_output[numb:].type(torch.complex128)
        output = torch.add(output[:, [0, 2, 3, 4, 8, 9, 10, 11, 12]], output[:, [1, 5, 6, 7, 13, 14, 15, 16, 17]], alpha = 1j)
        output = torch.matmul(output, self.irrep2tens.to(device = node_input.device))
        output = output.reshape((-1, 3, 3))
        Hs = self.glue(output, numb.item(), n)
        return Hs

class GraphNetwork_kMVN(torch.nn.Module):
    def __init__(self,
                 mul,
                 irreps_out,    #? irreps_node
                 lmax,
                 nlayers,
                 number_of_basis,
                 radial_layers,
                 radial_neurons,
                 node_dim,  #!
                 node_embed_dim,    #!
                 input_dim,
                 input_embed_dim    #!
                 ):
        super().__init__()
        
        self.mul = mul
        self.irreps_in = Irreps(str(input_embed_dim)+'x0e')
        self.irreps_node_attr = Irreps(str(node_embed_dim)+'x0e')
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.irreps_hidden = Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = Irreps(irreps_out)
        self.number_of_basis = number_of_basis

        act = {1: torch.nn.functional.silu,
               -1: torch.tanh}
        act_gates = {1: torch.sigmoid,
                     -1: torch.tanh}

        self.layers = torch.nn.ModuleList()
        irreps_in = self.irreps_in
        for _ in range(nlayers):
            irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps_in, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated)
            conv = GraphConvolution(irreps_in,
                                    self.irreps_node_attr,
                                    self.irreps_edge_attr,
                                    gate.irreps_in,
                                    number_of_basis,
                                    radial_layers,
                                    radial_neurons)

            irreps_in = gate.irreps_out

            self.layers.append(CustomCompose(conv, gate))

        self.layers.append(GraphHamiltonianConvolution(irreps_in,
                                                        self.irreps_node_attr,
                                                        self.irreps_edge_attr,
                                                        self.irreps_out,
                                                        number_of_basis,
                                                        radial_layers,
                                                        radial_neurons))
        self.emx = torch.nn.Linear(input_dim, input_embed_dim, dtype = torch.float64)
        self.emz = torch.nn.Linear(node_dim, node_embed_dim, dtype = torch.float64)

    def forward(self, data):
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'].item(), self.number_of_basis, basis = 'gaussian', cutoff = False)
        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization = 'component')
        edge_attr = edge_sh
        numb = data['numb']
        x = torch.relu(self.emx(torch.relu(data['x'])))
        z = torch.relu(self.emz(torch.relu(data['z'])))
        node_deg = data['node_deg']
        ucs = data['ucs'][0]    #!
        n = len(ucs.shift_reverse)  #!
        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, n) #!
        return x, torch.tensor(ucs.shift_reverse, dtype = torch.complex128).to(device = x.device)

def get_spectra(Hs, shifts, qpts):
    H = torch.sum(torch.mul(Hs.unsqueeze(1), torch.exp(2j*math.pi*torch.matmul(shifts, qpts.type(torch.complex128).t())).unsqueeze(-1).unsqueeze(-1)), dim = 0)
    eigvals = torch.linalg.eigvals(H)
    abx = torch.abs(eigvals)
    try:
        epsilon = torch.min(abx[abx > 0])/100
    except:
        epsilon = 1E-8
    eigvals = torch.sqrt(eigvals + epsilon)
    return torch.sort(torch.real(eigvals))[0]


class GraphNetwork_MVN(torch.nn.Module):
    def __init__(self,
                 mul,
                 irreps_out,    #? irreps_node
                 lmax,
                 nlayers,
                 number_of_basis,
                 radial_layers,
                 radial_neurons,
                 node_dim,  #!
                 node_embed_dim,    #!
                 input_dim,
                 input_embed_dim    #!
                 ):
        super().__init__()
        
        self.mul = mul
        self.irreps_in = Irreps(str(input_embed_dim)+'x0e')
        self.irreps_node_attr = Irreps(str(node_embed_dim)+'x0e')
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.irreps_hidden = Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = Irreps(irreps_out)
        self.number_of_basis = number_of_basis

        act = {1: torch.nn.functional.silu,
               -1: torch.tanh}
        act_gates = {1: torch.sigmoid,
                     -1: torch.tanh}

        self.layers = torch.nn.ModuleList()
        irreps_in = self.irreps_in
        for _ in range(nlayers):
            irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps_in, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated)
            conv = GraphConvolution(irreps_in,
                                    self.irreps_node_attr,
                                    self.irreps_edge_attr,
                                    gate.irreps_in,
                                    number_of_basis,
                                    radial_layers,
                                    radial_neurons)

            irreps_in = gate.irreps_out

            self.layers.append(CustomCompose(conv, gate))

        self.layers.append(GraphHamiltonianConvolution(irreps_in,
                                                        self.irreps_node_attr,
                                                        self.irreps_edge_attr,
                                                        self.irreps_out,
                                                        number_of_basis,
                                                        radial_layers,
                                                        radial_neurons))
        self.emx = torch.nn.Linear(input_dim, input_embed_dim, dtype = torch.float64)
        self.emz = torch.nn.Linear(node_dim, node_embed_dim, dtype = torch.float64)

    def forward(self, data):
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'].item(), self.number_of_basis, basis = 'gaussian', cutoff = False)
        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization = 'component')
        edge_attr = edge_sh
        # m = data['m'] #?
        m = 1
        numb = data['numb']
        x = torch.relu(self.emx(torch.relu(data['x'])))
        z = torch.relu(self.emz(torch.relu(data['z'])))
        node_deg = data['node_deg']
        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, m)
        eigvals = torch.linalg.eigvals(x)
        abx = torch.abs(eigvals)
        try:
            epsilon = torch.min(abx[abx > 0])/100
        except:
            epsilon = 1E-8
        eigvals = torch.sqrt(eigvals + epsilon)
        # order = torch.sort(torch.real(eigvals))[1]
        # output = eigvals[order]
        # sort the eigenvalues and get as output
        output = torch.sort(torch.real(eigvals))[0]
        return output

class GraphNetwork_VVN(torch.nn.Module):
    def __init__(self,
                 mul,
                 irreps_out,
                 lmax,
                 nlayers,
                 number_of_basis,
                 radial_layers,
                 radial_neurons,
                 node_dim,
                 node_embed_dim,
                 input_dim,
                 input_embed_dim):
        super().__init__()
        
        self.mul = mul
        self.irreps_in = Irreps(str(input_embed_dim)+'x0e')
        self.irreps_node_attr = Irreps(str(node_embed_dim)+'x0e')
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.irreps_hidden = Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = Irreps(irreps_out)
        self.number_of_basis = number_of_basis

        act = {1: torch.nn.functional.silu,
               -1: torch.tanh}
        act_gates = {1: torch.sigmoid,
                     -1: torch.tanh}

        self.layers = torch.nn.ModuleList()
        irreps_in = self.irreps_in
        for _ in range(nlayers):
            irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps_in, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated)
            conv = GraphConvolution(irreps_in,
                                    self.irreps_node_attr,
                                    self.irreps_edge_attr,
                                    gate.irreps_in,
                                    number_of_basis,
                                    radial_layers,
                                    radial_neurons)

            irreps_in = gate.irreps_out

            self.layers.append(CustomCompose(conv, gate))
        #last layer: conv
        self.layers.append(GraphConvolution(irreps_in,
                        self.irreps_node_attr,
                        self.irreps_edge_attr,
                        self.irreps_out,
                        number_of_basis,
                        radial_layers,
                        radial_neurons,)
                        )

        self.emx = torch.nn.Linear(input_dim, input_embed_dim, dtype = torch.float64)
        self.emz = torch.nn.Linear(node_dim, node_embed_dim, dtype = torch.float64)

    def forward(self, data):
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'].item(), self.number_of_basis, basis = 'gaussian', cutoff = False)
        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization = 'component')
        edge_attr = edge_sh
        numb = data['numb']
        x = torch.relu(self.emx(torch.relu(data['x'])))
        z = torch.relu(self.emz(torch.relu(data['z'])))
        node_deg = data['node_deg']
        n=None
        count = 0
        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, n)
            count += 1
        x = x.reshape((1, -1))[:, numb:]
        return x


def evaluate(model, dataloader, loss_fn, device, option='kmvn'):
    model.eval()
    loss_cumulative = 0.
    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            if option in ['kmvn']:
                Hs, shifts = model(d)
                output = get_spectra(Hs, shifts, d.qpts)
            else:
                output = model(d)
            loss = loss_fn(output, d.y).cpu()
            loss_cumulative += loss.detach().item()
    return loss_cumulative/len(dataloader)

def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))

def train(model,
          opt,
          tr_set,
          tr_nums,
          te_set,
          loss_fn,
          run_name,
          max_iter,
          scheduler,
          device,
          batch_size,
          k_fold,
          option='kmvn'):
    from utils.utils_plot import loss_plot, loss_test_plot
    model.to(device)
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()

    record_lines = []

    try:
        print('Use model.load_state_dict to load the existing model: ' + run_name + '.torch')
        model.load_state_dict(torch.load(run_name + '.torch')['state'])
    except:
        print('There is no existing model')
        results = {}
        history = []
        s0 = 0
    else:
        print('Use torch.load to load the existing model: ' + run_name + '.torch')
        results = torch.load(run_name + '.torch')
        history = results['history']
        s0 = history[-1]['step'] + 1

    tr_sets = torch.utils.data.random_split(tr_set, tr_nums)
    te_loader = DataLoader(te_set, batch_size = batch_size)
    for step in range(max_iter):
        k = step % k_fold
        tr_loader = DataLoader(torch.utils.data.ConcatDataset(tr_sets[:k] + tr_sets[k+1:]), batch_size = batch_size, shuffle=True)
        va_loader = DataLoader(tr_sets[k], batch_size = batch_size)
        model.train()
        N = len(tr_loader)
        for i, d in enumerate(tr_loader):
            start = time.time()
            d.to(device)
            if option in ['kmvn']:
                Hs, shifts = model(d)
                output = get_spectra(Hs, shifts, d.qpts)
            else:
                output = model(d)
            loss = loss_fn(output, d.y).cpu()
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'num {i+1:4d}/{N}, loss = {loss}, train time = {time.time() - start}', end = '\r')

        end_time = time.time()
        wall = end_time - start_time
        print(wall)
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            valid_avg_loss = evaluate(model, va_loader, loss_fn, device, option)
            train_avg_loss = evaluate(model, tr_loader, loss_fn, device, option)

            history.append({
                            'step': s0 + step,
                            'wall': wall,
                            'batch': {
                                    'loss': loss.item(),
                                    },
                            'valid': {
                                    'loss': valid_avg_loss,
                                    },
                            'train': {
                                    'loss': train_avg_loss,
                                    },
                           })

            results = {
                        'history': history,
                        'state': model.state_dict()
                      }

            print(f"Iteration {step+1:4d}   " +
                  f"train loss = {train_avg_loss:8.20f}   " +
                  f"valid loss = {valid_avg_loss:8.20f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

            with open(f'./models/{run_name}.torch', 'wb') as f:
                torch.save(results, f)

            record_line = '%d\t%.20f\t%.20f'%(step,train_avg_loss,valid_avg_loss)
            record_lines.append(record_line)
            loss_plot('./models/' + run_name, device, './models/' + run_name)
            loss_test_plot(model, device, './models/' + run_name, te_loader, loss_fn, option)
            df_tr = generate_dataframe(model, tr_loader, loss_fn, device, option)
            df_te = generate_dataframe(model, te_loader, loss_fn, device, option)
            palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
            if option == 'kmvn':
                plot_bands(df_tr, header='./models/' + run_name, title='train', n=6, m=2, palette=palette)
                plot_bands(df_te, header='./models/' + run_name, title='test', n=6, m=2, palette=palette)
            elif option in ['mvn', 'vvn']:
                plot_gphonons(df_tr, header='./models/' + run_name, title='train', n=6, m=2, lwidth=0.5, windowsize=(4, 2), palette=palette, formula=True)
                plot_gphonons(df_te, header='./models/' + run_name, title='test', n=6, m=2, lwidth=0.5, windowsize=(4, 2), palette=palette, formula=True)
        text_file = open('./models/' + run_name + ".txt", "w")
        for line in record_lines:
            text_file.write(line + "\n")
        text_file.close()

        if scheduler is not None:
            scheduler.step()



