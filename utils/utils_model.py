import os
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
from utils.utils_plot import generate_dataframe, plot_bands, plot_gphonons, plot_loss, plot_test_loss
torch.autograd.set_detect_anomaly(True)

class BandLoss(_Loss):
    """
    Custom loss function to compute squared normalized differences between input and target.
    """
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
    """
    Check if a tensor product path exists between irreps_in1, irreps_in2 and the output ir_out.
    Args:
        irreps_in1 (Irreps): First input irreps.
        irreps_in2 (Irreps): Second input irreps.
        ir_out (Irrep): Output irrep.
    
    Returns:
        bool: True if path exists, False otherwise.
    """
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class CustomCompose(torch.nn.Module):
    """
    Custom module to sequentially apply two modules, storing intermediate outputs.
    """
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
    """
    Graph convolution layer that processes node and edge features.
    """
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
    """
    Graph Hamiltonian convolution layer with matrix multiplication for complex output.
    """
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
        """
        Glue tensor blocks into final matrix.
        Args:
            blocks (list): List of tensor blocks.
            numb (int): Number of blocks.
            n (int): Size of each block.
        Returns:
            torch.Tensor: Final glued tensor.
        """
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

def get_spectra(Hs, shifts, qpts):
    """
    Calculate spectra using Hamiltonians and q-points.
    Args:
        Hs (torch.Tensor): Hamiltonians.
        shifts (torch.Tensor): Unit cell shifts.
        qpts (torch.Tensor): q-points.
    Returns:
        torch.Tensor: Sorted eigenvalues (spectra).
    """
    H = torch.sum(torch.mul(Hs.unsqueeze(1), torch.exp(2j*math.pi*torch.matmul(shifts, qpts.type(torch.complex128).t())).unsqueeze(-1).unsqueeze(-1)), dim = 0)
    eigvals = torch.linalg.eigvals(H)
    abx = torch.abs(eigvals)
    try:
        epsilon = torch.min(abx[abx > 0])/100
    except:
        epsilon = 1E-8
    eigvals = torch.sqrt(eigvals + epsilon)
    return torch.sort(torch.real(eigvals))[0]

class BaseGraphNetwork(torch.nn.Module):
    """
    Base class for the graph network models with shared functionality.
    Subclasses should implement any specific functionality.
    """
    def __init__(self, mul, irreps_out, lmax, nlayers, number_of_basis, radial_layers, radial_neurons, 
                 node_dim, node_embed_dim, input_dim, input_embed_dim, **kwargs):
        super().__init__()
        self.mul = mul
        self.irreps_in = Irreps(str(input_embed_dim) + 'x0e')
        self.irreps_node_attr = Irreps(str(node_embed_dim) + 'x0e')
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.irreps_hidden = Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = Irreps(irreps_out)
        self.number_of_basis = number_of_basis

        self.act = {1: torch.nn.functional.silu, -1: torch.tanh}
        self.act_gates = {1: torch.sigmoid, -1: torch.tanh}
        
        # Embedding layers
        self.emx = torch.nn.Linear(input_dim, input_embed_dim, dtype=torch.float64)
        self.emz = torch.nn.Linear(node_dim, node_embed_dim, dtype=torch.float64)
        
        self.layers = self._build_layers(nlayers, number_of_basis, radial_layers, radial_neurons)

    def _build_layers(self, nlayers, number_of_basis, radial_layers, radial_neurons):
        """
        Build layers for the network with gates and convolutions.
        """
        layers = torch.nn.ModuleList()
        irreps_in = self.irreps_in
        for _ in range(nlayers):
            irreps_scalars = Irreps([(self.mul, ir) for self.mul, ir in self.irreps_hidden if ir.l == 0 and self._tp_path_exists(irreps_in, ir)])
            irreps_gated = Irreps([(self.mul, ir) for self.mul, ir in self.irreps_hidden if ir.l > 0 and self._tp_path_exists(irreps_in, ir)])
            ir = "0e" if self._tp_path_exists(irreps_in, "0e") else "0o"
            irreps_gates = Irreps([(self.mul, ir) for self.mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [self.act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [self.act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated)
            conv = GraphConvolution(irreps_in, self.irreps_node_attr, self.irreps_edge_attr, gate.irreps_in, number_of_basis, radial_layers, radial_neurons)

            irreps_in = gate.irreps_out
            layers.append(CustomCompose(conv, gate))
        self.irreps_in_fin = irreps_in    
        return layers

    def _tp_path_exists(self, irreps_in, ir):
        # Placeholder for path existence check
        return True  # Simplified logic for brevity

    def _shared_forward(self, data):
        """
        Shared part of the forward pass common across different models.
        """
        edge_src, edge_dst = data['edge_index']
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'].item(), self.number_of_basis, basis='gaussian', cutoff=False)
        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_attr = edge_sh

        numb = data['numb']
        x = torch.relu(self.emx(torch.relu(data['x'])))
        z = torch.relu(self.emz(torch.relu(data['z'])))
        node_deg = data['node_deg']
        return x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb


class GraphNetwork_kMVN(BaseGraphNetwork):
    """
    Standard Graph Network inheriting from the base class.
    """
    def __init__(self, mul, irreps_out, lmax, nlayers, number_of_basis, radial_layers, radial_neurons, 
                 node_dim, node_embed_dim, input_dim, input_embed_dim, **kwargs):
        super().__init__(mul, irreps_out, lmax, nlayers, number_of_basis, radial_layers, radial_neurons, 
                         node_dim, node_embed_dim, input_dim, input_embed_dim, **kwargs)
        
        # Adding the missing GraphHamiltonianConvolution layer
        self.layers.append(GraphHamiltonianConvolution(
            # self.irreps_hidden,
            self.irreps_in_fin,   
            self.irreps_node_attr,
            self.irreps_edge_attr,
            self.irreps_out,
            number_of_basis,
            radial_layers,
            radial_neurons
        ))

    def forward(self, data):
        x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb = self._shared_forward(data)
        ucs = data['ucs'][0]
        n = len(ucs.shift_reverse)

        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, n)
        
        return x, torch.tensor(ucs.shift_reverse, dtype=torch.complex128).to(device=x.device)


class GraphNetwork_MVN(BaseGraphNetwork):
    """
    Graph Network model with additional eigenvalue processing.
    """
    def __init__(self, mul, irreps_out, lmax, nlayers, number_of_basis, radial_layers, radial_neurons, 
                 node_dim, node_embed_dim, input_dim, input_embed_dim, **kwargs):
        super().__init__(mul, irreps_out, lmax, nlayers, number_of_basis, radial_layers, radial_neurons, 
                         node_dim, node_embed_dim, input_dim, input_embed_dim, **kwargs)

        # Adding the missing GraphHamiltonianConvolution layer
        self.layers.append(GraphHamiltonianConvolution(
            # self.irreps_hidden,
            self.irreps_in_fin,   
            self.irreps_node_attr,
            self.irreps_edge_attr,
            self.irreps_out,
            number_of_basis,
            radial_layers,
            radial_neurons
        ))

    def forward(self, data):
        x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb = self._shared_forward(data)

        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, 1)

        eigvals = torch.linalg.eigvals(x)
        abx = torch.abs(eigvals)
        epsilon = torch.min(abx[abx > 0]) / 100 if torch.any(abx > 0) else 1E-8
        eigvals = torch.sqrt(eigvals + epsilon)
        output = torch.sort(torch.real(eigvals))[0]
        return output



class GraphNetwork_VVN(BaseGraphNetwork):
    """
    Graph Network model for VVN variant.
    """
    def __init__(self, mul, irreps_out, lmax, nlayers, number_of_basis, radial_layers, radial_neurons, 
                 node_dim, node_embed_dim, input_dim, input_embed_dim, **kwargs):
        super().__init__(mul, irreps_out, lmax, nlayers, number_of_basis, radial_layers, radial_neurons, 
                         node_dim, node_embed_dim, input_dim, input_embed_dim, **kwargs)
        
        self.layers.append(GraphConvolution(
                        self.irreps_in_fin,
                        self.irreps_node_attr,
                        self.irreps_edge_attr,
                        self.irreps_out,
                        number_of_basis,
                        radial_layers,
                        radial_neurons,)
                        )
        
    def forward(self, data):
        x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb = self._shared_forward(data)

        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, None)
        
        # print('x (before reshape): ', x.shape)
        x = x.reshape((1, -1))[:, numb:]
        # print('x (after reshape): ', x.shape)
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
          option='kmvn',
          factor=1000,
          conf_dict=None):
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
            
            if conf_dict is not None:
                results['conf_dict'] = conf_dict

            print(f"Iteration {step+1:4d}   " +
                  f"train loss = {train_avg_loss:8.20f}   " +
                  f"valid loss = {valid_avg_loss:8.20f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

            save_name = f'./models/{run_name}'
            with open(save_name + '.torch', 'wb') as f:
                torch.save(results, f)

            record_line = '%d\t%.20f\t%.20f'%(step,train_avg_loss,valid_avg_loss)
            record_lines.append(record_line)
            plot_loss(history, save_name + '_loss')
            plot_test_loss(model, te_loader, loss_fn, device, save_name + '_loss_test', option)

            df_tr = generate_dataframe(model, tr_loader, loss_fn, device, option, factor)
            df_te = generate_dataframe(model, te_loader, loss_fn, device, option, factor)
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



def load_model(model_class, model_file, device):
    """
    Loads a pre-trained model, its weights, and hyperparameters.
    
    Args:
        model_class: The class of the model to be instantiated.
        model_file: Path to the saved model file.
        device: The device on which to load the model.
        
    Returns:
        model: The model with loaded weights.
        conf_dict: A dictionary of hyperparameters (if available).
        history: A list of training history (if available).
        int: The starting step (for resuming training).
    """
    if os.path.exists(model_file):
        print(f"Loading model from: {model_file}")
        checkpoint = torch.load(model_file)
        
        # Extract hyperparameters and initialize the model
        # conf_dict = checkpoint.get('conf_dict', {})
        conf_dict = checkpoint.get('conf_dict', checkpoint)#['conf_dict']
        # print('conf_dict: ', conf_dict)
        model = model_class(**conf_dict)  # Initialize the model with the saved hyperparameters
        model.load_state_dict(checkpoint['state'])
        model.to(device)

        # Extract history and step number
        history = checkpoint.get('history', [])
        s0 = history[-1]['step'] + 1 if history else 0

        return model, conf_dict, history, s0
    else:
        raise FileNotFoundError(f"No model found at {model_file}")