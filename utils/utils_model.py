import os
import torch
from torch_scatter import scatter
from e3nn.o3 import Irrep, Irreps, spherical_harmonics, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import Gate, FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
import math
import time

torch.autograd.set_detect_anomaly(True)
palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
seedn = 42


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
            irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            ir = '0e' if tp_path_exists(irreps_in, self.irreps_edge_attr, '0e') else '0o'
            irreps_gates = Irreps([(self.mul, ir) for self.mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [self.act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [self.act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated)
            conv = GraphConvolution(irreps_in, self.irreps_node_attr, self.irreps_edge_attr, gate.irreps_in, number_of_basis, radial_layers, radial_neurons)

            irreps_in = gate.irreps_out
            layers.append(CustomCompose(conv, gate))
        self.irreps_in_fin = irreps_in    
        return layers

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
        
        x = x.reshape((1, -1))[:, numb:]
        return x


