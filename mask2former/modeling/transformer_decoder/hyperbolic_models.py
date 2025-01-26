import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .gcn.manifolds.lmath import expmap, logmap

class HyperbolicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, manifold, curvature=-1.0):
        super().__init__()
        self.manifold = manifold
        self.curvature = curvature  # Curvatura negativa del manifold
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        # Trasformazione nello spazio iperbolico
        u = x / torch.norm(x, dim=-1, keepdim=True)  # Vettore di velocità unitario
        x = expmap(x, u, k=self.curvature)  # Passa anche k

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = torch.tanh(x)  # Attivazione iperbolica

        # Trasformazione inversa nello spazio euclideo
        origin = torch.zeros_like(x)  # Origine sullo spazio iperbolico
        origin[..., -1] = 1
        x = logmap(origin, x, k=self.curvature)  # Usa l'origine come base
        return x


class HyperbolicKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """
        A linear layer for hyperbolic KAN that supports curvature scaling.

        Args:
            in_features: Input dimensionality.
            out_features: Output dimensionality.
            grid_size: Grid size for B-spline basis.
            spline_order: Order of the B-spline.
            scale_noise, scale_base, scale_spline: Scaling parameters.
            enable_standalone_scale_spline: Enable standalone scaling of spline weights.
            base_activation: Activation function used for the base weight.
            grid_eps: Grid perturbation factor.
            grid_range: Range of the grid for B-splines.
        """
        super(HyperbolicKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Initialize grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def forward(self, x: torch.Tensor, curvature: float):
        """
        Forward pass with curvature adjustment.

        Args:
            x (torch.Tensor): Input tensor.
            curvature (float): Curvature parameter for hyperbolic geometry.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape((-1, self.in_features))

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).reshape((x.size(0), -1)),
            self.spline_weight.reshape((self.out_features, -1)),
        )
        output = base_output + spline_output

        # Adjust by curvature
        # Adjust by curvature
        output = output / torch.sqrt(torch.tensor(abs(curvature) + 1e-6, device=x.device))

        output = output.reshape((*original_shape[:-1], self.out_features))
        return output

        # output = output.reshape((*original_shape[:-1], self.out_features))
        # return output

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()


class HyperbolicKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        curvature=-1,
    ):
        super(HyperbolicKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                HyperbolicKANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, curvature: float, update_grid=False):
        """
        Forward pass through the hyperbolic KAN layers.

        Args:
            x (torch.Tensor): Input tensor.
            curvature (float): Curvature parameter for hyperbolic geometry.
            update_grid (bool): Whether to update the grid for B-splines.

        Returns:
            torch.Tensor: Transformed tensor.
        """

        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x, curvature)  # Pass the curvature here
        return x

        # for layer in self.layers:
        #     if update_grid:
        #         layer.update_grid(x)
        #     x = layer(x, curvature)
        # return x

class LeviCivitaKANLayer(nn.Module):
    def __init__(self, in_features, out_features, curvature=-1.0, dropout_rate=0.5):
        super(LeviCivitaKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature

        # Levi-Civita connection weights
        self.levicivita_weights = nn.Parameter(torch.Tensor(out_features, in_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Initialization
        nn.init.kaiming_uniform_(self.levicivita_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Batch normalization and dropout
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(p=dropout_rate)

    def directional_derivative(self, x):
        # Assumiamo che x abbia dimensioni [batch_size, num_queries, feature_dim]
        batch_size, num_queries, feature_dim = x.size()
        output = torch.zeros((batch_size, num_queries, self.out_features), device=x.device)

        for i in range(self.out_features):
            for j in range(feature_dim):
                for k in range(feature_dim):
                    if j != k:  # Considera solo prodotti incrociati
                        connection = self.levicivita_weights[i, j, k] * x[:, :, j] * x[:, :, k]

                        # Denominatore con corretta gestione dimensionale
                        denom = 1 + self.curvature * torch.norm(x[:, :, j] * x[:, :, k], dim=1, keepdim=True) ** 2
                        denom = denom.expand_as(connection)

                        # Aggiorna l'output
                        output[:, :, i] += connection / denom

        return output

    def forward(self, x):
        levicivita_output = self.directional_derivative(x)
        base_output = F.linear(x, self.weight1, self.bias)
        combined_output = levicivita_output + base_output
        combined_output = self.bn(combined_output)
        return self.dropout(combined_output)

class HyperbolicLKAN(nn.Module):
    def __init__(self, layers_hidden, curvature=-1.0, dropout_rate=0.5):
        super(HyperbolicLKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(LeviCivitaKANLayer(in_features, out_features, curvature, dropout_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x