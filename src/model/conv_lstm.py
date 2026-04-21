import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=3, num_layers=2, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dims[i],
                    kernel_size=kernel_size,
                    bias=bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        b, t, c, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        return layer_output_list[-1], last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self._init_hidden_layer(batch_size, image_size, i))
        return init_states

    def _init_hidden_layer(self, batch_size, image_size, layer_idx):
        h, w = image_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (
            torch.zeros(batch_size, self.hidden_dims[layer_idx], h, w).to(device),
            torch.zeros(batch_size, self.hidden_dims[layer_idx], h, w).to(device),
        )


class AdriaticPredictor(nn.Module):
    def __init__(
        self, input_dim=6, hidden_dims=None, seasonal_dim=2, output_dim=2, prediction_horizon=1
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.prediction_horizon = prediction_horizon
        self.output_dim = output_dim

        # ConvLSTM for spatiotemporal features
        self.convlstm = ConvLSTM(
            input_dim=input_dim, hidden_dims=hidden_dims, num_layers=len(hidden_dims)
        )

        # Seasonal context processing
        self.seasonal_fc = nn.Sequential(
            nn.Linear(seasonal_dim, 16), nn.ReLU(), nn.Linear(16, hidden_dims[-1])
        )

        # Output layer - now outputs for all prediction horizons
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_dim * prediction_horizon, kernel_size=1),
        )

    def forward(self, x, seasonal):
        batch_size, seq_len, _, H, W = x.shape

        # Process through ConvLSTM
        lstm_out, _ = self.convlstm(x)  # [batch, seq_len, hidden_dim, H, W]

        # Use last timestep
        lstm_features = lstm_out[:, -1]  # [batch, hidden_dim, H, W]

        # Process seasonal context (use last timestep)
        seasonal_last = seasonal[:, -1]  # [batch, seasonal_dim]
        seasonal_emb = self.seasonal_fc(seasonal_last)  # [batch, hidden_dim]

        # Add seasonal context to spatial features
        seasonal_emb = seasonal_emb.unsqueeze(-1).unsqueeze(-1)  # [batch, hidden_dim, 1, 1]
        combined_features = lstm_features + seasonal_emb

        # Generate prediction for all horizons
        output = self.output_conv(
            combined_features
        )  # [batch, output_dim * prediction_horizon, H, W]

        # Reshape to separate timesteps: [batch, prediction_horizon, output_dim, H, W]
        output = output.view(batch_size, self.prediction_horizon, self.output_dim, H, W)

        return output
