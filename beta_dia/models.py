import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    profile
except:
    profile = lambda x: x

class DeepMap(nn.Module):
    '''In paper, it's also called DeepProfile'''
    def __init__(self, map_channels, nn_in_features=220):
        super(DeepMap, self).__init__()

        # 10@23*92 -- 4@21*90 -- 4@10*45
        conv1_in_channels = map_channels
        conv1_out_channels = 16
        conv2_in_channels = conv1_out_channels
        conv2_out_channels = 20
        self.nn_in_features = nn_in_features  # 13*50--220
        nn_out_features = 16

        self.embed = nn.Embedding(map_channels, nn_out_features)  # valid num

        # elution
        self.elution_conv1 = nn.Conv2d(
            in_channels=conv1_in_channels,
            out_channels=conv1_out_channels,
            kernel_size=3
        )
        self.elution_relu1 = nn.ReLU()
        self.elution_max_pool_1 = nn.MaxPool2d(kernel_size=2)
        # 4@10*45 -- 2@8*43 -- 2@4*21
        self.elution_conv2 = nn.Conv2d(
            in_channels=conv2_in_channels,
            out_channels=conv2_out_channels,
            kernel_size=3
        )
        self.elution_relu2 = nn.ReLU()
        self.elution_max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.elution_fc = nn.Linear(
            in_features=self.nn_in_features,
            out_features=nn_out_features
        )

        # ratio
        self.ratio_conv1 = nn.Conv2d(
            in_channels=conv1_in_channels,
            out_channels=conv1_out_channels,
            kernel_size=3
        )
        self.ratio_relu1 = nn.ReLU()
        self.ratio_max_pool_1 = nn.MaxPool2d(kernel_size=2)
        # 4@10*45 -- 2@8*43 -- 2@4*21
        self.ratio_conv2 = nn.Conv2d(
            in_channels=conv2_in_channels,
            out_channels=conv2_out_channels,
            kernel_size=3
        )
        self.ratio_relu2 = nn.ReLU()
        self.ratio_max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.ratio_fc = nn.Linear(
            in_features=self.nn_in_features,
            out_features=nn_out_features
        )

        # concat features
        self.fc1 = nn.Linear(in_features=3 * nn_out_features,
                             out_features=nn_out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=nn_out_features, out_features=2)

    # @profile
    def forward(self, maps, batch_valid_num):
        # two normalization methods
        e = 1e-7
        maps_elution = maps / (torch.amax(maps, dim=(2, 3), keepdim=True) + e)
        maps_ratio = maps / (torch.amax(maps, dim=(1, 2, 3), keepdim=True) + e)

        # embed of valid_num
        embed = self.embed(batch_valid_num - 1)

        # elution
        maps_elution = self.elution_conv1(maps_elution)
        maps_elution = self.elution_relu1(maps_elution)
        maps_elution = self.elution_max_pool_1(maps_elution)
        maps_elution = self.elution_conv2(maps_elution)
        maps_elution = self.elution_relu2(maps_elution)
        maps_elution = self.elution_max_pool_2(maps_elution)
        maps_elution = maps_elution.view(-1, self.nn_in_features)
        maps_elution = self.elution_fc(maps_elution)

        # ratio
        maps_ratio = self.ratio_conv1(maps_ratio)
        maps_ratio = self.ratio_relu1(maps_ratio)
        maps_ratio = self.ratio_max_pool_1(maps_ratio)
        maps_ratio = self.ratio_conv2(maps_ratio)
        maps_ratio = self.ratio_relu2(maps_ratio)
        maps_ratio = self.ratio_max_pool_2(maps_ratio)
        maps_ratio = maps_ratio.view(-1, self.nn_in_features)
        maps_ratio = self.ratio_fc(maps_ratio)

        # concat
        feature_map = torch.cat([maps_elution, maps_ratio], dim=1)
        feature_all = torch.cat([embed, maps_elution, maps_ratio], dim=1)

        # class
        x = self.fc1(feature_all)
        x = self.relu(x)
        x = self.dropout(x)
        result = self.fc2(x)

        return feature_map, result


class DeepMall(nn.Module):

    def __init__(self, input_dim, feature_dim):
        super(DeepMall, self).__init__()
        lstm_out_dim = 256
        att_size = 256
        # seq
        self.xic_gru = nn.GRU(batch_first=True,
                              bidirectional=True,
                              num_layers=2,
                              input_size=input_dim,
                              hidden_size=int(lstm_out_dim / 2),
                              # dropout=0.2,
                              )
        # attention
        self.attention = nn.Linear(lstm_out_dim, att_size)
        self.context = nn.Linear(att_size, 1, bias=False)

        # fc for classify
        # self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.fc1 = nn.Linear(lstm_out_dim, feature_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feature_dim, 2)

    # @profile
    def forward(self, batch_mall, batch_valid_num):
        batch_mall = batch_mall.permute((0, 2, 1))

        # self.xic_gru.flatten_parameters()
        batch_mall = pack_padded_sequence(batch_mall,
                                          batch_valid_num.cpu(),
                                          batch_first=True,
                                          enforce_sorted=False)
        outputs, _ = self.xic_gru(batch_mall)

        # attention
        # [batch_size, batch_lens, att_size]
        att_w = torch.tanh(self.attention(outputs.data))
        att_w = self.context(att_w).squeeze(1)  # [batch_size*batch_lens]
        max_w = att_w.max()
        att_w = torch.exp(att_w - max_w)
        att_w, _ = pad_packed_sequence(
            PackedSequence(data=att_w,
                           batch_sizes=outputs.batch_sizes,
                           sorted_indices=outputs.sorted_indices,
                           unsorted_indices=outputs.unsorted_indices
            ), batch_first=True
        )
        # [batch_size, max_lens]
        alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # [batch_size, out_dim]
        outputs = (outputs * alphas.unsqueeze(2)).sum(dim=1)

        # norm
        # outputs = self.layer_norm(outputs)

        # fc
        feature = self.fc1(outputs)
        result = self.fc2(self.relu(feature))

        return feature, result
