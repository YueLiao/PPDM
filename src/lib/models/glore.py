import torch
import torch.nn as nn



class GraphConv1d(nn.Module):
    """Conducting reasoning on graph data"""

    def __init__(self, nodes, channels):
        super(GraphConv1d, self).__init__()
        self.node_conv = nn.Sequential(
            nn.Conv1d(nodes, nodes, kernel_size=1),
            nn.BatchNorm1d(nodes),
            nn.ReLU(inplace=True)
        )
        self.feat_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        identity = x                            # [N, channel_feature, channel_node]

        out = torch.transpose(x, 1, 2)          # [N, channel_node, channel_feature]
        out = self.node_conv(out)               # [N, channel_node, channel_feature]
        out = torch.transpose(out, 1, 2)        # [N, channel_feature, channel_node]
        # In yunpeng's implementation, it's element-wise sum
        # While in his paper, it's element-wise subtract
        out = out + identity

        out = self.feat_conv(out)             # [N, channel_feature, channel_node]

        return out


class GloRe(nn.Module):
    """Global Reasoning Unit"""

    def __init__(self, channel_in, channel_reduced, channel_node):
        super(GloRe, self).__init__()
        # reduce dim
        self.phy = nn.Sequential(nn.Conv2d(channel_in, channel_reduced, 1),
                                 nn.BatchNorm2d(channel_reduced),
                                 nn.ReLU(inplace=True))
        # projection Matrix
        self.theta = nn.Sequential(nn.Conv2d(channel_in, channel_node, 1),
                                   nn.BatchNorm2d(channel_node),
                                   nn.ReLU(inplace=True))
        # extend dim
        self.extend = nn.Sequential(nn.Conv2d(channel_reduced, channel_in, 1),
                                    nn.BatchNorm2d(channel_in))
        # global reasoning
        self.reasoning = GraphConv1d(channel_node, channel_reduced)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x                          # [batch, channel_in, H, W]
        height, width = x.size(2), x.size(3)

        # reduce dim
        reduced_x = self.phy(x)               # [batch, channel_reduced, H, W]
        reduced_x = reduced_x.view(reduced_x.size(0), reduced_x.size(1), -1)  # [batch, channel_reduced, H * W]

        # projection matrix
        B = self.theta(x)                     # [batch, channel_node, H, W]
        B = B.view(B.size(0), B.size(1), -1)  # [batch, channel_node, H * W]
        B = torch.transpose(B, 1, 2)          # [batch, H * W, channel_node]

        # grid to graph data
        # channel_node in dim 1
        V = torch.bmm(reduced_x, B)           # [batch, channel_reduced, channel_node]

        # graph reasoning
        V = self.reasoning(V)                 # [batch, channel_reduced, channel_node]

        # graph to grid data
        B = torch.transpose(B, 1, 2)          # [batch, channel_node, H * W]
        V = torch.bmm(V, B)                   # [batch, channel_reduced, H * W]
        V = V.view(V.size(0), V.size(1), height, width)  # [batch, channel_reduced, H, W]

        # extend dim
        V = self.extend(V)                    # [batch, channel_in, H, W]

        V += identity                         # [batch, channel_in, H, W]
        V = self.relu(V)

        return V

