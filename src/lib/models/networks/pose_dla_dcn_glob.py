from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .pose_dla_dcn import *


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])
        self.heads = heads
        self.glore = GloRe(channels[self.first_level], channels[self.first_level] // 2,
                           channels[self.first_level] // 4)
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                if 'rel' in head:
                    fc = nn.Sequential(
                        nn.Conv2d(channels[self.first_level], head_conv,
                                  kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, classes,
                                  kernel_size=final_kernel, stride=1,
                                  padding=final_kernel // 2, bias=True))
                else:
                    fc = nn.Sequential(
                        nn.Conv2d(channels[self.first_level], head_conv,
                                  kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, classes,
                                  kernel_size=final_kernel, stride=1,
                                  padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        glob_feat = self.glore(y[-1])
        for head in self.heads:
            if head in ['hm_rel', 'sub_offset', 'obj_offset']:
                z[head] = self.__getattr__(head)(glob_feat)
            else:
                z[head] = self.__getattr__(head)(y[-1])
        return [z]


def get_pose_net_glob(num_layers, heads, head_conv=256, down_ratio=4):
    model = DLASeg('dla{}'.format(num_layers), heads,
                   pretrained=True,
                   down_ratio=down_ratio,
                   final_kernel=1,
                   last_level=5,
                   head_conv=head_conv)
    return model
