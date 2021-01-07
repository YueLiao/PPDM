from .pose_dla_dcn import get_pose_net as dla
from .large_hourglass import get_large_hourglass_net as hourglass
from .pose_dla_dcn_glob import get_pose_net_glob as dlaglob
from .pose_dla_dcn_3level import get_pose_net_3level as dla3level
from .pose_dla_dcn_glob_3level import get_pose_net_glob_3level as dla3levelglob
from .resnet_dcn import get_pose_net as resdcn

__all__ = ['dla', 'hourglass', 'dlaglob', 'dla3level', 'dla3levelglob', 'resdcn']