import mlconfig
from torch import optim

mlconfig.register(optim.lr_scheduler.MultiStepLR)
mlconfig.register(optim.lr_scheduler.StepLR)
mlconfig.register(optim.lr_scheduler.ExponentialLR)
mlconfig.register(optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(optim.lr_scheduler.ReduceLROnPlateau)



from tasks.base import *
from tasks.Watermark import *
from tasks.ChannelAE import *
from tasks.CoverOrStega import *