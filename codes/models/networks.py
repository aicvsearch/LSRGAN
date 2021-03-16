import torch
import torch.nn as nn
import models.modules.architecture as arch

####################
# define network
####################

# Generator
def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    if which_model == 'RRDB_net':  # RRDB
        netG = arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

