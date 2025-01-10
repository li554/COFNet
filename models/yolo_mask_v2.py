# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

# from mmcv.ops import DeformConv2dPack as DCN

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for i, x in enumerate(ch) if i < self.nl)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        logits_ = []
        self.training |= self.export
        nx = len(x)  # number of input
        for i in range(nx):
            # if nx > self.nl and nx == 4 and i == 3:
            #     j = 2
            # else:
            #     j = i % 3
            j = i
            x[i] = self.m[j](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[j].shape[2:4] != x[i].shape[2:4]:
                    self.grid[j] = self._make_grid(nx, ny).to(x[i].device)

                logits = x[i][..., 5:]

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[j]) * self.stride[j]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[j]  # wh
                z.append(y.view(bs, -1, self.no))
                logits_.append(logits.view(bs, -1, self.no - 5))

        return x if self.training else (z, torch.cat(logits_, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, img_size=640,
                 pool_mode='avg_pool'):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # print(self.model)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # print("1, ch, s, s", 1, ch, s, s)
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward
            if len(self.yaml['anchors']) == 4:
                m.stride = torch.Tensor([8.0, 16.0, 32.0, 32.0])
            else:
                m.stride = torch.Tensor([8.0, 16.0, 32.0])
            # print("m.stride", m.stride)
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            # self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())
        self.pool_mode = pool_mode
        self.spatial_dim = img_size // 32
        self.embed_dim = int(1024 * self.yaml.get('width_multiple', 0.5))
        if pool_mode == 'avg_pool':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_mode == 'attn_pool':
            self.attn_pool1 = AttentionPool2d(self.spatial_dim, self.embed_dim, 8)
            self.attn_pool2 = AttentionPool2d(self.spatial_dim, self.embed_dim, 8)
            self.attn_pool3 = AttentionPool2d(self.spatial_dim, self.embed_dim, 8)
            self.attn_pool4 = AttentionPool2d(self.spatial_dim, self.embed_dim, 8)
        initialize_weights(self)
        logger.info('')

    def forward(self, x, x2=None, x3=None, p="predicted", augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi, x2, x3, p)[0]  # forward
                yi[..., :4] /= si
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, x2, x3, p, profile)  # single-scale inference, train

    def forward_once(self, x, x2=None, x3=None, p="predicted", profile=False):
        # x: 可见光 x2: 红外 x3: gt mask x4: noise mask
        y, dt = [], []  # outputs
        i = 0
        layer_maskf = []
        layer_maskf_conv = []
        vi_pool = None
        mask_pool = None
        fu_pool = None
        ir_pool = None
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                if m.f != -4 and m.f != -8:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
            if m.f == -4:
                if x2 is not None:
                    x = m(x2)
            elif m.f == -8:
                x = m(x3)
            elif isinstance(m.f, list) and len(m.f) == 3:
                vi_f = x[0]
                ir_f = x[1]
                mask_f = x[2].detach()
                x = m(x, p, x3)
                if self.training:
                    if isinstance(x, tuple):
                        x, mask_f_conv = x
                        layer_maskf.append(mask_f)
                        layer_maskf_conv.append(mask_f_conv)
                    if vi_f.shape[1] == self.embed_dim:
                        if self.pool_mode == 'attn_pool':
                            vi_pool = self.attn_pool1(vi_f).view(len(vi_f), -1)
                            ir_pool = self.attn_pool2(ir_f).view(len(ir_f), -1)
                            fu_pool = self.attn_pool3(x).view(len(x), -1)
                            mask_pool = self.attn_pool4(mask_f).view(len(mask_f), -1)
                        else:
                            vi_pool = self.avg_pool(vi_f).view(len(vi_f), -1)
                            ir_pool = self.avg_pool(ir_f).view(len(ir_f), -1)
                            fu_pool = self.avg_pool(x).view(len(x), -1)
                            mask_pool = self.avg_pool(mask_f).view(len(mask_f), -1)
            else:
                x = m(x)
            y.append(x if m.i in self.save else None)  # save output
            i += 1

        if profile:
            logger.info('%.1fms total' % sum(dt))
        if self.training:
            return x, vi_pool, ir_pool, fu_pool, mask_pool, layer_maskf, layer_maskf_conv
        else:
            return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batch norm
                m.forward = m.fuseforward  # update forward
        # self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP,
                 C3, C3TR]:

            if m is Focus:
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is Conv and args[0] == 64:  # new
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR]:
                    args.insert(2, n)  # number of repeats
                    n = 1

        elif m is ResNetlayer:
            if args[3] == True:
                c2 = args[1]
            else:
                c2 = args[1] * 4
        elif m is VGGblock:
            c2 = args[2]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Add, DMAF]:
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            c2 = ch[f[0]]
            args = [c2, args[1]]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is NiNfusion:
            c1 = sum([ch[x] for x in f])
            c2 = c1 // 2
            args = [c1, c2, *args]
        elif m is TransformerFusionBlock:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        elif m is SEFusionBlock:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        elif m is MaskGuideFusionBlock:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='transformer/yolov5l_MaskFusion_mask_FLIR.yaml',
                        help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)
    print(device)

    model = Model(opt.cfg, ch=3, img_size=640, pool_mode="avg_pool").to(device)
    input_rgb = torch.randn(2, 3, 640, 640).to(device)
    input_ir = torch.randn(2, 3, 640, 640).to(device)
    mask = torch.ones(2, 3, 640, 640).to(device) * 2

    output = model(input_rgb, input_ir, mask, p="sam")

    import thop

    with torch.no_grad():
        gflops, params = thop.profile(model, inputs=(input_rgb, input_ir, mask))
        print('gflops:', gflops / 1E9 / 2, 'params:', params / 1E6)