import argparse
# import tensorwatch as tw

from mmcv import Config
# from mmcv.cnn import get_model_complexity_info
from torchstat_utils import model_stats
import torch
import tensorrt as trt
from torch2trt import torch2trt
import sys

sys.path.append('.')
from models import build_posenet


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--shape',
                        type=int,
                        nargs='+',
                        default=[2048, 1024],
                        help='input image size')
    parser.add_argument('--out',
                        type=str,
                        default="hrnet_lite.plan",
                        help='Output file name')

    args = parser.parse_args()
    return args


def save_engine(model, name='test.engine'):
    with open(name, 'wb') as f:
        f.write(model.engine.serialize())


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_posenet(cfg.model)
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    x = torch.rand(input_shape,
                   dtype=next(model.parameters()).dtype,
                   device=next(model.parameters()).device).cuda()

    with torch.no_grad():
        model_trt = torch2trt(
            model.cuda(),
            [x],
            log_level=trt.Logger.INFO,  #max_workspace_size= (2<<30),
            input_names=["input"],
            output_names=["output"])

        print(f'saving model in {args.out}')
        save_engine(model_trt, args.out)

        y = model(x)
        y_trt = model_trt(x)
        for y0, y1 in zip(y, y_trt):
            print(y0.shape, y1.shape, torch.max(torch.abs(y0 - y1)))


if __name__ == '__main__':
    main()
