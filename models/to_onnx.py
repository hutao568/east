import east_onnx
import torch
import argparse

def parse():
    parser = argparse.ArgumentParser('convert to onnx')
    parser.add_argument('pth', type=str, help='pytorch pth.tar path')
    parser.add_argument('onnx', type=str, help='onnx model path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    dummy_input = torch.randn(1, 3, 480, 672)

    state = torch.load(args.pth)
    model = east_onnx.East(pretrained_model=False)
    state_dict = dict()
    for k, v in state['model_state_dict'].items():
        state_dict[k.replace('module.','')] = v
    model.load_state_dict(state_dict)
    model.eval()
    torch.onnx.export(model, dummy_input, args.onnx, export_params=True)
