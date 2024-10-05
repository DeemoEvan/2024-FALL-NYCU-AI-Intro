import argparse
import glob
import os
import pickle
import sys

import numpy as np


def get_rtol(grad_np, grad_pt, atol=1e-8):
    rtols = np.maximum(np.abs(grad_np - grad_pt) - atol, 0) / (np.abs(grad_pt) + 1e-10)
    rtol = np.max(rtols)
    return rtol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grads_np', default='./grads_np/', type=str)
    parser.add_argument('--grads_pt', default='./grads_pt/', type=str)
    parser.add_argument('--verbose', action="store_true", default=False)
    args = parser.parse_args()

    rtol_list = []
    for pkl_path in glob.glob(os.path.join(args.grads_np, "*.pkl")):
        with open(pkl_path, "rb") as f:
            grads_np = pickle.load(f)

        pkl_path = pkl_path.replace(args.grads_np, args.grads_pt)
        with open(pkl_path, "rb") as f:
            grads_pt = pickle.load(f)

        for grad_np, grad_pt in zip(grads_np, grads_pt):
            if grad_np.shape != grad_pt.shape:
                raise ValueError(
                    f"shape mismatch: {grad_np.shape} != {grad_pt.shape}. "
                    f"Please check that \n"
                    f"\t1. Your return order of grads() exactly match the order of params(). \n"
                    f"\t2. The shape of your gradients is same as the shape of parameters. \n"
                    f"\t3. You don't transpose or reshape the parameters."
                )
            rtol = get_rtol(grad_np, grad_pt)
            rtol_list.append(rtol)
            if args.verbose:
                print(f"rtol: {rtol:.5E}", file=sys.stderr)

    rtol_list = np.array(rtol_list)
    print(np.exp(np.log(rtol_list + 1e-10).mean()))

if __name__ == '__main__':
    main()
