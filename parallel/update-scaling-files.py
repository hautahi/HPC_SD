"""Update all the simulation files in one turn."""

import argparse
import os

from string import Template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action='store_true', help='stochastic')
    parser.add_argument('-p', metavar='output_path', type=str, required=True)
    parser.add_argument('-n', metavar='cores', type=int, required=True)
    parser.add_argument('-t', metavar='tmpl_dir', type=str, required=True)
    args = parser.parse_args()
    print(args)

    source_files = filter(
        lambda f: f.endswith('.py.tmpl'),
        os.listdir(args.t))

    for src in source_files:
        tmpl_path = os.path.join(args.t, src)
        exec_path = os.path.join(os.getcwd(), src.rstrip('.tmpl'))
        with open(tmpl_path, 'r') as fh:
            tmpl = Template(fh.read())
        with open(exec_path, 'w') as fh:
            fh.write(tmpl.safe_substitute(
                PATH=args.p,
                FILETYPE=('stochastic' if args.s else 'deterministic'),
                NCORES=args.n,
                STOCHASTIC=int(args.s)
            ))
