import csv, os
from argparse import ArgumentParser
from delete_duplicates import find_duplicates, read_csv



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('fpath1',
        type=str)
    parser.add_argument('fpath2',
        type=str)
    args = parser.parse_args()

    seqs1 = read_csv(args.fpath1)
    seqs2 = read_csv(args.fpath2)

    seqs = seqs1 + seqs2
    seqs.sort(key=len)

    remove_indices = find_duplicates(seqs)
    print(len(remove_indices))
