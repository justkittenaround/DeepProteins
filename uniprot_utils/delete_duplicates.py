from progressbar import ProgressBar
import os, csv
from argparse import ArgumentParser


def read_csv(fpath):
    items = []
    with open(fpath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            items.append(row)
    return items


def write_csv(fpath, items):
    with open(fpath, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for item in items:
            writer.writerow(item)


def find_duplicates(seqs):
    remove_indices = []
    pbar = ProgressBar()

    for i in pbar(range(len(seqs)-1, 0, -1)):
        seq1 = seqs[i][1:]

        for j in range(i-1, -1, -1):
            seq2 = seqs[j][1:]

            if len(seq1) != len(seq2):
                break

            if all([c1 == c2 for c1, c2 in zip(seq1, seq2)]):
                remove_indices.append(i)

    return remove_indices


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path',
        type=str,
        help='path to the file you want to remove duplicates in.')
    args = parser.parse_args()

    assert(os.path.isfile(args.path)), 'path does not exist.'

    seqs = read_csv(args.path)
    seqs.sort(key=len)
    remove_indices = list(set(find_duplicates(seqs)))

    print('[INFO] REMOVED {} SEQUENCES.'.format(len(set(remove_indices))))
    seqs = [seq for i_seq, seq in enumerate(seqs) if i_seq not in remove_indices]
    write_csv(args.path, seqs)
