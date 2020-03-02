import csv
from urllib.request import Request, urlopen
from progressbar import ProgressBar
import os
from argparse import ArgumentParser


def letter2num(seqs):
    seqList = []

    for seq in seqs:
        seqNums = []
        for letter in seq:
            seqNums.append(max(ord(letter)-97, 0))
        seqList.append(seqNums)

    return seqList


def get_uniprot_data(kw, num_prots=None, sort=False):
    '''Goes to the uniprot website and searches for
       data with the keyword given. Returns the data
       found up to limit elements.'''

    url1 = 'http://www.uniprot.org/uniprot/?query='  # first part of URL

    # make 2nd part of URL depending if number of proteins is specified or not
    if num_prots is None:
        url2 = '&columns=sequence&format=tab'
    else:
        url2 = '&columns=sequence&format=tab&limit='+str(num_prots)

    if sort:
        url2 = url2 + '&sort=length&desc=no'

    # Query uniprot with keyword and fetch data
    query_complete = url1 + kw + url2
    request = Request(query_complete)
    response = urlopen(request)
    data = response.read()
    data = str(data, 'utf-8')
    data = data.split('\n')
    data = data[1:-1]
    data = list(map(lambda x:x.lower(), data))

    return data


def read_txtfile(fname):
    items = []
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            items.append(row[0])
    return items



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('save_dir',
        type=str,
        help='Directory where to save the protein sequence files.')
    parser.add_argument('keyword_file',
        type=str,
        help='Name of the file with the keywords you want to pull.')
    parser.add_argument('--batch_size',
        type=int,
        default=256,
        help='How many sequences you want per batch.')
    parser.add_argument('--max_num_prots',
        type=int,
        default=70000,
        help='The max number of proteins to pull per keyword.')
    args = parser.parse_args()


    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    else:
        assert(len(os.listdir(args.save_dir)) != 0), \
            'save_dir exists and is not empty.'

    keywords = read_txtfile(args.keyword_file)

    if 'keywords-used.txt' in os.listdir():
        keywords_used = read_txtfile('keywords-used.txt')
        keywords = [kw for kw in keywords if kw not in keywords_used]


    f_used = open('keywords-used.txt', 'a')
    writer_used = csv.writer(f_used, delimiter=',')

    pbar = ProgressBar()
    count = 0
    file_num = 0

    for keyword in pbar(keywords):
        string = ''
        for c in keyword:
            string += c if c != ' ' else '%20'

        protein_seqs = get_uniprot_data(string, num_prots=args.max_num_prots)
        protein_seqs = letter2num(protein_seqs)

        for protein_seq in protein_seqs:
            if count % args.batch_size == 0 and count != 0:
                f_seq.close()
                count = 0
                file_num += 1
                f_seq = open(os.path.join(args.save_dir, 'uniprot_data_batch_{}.txt'.format(file_num)), 'w')
                writer = csv.writer(f_seq, delimiter=',')
            elif count == 0 and file_num == 0:
                f_seq = open(os.path.join(args.save_dir, 'uniprot_data_batch_0.txt'), 'w')
                writer = csv.writer(f_seq, delimiter=',')

            protein_seq.insert(0, keyword)
            writer.writerow(protein_seq)
            count += 1

        writer_used.writerow([keyword])

    f_seq.close()
    f_used.close()
