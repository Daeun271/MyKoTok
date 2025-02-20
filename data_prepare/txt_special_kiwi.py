import sys
from tqdm import tqdm
import kiwipiepy


in_filename = sys.argv[1]
out_filename = sys.argv[2]

kiwi = kiwipiepy.Kiwi()

with open(in_filename, 'r') as f:
    with open(out_filename, 'w') as g:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            sents = kiwi.split_into_sents(line)
            
            g.write('[CLS] ')
            g.write(' [SEP] '.join(sent.text for sent in sents))
            g.write(' [SEP]\n')
