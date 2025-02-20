import sys
import os
import shutil


for fname in sys.argv[1:]:
    shutil.move(fname, fname + '.bak')

    removed = 0

    with open(fname + '.bak', 'r') as f:
        with open(fname, 'w') as g:
            for line in f:
                line = line.strip()
                if not line:
                    removed += 1
                    continue
                g.write(line + '\n')

    print(fname, removed)

    os.remove(fname + '.bak')