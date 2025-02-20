import sys

target_file = sys.argv[1]

concat_files = sys.argv[2:]


with open(target_file, 'w') as f:
    for fname in concat_files:
        print(fname)
        with open(fname, 'r') as g:
            for line in g:
                f.write(line)
