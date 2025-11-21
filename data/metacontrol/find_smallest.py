import sys
import numpy

if __name__=='__main__':
    if(len(sys.argv) < 2):
        print("Usage: exec diff_file")
        sys.exit(1)
    with open(sys.argv[1], "r") as f:
        recs = f.readlines()
        vals = []
        for rec in recs:
            tokens = rec.strip().split()
            vals.append(float(tokens[-1]))
        idx = numpy.argmin(vals)
        sys.stdout.write(recs[idx])
