

import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    line = line.strip()
    line = line.split(",")

    if len(line) >=8:
        airport = line[3]
        arrdelay = line[8]
        depdelay=[6]

        print ('%s\t%s' % (airport, arrdelay,depdelay))