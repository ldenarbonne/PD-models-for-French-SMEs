import sys, csv
sirens=set(open(sys.argv[1]).read().split())
csv.field_size_limit(10**8)
r=csv.reader(sys.stdin, delimiter=';'); next(r, None)
w=csv.writer(sys.stdout)
for row in r:
    if row and row[0].zfill(9) in sirens:
        w.writerow([row[0].zfill(9)]+row[1:])
