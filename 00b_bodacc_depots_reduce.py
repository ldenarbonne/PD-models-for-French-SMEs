import sys, csv, json
sirens=set(open(sys.argv[1]).read().split())
csv.field_size_limit(10**8)
r=csv.reader(sys.stdin, delimiter=';'); next(r)
w=csv.writer(sys.stdout)
for row in r:
    if len(row)<3 or not row[0]: continue
    s=row[0].split(',')[-1].replace(' ','')
    if s not in sirens: continue
    try: dc=json.loads(row[2]).get('dateCloture','') if row[2] else ''
    except Exception: dc=''
    w.writerow([s,row[1],dc])
