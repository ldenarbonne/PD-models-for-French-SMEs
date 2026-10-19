import pandas as pd, glob, json, re, sys
out=[]
for f in sorted(glob.glob(sys.argv[1]+'/y*.csv')):
    d=pd.read_csv(f,sep=';',dtype=str,usecols=['registre','dateparution','jugement'])
    d=d.dropna(subset=['registre','jugement'])
    d['siren']=d.registre.str.split(',').str[-1].str.replace(' ','')
    def p(j):
        try: x=json.loads(j); return x.get('famille',''), x.get('nature','')
        except: return '',''
    fn=d.jugement.map(p)
    d['famille']=fn.str[0]; d['nature']=fn.str[1]
    out.append(d[['siren','dateparution','famille','nature']])
b=pd.concat(out)
b=b[b.famille.str.contains("ouverture",case=False)]
b.to_csv(sys.argv[2],index=False)
print(b.shape); print(b.nature.value_counts().head(15))
