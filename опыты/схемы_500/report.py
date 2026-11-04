import json, sys, statistics as st
from collections import Counter
key=sys.argv[1]
R=[json.loads(x) for x in open(f'итог_{key}.jsonl',encoding='utf-8')]
R=[r for r in R if not r.get('пропущен')]
print(key,len(R),'падений',sum('падение' in r for r in R),'зависло',sum('зависло' in r for r in R))
T=('разбор','физика','диагностика','неисправности','плата')
for t in T:
    c=Counter('ok' if r[t]['ok'] else r[t]['kind'] for r in R if t in r); print(t,dict(c),'ср.с',round(sum(r[t]['s'] for r in R if t in r)/max(1,sum(t in r for r in R)),1))
E=Counter((t,r[t].get('err','')[:100],r[t].get('where','')[:90]) for r in R for t in T if t in r and not r[t]['ok'] and r[t].get('kind')=='сбой')
for k,v in E.most_common(12): print(v,k)
A=Counter((r[t].get('err','')[:70]) for r in R for t in T if t in r and not r[t]['ok'] and r[t].get('kind')=='отказ')
print('отказы',A.most_common(5))
ok=[r for r in R if 'деталей' in r]
for f in ('деталей','с обозначением','с номиналом','сомнительных','подпись под вопросом','неопознанных','цепей','находок физики','кандидатов','рисков','шагов плана','вариантов отказа'):
    v=[r.get(f,0) for r in ok]; print(f,'сумма',sum(v),'медиана',st.median(v))
print('без цепей/пустые',sum(r['деталей']<5 for r in ok))
print('типы',Counter(k for r in ok for k,v in r['типы'].items() for _ in range(v)).most_common(14))
print('падения',Counter(r.get('падение','')[-120:] for r in R if 'падение' in r).most_common(3))
print('зависшие',[r['project'] for r in R if 'зависло' in r][:12])
