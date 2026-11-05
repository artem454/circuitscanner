import sys
from pathlib import Path
sys.path.insert(0,'code')
import cv2, core
OUT=Path('C:/Users/Artem/Downloads/схемы_на_проверку')
n,x0,y0,x1,y1,k=[int(a) for a in sys.argv[1:7]]
d=next(p for p in sorted(OUT.iterdir()) if p.name.startswith(f'{n:02d}_'))
g=core.imread(d/'source.png')[y0:y1,x0:x1]
core.imwrite(f'zoom.png',cv2.resize(g,None,fx=k,fy=k,interpolation=cv2.INTER_CUBIC))
