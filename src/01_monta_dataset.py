# Monta dataset_2pilotos.csv (P1 e P2, 112 voltas, 1.120 observacoes) a partir dos CSV exportados pelo DuckDB.
# Entradas (nao publicadas, ver dados/README.md): telemetria.csv (P1), p2_Q.csv, p2_R.csv, p2_TL.csv (P2).
# Uso: python src/01_monta_dataset.py [pasta_dos_csv_brutos]   (padrao: dados/brutos/)
import pandas as pd, numpy as np, re, sys
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1]
L=4309.0; N=10; U=str(Path(sys.argv[1]) if len(sys.argv)>1 else RAIZ/'dados'/'brutos')+'/'
def extrai(s, piloto, nome, tipo, pref):
    out=[]
    if s.Distance.max()<4000 or s.Speed.max()<40: return out
    dist=s.Distance.values; tt=s.time_ns.values/1e9
    vi=(dist//L).astype(int)
    for v in np.unique(vi):
        m=vi==v; g=s[m]; dg=dist[m]
        if dg.max()-dg.min()<L*0.985 or g.Speed.min()*3.6<25: continue
        dur=tt[m].max()-tt[m].min()
        if not (95<dur<145): continue
        setor=np.minimum(((dg-dg.min())/L*N).astype(int)+1,N)
        for si in range(1,N+1):
            q=g[setor==si]
            if len(q)<20: continue
            tq=q.time_ns.values/1e9; pb=q.dash_pbrake_f.values/1e5; sp=q.Speed.values
            acx=np.gradient(sp,tq)/9.81
            out.append(dict(piloto=piloto,sessao=nome,tipo=tipo,volta=f'{pref}{nome}_L{v:02d}',setor=f'T{si}',
              tempo_setor=round(float(tq.max()-tq.min()),4),pbrake_max=round(float(pb.max()),2),pbrake_media=round(float(pb.mean()),2),
              t_freio=round(float((pb>5).sum()/100.0),3),pbrake_taxa=round(float(np.abs(np.gradient(pb,tq)).max()),1),
              accx_min=round(float(np.percentile(acx,1)),3),accx_max=round(float(np.percentile(acx,99)),3),
              rpm_max=round(float(q.RPM.max()*9.549),0),marcha_max=float(q.ecu_gear_pt.max()),
              trocas=int((np.diff(q.ecu_gear_pt.values)!=0).sum()),slip=round(float(np.abs(q.dash_speed_rl-q.dash_speed_fl).max()*3.6),3)))
    return out
def blocos(d):
    c=np.flatnonzero(np.diff(d.time_ns.values)<=0)+1
    return list(zip(np.concatenate(([0],c)),np.concatenate((c,[len(d)]))))
# ordem alfabetica dos .pds exportados (cada bloco de time_ns = um arquivo)
nomes=['Q1.1-27.02-18h15-01','Q1.1-27.02-18h15','Q1.1-28.02-09h00','Q1.1-28.02-10h03','R1.1-28.02-13h13','R1.1-28.02-14h09','R1.1-28.02-14h25','R1.2-28.02-14h41',
 'R2.1-01.03-11h32','R2.1-01.03-11h59','R2.1-01.03-13h23','R2.1-01.03-13h43','R2.1-01.03-13h54','TL1.1-27.02-15h25','TL1.2-27.02-15h41','TL1.3-27.02-15h58',
 'TO1.1-27.02-09h28','TO1.1-27.02-09h30','TO1.1-27.02-10h19','TO1.2-27.02-10h50','TO1.3-27.02-11h05','TRP1.1-26.02-10h51','TRP1.2-26.02-11h07','TRP1.3-26.02-11h21',
 'TRP2.1-26.02-14h18','TRP2.2-26.02-14h37','TRP2.3-26.02-14h53','TRP3.1-26.02-16h36','TRP3.2-26.02-16h51','s1']
rows=[]
d=pd.read_csv(U+'telemetria.csv')
for k,(a,b) in enumerate(blocos(d)):
    nm=nomes[k]
    if nm.endswith('-01') or nm=='s1': continue   # duplicata e arquivo de teste
    tp=re.match(r'(TRP|TO|TL|Q1|R1|R2)',nm).group(1)
    rows+=extrai(d.iloc[a:b].reset_index(drop=True),'P1',nm,tp,'')
mapa={'p2_Q':[('Q1','Q1.1-10h22'),('Q1','Q1.2-10h39'),('Q1','Q1.2-10h42')],'p2_R':[('R1','R1.1-14h26'),('R2','R2.1-13h54')],
      'p2_TL':[('TL','TL1.1-15h25'),('TL','TL1.2-15h29'),('TL','TL1.3-15h42')]}
for arq,sess in mapa.items():
    d=pd.read_csv(U+arq+'.csv')
    for k,(a,b) in enumerate(blocos(d)):
        if k>=len(sess): break
        tp,nm=sess[k]; rows+=extrai(d.iloc[a:b].reset_index(drop=True),'P2',nm,tp,'P2_')
ds=pd.DataFrame(rows); ds.to_csv(RAIZ/'dados'/'dataset_2pilotos.csv',index=False)
print(ds.shape,'| voltas:',ds.volta.nunique())   # esperado (1120, 16) e 112
