# Reproduz a Tabela 4 do TCC: media de 10 particoes agrupadas por volta, hiperparametros da Tabela 2.
# Uso: python src/02_avalia_10_particoes.py
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1]
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
RS=7895
d=pd.read_csv(RAIZ/'dados'/'dataset_2pilotos.csv')
d['alvo']=d.tempo_setor-d.groupby(['piloto','setor']).tempo_setor.transform('median')
CAT=['setor','tipo']; NUM=['pbrake_max','pbrake_media','t_freio','pbrake_taxa','accx_min','accx_max','rpm_max','marcha_max','trocas','slip']
X=d[CAT+NUM]; y=d.alvo; g=d.volta
oh=lambda: OneHotEncoder(handle_unknown='ignore',sparse_output=False)
pa=lambda: ColumnTransformer([('c',oh(),CAT),('n','passthrough',NUM)]); pl=lambda: ColumnTransformer([('c',oh(),CAT),('n',StandardScaler(),NUM)])
mods={'Gradient Boosting':lambda: Pipeline([('p',pa()),('m',GradientBoostingRegressor(n_estimators=800,learning_rate=.05,max_depth=4,subsample=.8,random_state=RS))]),
 'Random Forest':lambda: Pipeline([('p',pa()),('m',RandomForestRegressor(n_estimators=600,max_features=1.0,min_samples_leaf=1,random_state=RS,n_jobs=-1))]),
 'Rede neural':lambda: Pipeline([('p',pl()),('m',MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-4,learning_rate_init=1e-2,max_iter=3000,early_stopping=True,random_state=RS))]),
 'Arvore de decisao':lambda: Pipeline([('p',pa()),('m',DecisionTreeRegressor(max_depth=8,min_samples_leaf=1,random_state=RS))]),
 'Regressao linear multipla':lambda: Pipeline([('p',pl()),('m',LinearRegression())])}
acc={k:[] for k in list(mods)+['Mediana por condutor-setor']}
for s in range(10):
    tr,te=next(GroupShuffleSplit(1,test_size=.25,random_state=1000+s).split(X,y,groups=g))
    yt=y.iloc[te]; b=np.sqrt(mean_squared_error(yt,0*yt))
    acc['Mediana por condutor-setor'].append((b,mean_absolute_error(yt,0*yt),r2_score(yt,0*yt),0.0))
    for k,f in mods.items():
        p=f().fit(X.iloc[tr],y.iloc[tr]).predict(X.iloc[te]); r=np.sqrt(mean_squared_error(yt,p))
        acc[k].append((r,mean_absolute_error(yt,p),r2_score(yt,p),(1-r/b)*100))
t=pd.DataFrame([dict(Modelo=k,RMSE=np.mean([x[0] for x in v]),RMSE_dp=np.std([x[0] for x in v]),MAE=np.mean([x[1] for x in v]),
     R2=np.mean([x[2] for x in v]),Ganho=np.mean([x[3] for x in v])) for k,v in acc.items()]).sort_values('RMSE')
print(t.round(3).to_string(index=False))
t.to_csv(RAIZ/'resultados'/'tabela4_reproduzida.csv',index=False)   # GB esperado: RMSE 1,018; R2 0,806; ganho 57,1%
