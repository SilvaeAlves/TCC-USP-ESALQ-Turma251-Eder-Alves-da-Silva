# Reproduz a Tabela 2 (busca em grade, GroupKFold 5 no treino, 1 particao de teste agrupada por volta).
# Uso: python src/03_ajuste_hiperparametros.py {base|lr|arvore|rf|gb|mlp} [n_estimators]
import pandas as pd, numpy as np, warnings, time, json, sys
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1]
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
RS=7895
d=pd.read_csv(RAIZ/'dados'/'dataset_2pilotos.csv')
d['alvo']=d.tempo_setor-d.groupby(['piloto','setor']).tempo_setor.transform('median')
CAT=['setor','tipo']; NUM=['pbrake_max','pbrake_media','t_freio','pbrake_taxa','accx_min',
     'accx_max','rpm_max','marcha_max','trocas','slip']
X=d[CAT+NUM]; y=d.alvo; g=d.volta
tr,te=next(GroupShuffleSplit(1,test_size=.25,random_state=RS).split(X,y,groups=g))
Xtr,Xte,ytr,yte,gtr=X.iloc[tr],X.iloc[te],y.iloc[tr],y.iloc[te],g.iloc[tr]
cv=GroupKFold(n_splits=5)
oh=lambda: OneHotEncoder(handle_unknown='ignore',sparse_output=False)
pa=lambda: ColumnTransformer([('c',oh(),CAT),('n','passthrough',NUM)])
pl=lambda: ColumnTransformer([('c',oh(),CAT),('n',StandardScaler(),NUM)])
config={
 'arvore':(Pipeline([('p',pa()),('m',DecisionTreeRegressor(random_state=RS))]),
   {'m__max_depth':[3,5,8,12,None],'m__min_samples_leaf':[1,5,10,20]}),
 'rf':(Pipeline([('p',pa()),('m',RandomForestRegressor(random_state=RS,n_jobs=1))]),
   {'m__n_estimators':[300,600],'m__max_features':[0.3,0.6,1.0],'m__min_samples_leaf':[1,3,8]}),
 'gb':(Pipeline([('p',pa()),('m',GradientBoostingRegressor(random_state=RS))]),
   {'m__n_estimators':[200,400,800],'m__learning_rate':[0.02,0.05,0.1],'m__max_depth':[2,3,4],'m__subsample':[0.8,1.0]}),
 'mlp':(Pipeline([('p',pl()),('m',MLPRegressor(max_iter=3000,early_stopping=True,random_state=RS))]),
   {'m__hidden_layer_sizes':[(32,),(64,32),(128,64)],'m__alpha':[1e-4,1e-3,1e-2],'m__learning_rate_init':[1e-3,1e-2]}),
}
def met(p): return dict(RMSE=float(np.sqrt(mean_squared_error(yte,p))),MAE=float(mean_absolute_error(yte,p)),R2=float(r2_score(yte,p)))
sub=sys.argv[2] if len(sys.argv)>2 else None
for nome in sys.argv[1:2]:
    t0=time.time()
    if nome=='base':
        out=dict(modelo=nome,CV=None,n=0,params={},**met(np.zeros(len(yte))))
    elif nome=='lr':
        lr=Pipeline([('p',pl()),('m',LinearRegression())])
        cvs=float(-cross_val_score(lr,Xtr,ytr,groups=gtr,cv=cv,scoring='neg_root_mean_squared_error').mean())
        out=dict(modelo=nome,CV=cvs,n=1,params={},**met(lr.fit(Xtr,ytr).predict(Xte)))
    else:
        pipe,grade=config[nome]
        if sub is not None:
            grade=dict(grade); grade['m__n_estimators']=[int(sub)]
        gs=GridSearchCV(pipe,grade,cv=cv,scoring='neg_root_mean_squared_error',n_jobs=-1).fit(Xtr,ytr,groups=gtr)
        out=dict(modelo=nome,CV=float(-gs.best_score_),n=int(np.prod([len(v) for v in grade.values()])),
                 params={k[3:]:(list(v) if isinstance(v,tuple) else v) for k,v in gs.best_params_.items()},
                 **met(gs.best_estimator_.predict(Xte)))
    out['seg']=round(time.time()-t0,1)
    json.dump(out,open(RAIZ/'resultados'/f'res_{nome}{"_"+sub if sub else ""}.json','w'))
    print(nome,'feito',out['seg'],'s',flush=True)
