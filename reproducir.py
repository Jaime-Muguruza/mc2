from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey, linear_reset
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE/'recalculado'
FIG = OUT/'images'
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261028)
# Orden de generación = orden de ejecución simulado. La partición se fija antes del ajuste.
w = rng.integers(20,101,48).astype(float)
v = rng.integers(5,46,48)/10
true_b = np.array([2.,.16,1.2])
X = np.column_stack([np.ones(48),w,v])
y = np.round(X@true_b+rng.normal(0,.55,48),3)
df = pd.DataFrame({'id':np.arange(1,49),'conjunto':['ajuste']*36+['prueba']*12,'W_millones':w,'V_GB':v,'T_s':y})
train=X[:36]; yt=y[:36]
model=sm.OLS(yt,train).fit()
pred=model.get_prediction(X).summary_frame(alpha=.05)
inf=model.get_influence()
e=model.resid; s=float(np.sqrt(model.mse_resid))
df['T_estimado_s']=model.predict(X)
df['residuo_s']=y-df.T_estimado_s
df['IC_media_inf']=pred.mean_ci_lower.to_numpy()
df['IC_media_sup']=pred.mean_ci_upper.to_numpy()
df['IP_inf']=pred.obs_ci_lower.to_numpy()
df['IP_sup']=pred.obs_ci_upper.to_numpy()
df['h']=np.r_[inf.hat_matrix_diag,[np.nan]*12]
df['r_interno']=np.r_[inf.resid_studentized_internal,[np.nan]*12]
df['Cook']=np.r_[inf.cooks_distance[0],[np.nan]*12]
df.to_csv(OUT/'datos_y_residuos.csv',index=False,float_format='%.10g')
def metrics(actual, predicted, baseline=None):
    err=actual-predicted
    return {'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(np.abs(err))), 'r2':float(1-np.sum(err**2)/np.sum((actual-actual.mean())**2))}
bp=het_breuschpagan(e,train,robust=True)
bg=acorr_breusch_godfrey(model,nlags=1)
reset=linear_reset(model,power=2,use_f=True)
jb=jarque_bera(e)
sw=stats.shapiro(e)
fold_rng=np.random.default_rng(20261029)
folds=np.array_split(fold_rng.permutation(36),6)
oof=np.empty(36); base_oof=np.empty(36); fold_results=[]
for j,valid in enumerate(folds):
    fit=np.setdiff1d(np.arange(36),valid)
    m=sm.OLS(yt[fit],train[fit]).fit()
    oof[valid]=m.predict(train[valid]); base_oof[valid]=yt[fit].mean()
    fold_results.append({'fold':j+1,'valid_ids':(valid+1).tolist(),**metrics(yt[valid],oof[valid])})
test=metrics(y[36:],df.T_estimado_s.to_numpy()[36:])
baseline=metrics(y[36:],np.full(12,yt.mean()))
test['coverage']=float(np.mean((y[36:]>=pred.obs_ci_lower.to_numpy()[36:])&(y[36:]<=pred.obs_ci_upper.to_numpy()[36:])))
example=model.get_prediction([[1,60,2.5]]).summary_frame().iloc[0].to_dict()
simrng=np.random.default_rng(20261030)
B=5000; inv=np.linalg.inv(train.T@train)
Ysim=train@true_b+simrng.normal(0,.55,(B,36))
bs=Ysim@train@inv
es=Ysim-bs@train.T
ss=np.sqrt(np.sum(es**2,axis=1)/33)
tcrit=float(stats.t.ppf(.975,33))
half=tcrit*ss[:,None]*np.sqrt(np.diag(inv))
cover=np.mean((bs-half<=true_b)&(true_b<=bs+half),axis=0)
sim={'B':B,'seed':20261030,'beta_true':true_b.tolist(),'mean_beta':bs.mean(axis=0).tolist(),'sd_beta':bs.std(axis=0,ddof=1).tolist(),'coverage':cover.tolist()}
corr=np.corrcoef(np.column_stack([w[:36],v[:36],yt]).T)
diag=[{'name':'Shapiro-Wilk','stat':float(sw.statistic),'p':float(sw.pvalue),'null':'Los errores son normales.'},
      {'name':'Jarque-Bera','stat':float(jb[0]),'p':float(jb[1]),'null':'Asimetría 0 y curtosis 3 (normalidad).'},
      {'name':'Koenker / Breusch-Pagan','stat':float(bp[0]),'p':float(bp[1]),'null':'Varianza condicional constante.'},
      {'name':'Breusch-Godfrey, orden 1','stat':float(bg[0]),'p':float(bg[1]),'null':'No hay autocorrelación de primer orden.'},
      {'name':'RESET, potencia 2','stat':float(reset.fvalue),'p':float(reset.pvalue),'null':'El término cuadrático del ajustado es innecesario.'}]
result={'n':36,'ntest':12,'k':3,'df_resid':33,'seed':20261028,'beta':model.params.tolist(),'se':model.bse.tolist(),'t':model.tvalues.tolist(),'p':model.pvalues.tolist(),'ci':model.conf_int().tolist(),'se_hc3':model.get_robustcov_results(cov_type='HC3').bse.tolist(),
        's':s,'mse':float(model.mse_resid),'sse':float(model.ssr),'ssreg':float(model.ess),'sst':float(model.centered_tss),'f':float(model.fvalue),'f_p':float(model.f_pvalue),'fcrit':float(stats.f.ppf(.95,2,33)),'r2':float(model.rsquared),'r2adj':float(model.rsquared_adj),'tcrit':tcrit,
        'xtx':(train.T@train).tolist(),'xty':(train.T@yt).tolist(),'inv':inv.tolist(),'means':np.mean(np.column_stack([w[:36],v[:36],yt]),axis=0).tolist(),
        'centered':{'Sww':float(np.sum((w[:36]-w[:36].mean())**2)),'Svv':float(np.sum((v[:36]-v[:36].mean())**2)),'Swv':float(np.sum((w[:36]-w[:36].mean())*(v[:36]-v[:36].mean()))),'Swy':float(np.sum((w[:36]-w[:36].mean())*(yt-yt.mean()))),'Svy':float(np.sum((v[:36]-v[:36].mean())*(yt-yt.mean())))},
        'diag':diag,'dw':float(durbin_watson(e)),'vif':[float(variance_inflation_factor(train,j)) for j in [1,2]],'corr':corr.tolist(),'rank':int(np.linalg.matrix_rank(train)),
        'train':metrics(yt,model.fittedvalues),'test':test,'baseline':baseline,'cv':metrics(yt,oof),'cv_baseline':metrics(yt,base_oof),'folds':fold_results,'example':example,'sim':sim,'sum_e':float(e.sum()),'max_h':float(inf.hat_matrix_diag.max()),'max_cook':float(inf.cooks_distance[0].max()),'max_r':float(np.abs(inf.resid_studentized_internal).max()),
        'cook_ids':(np.flatnonzero(inf.cooks_distance[0]>4/36)+1).tolist(),'h_ids':(np.flatnonzero(inf.hat_matrix_diag>2*3/36)+1).tolist(),'r_ids':(np.flatnonzero(np.abs(inf.resid_studentized_internal)>2)+1).tolist(),
        'limits':[[float(w[:36].min()),float(w[:36].max())],[float(v[:36].min()),float(v[:36].max())]],'hull':np.column_stack([w[:36],v[:36]])[ConvexHull(np.column_stack([w[:36],v[:36]])).vertices].tolist(),
        'rows':json.loads(df.to_json(orient='records')),'versions':{m:__import__(m).__version__ for m in ['numpy','scipy','statsmodels','pandas','matplotlib']}}
flag=inf.cooks_distance[0]>4/36
sens=sm.OLS(yt[~flag],train[~flag]).fit()
result['sensitivity']={'removed_ids':(np.flatnonzero(flag)+1).tolist(),'beta':sens.params.tolist(),'change_percent':(100*(sens.params/model.params-1)).tolist(),'test_rmse':metrics(y[36:],sens.predict(X[36:]))['rmse']}
result['p_hc3']=model.get_robustcov_results(cov_type='HC3').pvalues.tolist()
(OUT/'resultados.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#23384c','text.color':'#23384c','axes.titleweight':'bold','figure.facecolor':'white','axes.grid':True,'grid.alpha':.18,'savefig.bbox':'tight'})
teal='#087e8b'; orange='#e39842'
fig,axs=plt.subplots(2,3,figsize=(12,7.3),layout='constrained')
axs[0,0].scatter(model.fittedvalues,e,c=teal,s=28); axs[0,0].axhline(0,color=orange); axs[0,0].set(xlabel='Tiempo ajustado (s)',ylabel='Residuo (s)',title='Residuos frente a ajustados')
stats.probplot(e,dist='norm',plot=axs[0,1]); axs[0,1].set(title='Q-Q normal de residuos',xlabel='Cuantil normal teórico',ylabel='Residuo ordenado (s)')
axs[0,2].plot(np.arange(1,37),e,'o-',color=teal,ms=3,lw=1); axs[0,2].axhline(0,color=orange); axs[0,2].set(xlabel='Orden de ejecución simulado',ylabel='Residuo (s)',title='Residuos en orden de ejecución')
axs[1,0].scatter(inf.hat_matrix_diag,inf.resid_studentized_internal,c=teal,s=28); axs[1,0].axhline(2,color=orange,ls='--'); axs[1,0].axhline(-2,color=orange,ls='--'); axs[1,0].axvline(6/36,color=orange,ls=':'); axs[1,0].set(xlabel='Leverage h',ylabel='Residuo estandarizado interno',title='Leverage y residuos')
axs[1,1].scatter(w[:36],e,c=teal,s=28); axs[1,1].axhline(0,color=orange); axs[1,1].set(xlabel='Trabajo W (millones)',ylabel='Residuo (s)',title='Residuos frente a W')
axs[1,2].scatter(v[:36],e,c=teal,s=28); axs[1,2].axhline(0,color=orange); axs[1,2].set(xlabel='Volumen V (GB)',ylabel='Residuo (s)',title='Residuos frente a V')
fig.savefig(FIG/'diagnosticos.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(7.8,4.8),layout='constrained')
ax.scatter(yt,model.fittedvalues,label='Ajuste (36)',c=teal); ax.scatter(y[36:],df.T_estimado_s.to_numpy()[36:],label='Prueba (12)',c=orange,marker='s')
lims=[min(y.min(),df.T_estimado_s.min())-.5,max(y.max(),df.T_estimado_s.max())+.5]; ax.plot(lims,lims,'--',color='#6b7984'); ax.set(xlabel='Tiempo observado simulado (s)',ylabel='Tiempo estimado (s)',title='Observado frente a estimado'); ax.legend(); fig.savefig(FIG/'ajuste.png',dpi=180); plt.close(fig)
fig=plt.figure(figsize=(8.8,5.5)); ax=fig.add_subplot(111,projection='3d'); wg,vg=np.meshgrid(np.linspace(20,100,20),np.linspace(.5,4.5,20)); z=model.params[0]+model.params[1]*wg+model.params[2]*vg
ax.plot_surface(wg,vg,z,color=teal,alpha=.35); ax.scatter(w[:36],v[:36],yt,c=teal,s=18); ax.scatter(w[36:],v[36:],y[36:],c=orange,s=20,marker='s'); ax.set(xlabel='W (millones)',ylabel='V (GB)',zlabel='T (s)',title='Plano de regresión y datos simulados'); ax.view_init(23,-125); fig.savefig(FIG/'plano.png',dpi=180); plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
for j,ax in enumerate(axs):
    ax.hist(bs[:,j],bins=35,color=teal,alpha=.8); ax.axvline(true_b[j],color=orange,lw=2); ax.set(title=f'Coeficiente beta {j+1}',xlabel='Estimación',ylabel='Frecuencia')
fig.savefig(FIG/'simulacion.png',dpi=180); plt.close(fig)
print(json.dumps({key:result[key] for key in ['beta','r2','r2adj','s','f','f_p','diag','test','cv','cook_ids','h_ids','r_ids','example','sim']},ensure_ascii=False,indent=2))
