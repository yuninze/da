import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdd
import scipy.stats
import seaborn as sns

from itertools import product
from tqdm import tqdm
from scipy.signal import detrend

from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.model_selection import GridSearchCV,train_test_split


rnd=np.random.RandomState(0)
sns.set_theme(style="whitegrid",palette="bright",font="monospace")
pd.options.display.min_rows=6
pd.options.display.max_columns=6
pd.options.display.precision=5


intern={"ci":"CPIAUCSL",
        "pi":"PPIACO",
        "ii":"PCEPI",
        "hi":"CSUSHPISA",
        "cb":"BAMLH0A0HYM2",
        "ie":"T5YIE",
        "fs":"STLFSI4",
        "ic":"ICSA",
        "pr":"PAYEMS",
        "ys":"T10Y3M",
        "ng":"DHHNGSP",
        "cl":"DCOILWTICO",
        "fr":"DFF",
        "nk":"NIKKEI225",
        "fert":"PCU325311325311",}
extern={"zs":"ZS=F",
        "zc":"ZC=F",
        "zw":"ZW=F",
        "hg":"HG=F",
        "si":"SI=F",
        "ng":"NG=F",
        "cl":"CL=F",
        "uj":"JPY=X",
        "uy":"CNY=X",
        "ue":"EURUSD=X",
        "hs":"^HSI",
        "vn":"VNM",
        "yt":"^TYX",}


def apnd(path:str)->pd.DataFrame:
	if not path.endswith("/"):path+="/"
	return pd.concat(
        [pd.read_csv(f"{path}{q.name}") for q in
         os.scandir(path) if ".csv" in q.name],axis=0)


def index_full_range(f:pd.DataFrame):
    return pd.DataFrame(
        index=pd.date_range(f.index.min(),f.index.max(),freq="D"))


def getdata(days_visit=100):
    f=pd.read_csv("c:/code/f.csv",
        index_col="date",
        converters={"date":pd.to_datetime})

    renew_data_start=f.index.max()-pd.Timedelta(days=days_visit)

    renew_ids_fred=list(set(intern.keys())-set(extern.keys()))
    renew_data_fred:pd.DataFrame=(
        pdd.DataReader([intern[q] for q in renew_ids_fred],"fred",
        start=renew_data_start)
        .set_axis(renew_ids_fred,axis=1))

    renew_ids_yahoo=list(extern.values())
    renew_data_yahoo:pd.DataFrame=(
        pdd.DataReader(renew_ids_yahoo,"yahoo",
        start=renew_data_start)
        .loc[:,"Close"]
        .set_axis(list(extern.keys()),axis=1))

    renew_data=renew_data_fred.combine_first(renew_data_yahoo)
    f=(f.reindex(
        pd.date_range(f.index.min(),f.index.max(),freq="D",name="date"))
        .combine_first(renew_data))

    print(f[["ie","zs","si","cl","zc","uj"]].tail(5))
    ask=input("input y to save above::")
    if ask in ["y","Y"]:
        f.to_csv("c:/code/f.csv")
    return f


def mon(f:pd.DataFrame,start,stop)->np.ndarray:
    return np.asarray(
        sum([f.index.month==q for q in np.arange(start,stop,1)]),
            dtype="bool")


def dtr(arr,o=2):
    if any(np.isnan(arr)):
        raise ValueError(f"nan in the array")
    a=arr.copy()
    x=np.arange(len(a))
    q=np.polyval(np.polyfit(x,a,deg=o),x)
    a-=q
    return a


def arigeo(a)->tuple:
    a=a.dropna()
    mean_ari=np.mean(a)
    mean_geo=np.exp(np.mean(np.log(a)))
    median_ari=np.median(a)
    return (mean_ari,mean_geo,median_ari)


def deflator(inflator):
    '''results a deflator series from index series'''
    deflator=inflator.interpolate("pchip",limit=7).ffill()
    deflator[pd.isna(deflator)]=0
    return 1-(deflator*.01)


def act(t,i,adf=False):
    '''results an actual adjusted and its occurance'''
    a      =pd.concat([t,i],axis=1).dropna()
    a      =a.iloc[:,0]*a.iloc[:,1]
    a_l    =scipy.stats.yeojohnson(a)[0]
    a_ls   =scipy.stats.zscore(a_l)
    a_lsp  =scipy.stats.norm.cdf(a_ls,
      loc  =a_ls.mean(),
      scale=a_ls.std(ddof=1))
    a_lsp_f=pd.Series(a_lsp,index=a.index)
    return (pd.concat([a,a_lsp_f],axis=1)
        .set_axis([f"{t.name}",f"{t.name}lp"],axis=1))


def rng(f:pd.DataFrame,i:str,
        rng=(.05,3),test=False)->pd.DataFrame:
    f=f.copy()
    if not i in f.columns:
        raise NameError(f"{i} does not exist")
    col=f"{i}lzp"
    if test:
        rng=np.delete(
            np.round(np.flip(
            np.geomspace(rng[0],1,rng[1])),2),2)
    else:
        rng=np.round(np.flip(
            np.percentile(f[col].dropna(),(2,15,30,100))),2)
    f.loc[:,f"{col}rng"]=None
    for q in range(len(rng)):
        f.loc[:,f"{col}rng"]=np.where(
        (~pd.isna(f[col])) & (f[col]<=rng[q]),
        rng[q],
        f[f"{col}rng"])
    f_=f.loc[:,f"{col}rng"].astype("category").copy()
    f.update(f_)
    return f


def nav(f:pd.DataFrame,i:str,v:float):
    rowidx=np.abs(f[f"{i}"]-v).argmin()
    colidx=f.columns.get_indexer([f"{i}"])[0]
    viewport=f.iloc[rowidx,colidx:colidx+2]
    cdf_i=viewport[1]
    print(f"{cdf_i*100:.5f}%")
    return viewport


def ns(f:pd.DataFrame,x:str,y:str):
    f=f[[x,y]].dropna()
    rowidx=np.amin((f.count()[x],f.count()[y]))
    return f[f.shape[0]-rowidx:]


def cx(f:pd.DataFrame,x:str,y:str,
        d=180,normed=True,save=True,detrend=lambda q:q,test=False):
    f=ns(f,y,x)
    if save:
        plt.figure(figsize=(22,14))
        xc=plt.xcorr(f[y],f[x],
            detrend=detrend,maxlags=d,normed=normed)
        plt.suptitle(f"{y},{x},{d}")
        plt.savefig(f"e:/capt/{y}_{x}_{d}.png")
        plt.cla()
        plt.clf()
        plt.close()
        idx=abs(xc[1]).argmax()
        return xc[0][idx],xc[1][idx]
    fg,ax=plt.subplots(1,2)
    ac=ax[0].acorr(f[x],
        detrend=detrend,maxlags=d)
    xc=ax[1].xcorr(f[y],f[x],
        detrend=detrend,maxlags=d,normed=normed)
    fg.suptitle(f"{y},{x},{d}")
    ac_=ac[0][abs(ac[1]).argmax()]
    xc_=xc[0][abs(xc[1]).argmax()]
    if test:
        return (ac[0],ac[1]),(xc[0],xc[1])
    return ac_,xc_


def cx_(f:pd.DataFrame,x,d=12):
    f=f[x]
    cache=[]
    for col in tqdm(list(product(x,x)),desc=f"cx-rel"):
        if not col[0] is col[1]:
            rslt=cx(f[[col[0],col[1]]],
                col[0],col[1],d=d,save=True)
            cache.append((col[0],col[1],rslt[0],np.round(rslt[1],2)))
    rslt=(pd.DataFrame(cache,columns=["x","y","dur","coef"])
            .sort_values(by="coef",ascending=False)
            .set_index(keys=["x","y"])
            .sort_index())
    return rslt


def impt(f:pd.DataFrame,x,n=10)->pd.DataFrame:
    return (pd.DataFrame(
        KNNImputer(n_neighbors=n,weights="distance").fit_transform(f),
            index=f.index,columns=x))


def regr(x,y,s=0.2,cv=5):
    xi,xt,yi,yt=train_test_split(x,y,test_size=s)
    params={"learning_rate":
                [.1,.01,.001,.0001],
            "max_depth":
                [a for a in np.arange(1,8,2)],
            "n_estimators":
                [a for a in np.arange(48,97,16)],
            "max_features":
                ["sqrt"],
            "min_samples_leaf":
                [a for a in np.arange(10,41,10)],
            "random_state":[rnd]}
    regressor=gbr()
    cursor=GridSearchCV(regressor,params,cv=cv,n_jobs=-1,verbose=3)
    cursor.fit(xi,yi)
    print(f"{yi.name}::{cursor.score(xt,yt)}")
    return cursor


def proc(f:pd.DataFrame,
        x=["cb","yt","ys","ng","cl","zc","zw","uj","ue","ic"],
        y=["pi","ci","ii"],
        thresh=6):
    proc_f=f[x].copy()
    proc_f.update(proc_f.ic.dropna().pct_change())
    proc_f=proc_f.interpolate("index",limit=2).dropna(thresh=thresh)
    proc_f.update(proc_f.ic.ffill())
    proc_f=impt(proc_f,x=proc_f.columns)
    return (pd.DataFrame(
            [scipy.stats.yeojohnson(proc_f[q])[0] for q in proc_f.columns],
            index=proc_f.columns,columns=proc_f.index)
            .T.join(f[y].shift(-60).bfill()))


def regr_(f:pd.DataFrame,t,
        x=["cb","yt","ys","ng","cl","zc","zw","uj","ue","ic"],
        y=["pi","ci","ii"],cv=5,test=1):
    f=proc(f,x=x,y=y,thresh=int(len(x)*.8))
    x_=x.copy()
    x_.extend([y for y in y if y!=t])
    x0=f.dropna(subset=y)[x_]
    y0=f.dropna(subset=y)[t]
    x1=f.loc["2022-07":].copy()[x_].ffill()
    if test:
        print(f"{os.linesep}==x0==")
        print(x0)
        print(f"{os.linesep}==y0::{t}==")
        print(y0)
        print(f"{os.linesep}==x1==")
        print(x1)
        toi=input(f"{os.linesep}go?")
        if not toi:
            return None
    i=regr(x0,y0,cv=cv)
    return {f"{t}":[i,x1.assign(i_=i.best_estimator_.predict(x1))],
            f"{t}_stat":i.cv_results_}


# innermost visualisations
def hm(f,
    mm=(-1,1),ax=None,cbar=False,title=None):
    if title is None:
        title=f"{', '.join(f.columns)}"
    mask=np.triu(np.ones_like(f,dtype=bool))
    cmap=sns.diverging_palette(240,10,as_cmap=True)
    if ax is None:
        plt.subplots(figsize=(22,20))
    plt.title(title)
    sns.heatmap(f,
        mask=mask,cmap=cmap,ax=ax,cbar=cbar,
        vmin=mm[0],vmax=mm[1],
        annot=True,center=0,square=True,linewidths=.5,fmt=".2f")


def hm_(f):
    if len(f.columns)>20:
        q=input(f"{len(f.columns)} columns:: ")
        if not q:raise ValueError(f"{len(f.columns)} is too much")
    _,ax=plt.subplots(1,3,figsize=(22,16))
    hm(f,
        title=f"",ax=ax[0])
    ax[0].title.set_text("org")
    hm(f.dropna(),
        title=f"",ax=ax[1])
    ax[1].title.set_text("dropna")
    hm(impt(f,f.columns),
        title=f"",ax=ax[2])
    ax[2].title.set_text("impt")


def pp(f,
    vars=None,l=False,hue=None):
    (sns.pairplot(data=f,vars=vars,hue=hue,
        dropna=False,kind="scatter",diag_kind="hist")
        .map_diag(sns.histplot,log_scale=l,
        multiple="stack",element="step"))


def vs(f,x:str,y:str):
    fg,ax=plt.subplots()
    ax0=ax.twinx()
    ax.plot(f.loc[:,x],label=x,color="red")
    ax0.plot(f.loc[:,y],label=y,color="blue")
    ax.set_ylabel(x)
    ax0.set_ylabel(y)
    h,l=ax.get_legend_handles_labels()
    fg.legend(h,(),loc="upper center")


def rp(f,x,y,
    hue=None,size=None,sizes=(20,200)):
    sns.relplot(data=f,x=x,y=y,hue=hue,size=size,sizes=sizes)
