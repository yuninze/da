import os
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product
from tqdm import tqdm
from glob import glob
from time import time
from full_fred.fred import Fred

from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split,GridSearchCV

from selenium import webdriver
from selenium.webdriver.common.by import By


BD=251
PATH="c:/code/"
P="bright"
intern={
"cb":"BAMLH0A0HYM2",
"fs":"STLFSI3",
"ie":"T5YIE",
"ic":"ICSA",
"pr":"PAYEMS",
"ce":"PCETRIM12M159SFRBDAL",
"yt":"DGS30",
"ys":"T10Y2Y",
"ng":"DHHNGSP",
"cl":"DCOILWTICO",
"fr":"DFF",
"nk":"NIKKEI225",
"ci":"CPIAUCSL",
"pi":"PPIACO",
"ii":"PCEPILFE",
"hi":"CSUSHPISA",
"ue":"DEXUSEU",
"uy":"DEXCHUS",
"ua":"DEXUSNZ",
"uj":"DEXJPUS",
"fert":"PCU325311325311",
}
extern={
"zs":"https://www.investing.com/commodities/us-soybeans-historical-data",
"zc":"https://www.investing.com/commodities/us-corn-historical-data",
"zw":"https://www.investing.com/commodities/us-wheat-historical-data",
"hg":"https://www.investing.com/commodities/copper-historical-data",
"si":"https://www.investing.com/commodities/silver-historical-data",
"ng":"https://www.investing.com/commodities/natural-gas-historical-data",
"cl":"https://www.investing.com/commodities/crude-oil-historical-data",
"hs":"https://www.investing.com/indices/hang-sen-40-historical-data",
"uj":"https://www.investing.com/currencies/usd-jpy-historical-data",
"ue":"https://www.investing.com/currencies/eur-usd-historical-data"
}
rnd=np.random.RandomState(0)
sns.set_theme(style="whitegrid",palette=P,font="monospace")
pd.options.display.min_rows=6
pd.options.display.float_format=lambda q:f"{q:.5f}"


def apnd(path:str)->pd.DataFrame:
	if not path.endswith("/"):path+="/"
	return pd.concat(
        [pd.read_csv(f"{path}{q.name}") for q in
         os.scandir(path) if ".csv" in q.name],axis=0)


def mon(f:pd.DataFrame,start,stop)->np.ndarray:
    return np.asarray(
        sum([f.index.month==q for q in np.arange(start,stop,1)]),
        dtype="bool")


def full_range_idx(f:pd.DataFrame):
    return (pd.DataFrame(
        pd.date_range(f.index.min(),f.index.max()),
        columns=["date"]).set_index("date"))


def upd(url:str,i:str):
    options=webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver=webdriver.Chrome(options=options)
    driver.get(url)
    cache=[]
    for x in np.arange(1,20,1):
        if "currencies" in url:
            date=driver.find_element(By.XPATH,
            f'''//*[@id="__next"]/div/div/div/div[2]/main/div/div[4]/'''
            f'''div/div[1]/div/div[3]/div/table/tbody/tr[{x}]/td[1]/time''').text
            value=driver.find_element(By.XPATH,
            f'''//*[@id="__next"]/div/div/div/div[2]/main/div/div[4]/'''
            f'''div/div[1]/div/div[3]/div/table/tbody/tr[{x+1}]/td[2]''').text
            cache.append((date,value))
        else:
            date=driver.find_element(By.XPATH,
            f'''//*[@id="__next"]/div/div/div/div[2]/main/div/div[4]/'''
            f'''div/div/div[3]/div/table/tbody/tr[{x}]/td[1]/time''').text
            value=driver.find_element(By.XPATH,
            f'''//*[@id="__next"]/div/div/div/div[2]/main/div/div[4]/'''
            f'''div/div/div[3]/div/table/tbody/tr[{x+1}]/td[2]''').text
            cache.append((date,value))
    driver.close()
    driver.quit()
    s=pd.DataFrame(cache,columns=["date",f"{i}"])
    s["date"]=pd.to_datetime(s["date"])
    s=s.set_index("date").iloc[:,0].str.replace(",","")
    return s


def upd_(url:str,i:str):
    import requests
    from bs4 import BeautifulSoup as bs
    ua={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "\
    "AppleWebKit/537.36 (KHTML, like Gecko) "\
    "Chrome/104.0.0.0 Safari/537.36"}
    cnxt=bs(requests.get(url,headers=ua).text)
    # bs::parsables
    cnxt=cnxt.select("tbody")
    # tbodies[1]
    cnxt=cnxt.find_all("tr",class_="datatable_row__2vgJl")
    # trs iterable
    cache=[]
    for a in range(len(cnxt)):
        # each-tr
        date=cnxt[a].select("time")[0]["datetime"]
        val =cnxt[a].select("td")[1].text
        cache.append((date,val))
    s=pd.DataFrame(cache,columns=["date",f"{i}"])
    s["date"]=pd.to_datetime(s["date"])
    s=s.set_index("date").iloc[:,0].str.replace(",","")
    return s


def getdata(
    local=True,update=True)->pd.DataFrame:
    t0=time()
    if local:
        f=pd.read_csv(f"{PATH}data0.csv",
            index_col="date",
            converters={"date":pd.to_datetime},
            na_filter=False)
    else:
        fred=Fred(f"{PATH}fed")
        fs={i:fred.get_series_df(intern[i])
            .loc[:,"date":]
            .astype({"date":"datetime64[ns]"})
            .set_index("date")
            .rename(columns={"value":i}
            .copy()) for i in intern}
        fsincsv=[pd.read_csv(q,
            index_col="date",
            converters={"date":pd.to_datetime},
            na_filter=False) for q in sorted(glob(rf"{PATH}da_*.csv"))]
        for q in range(len((fsincsv))):
            fs[f"{fsincsv[q].columns[0]}"]=fsincsv[q]
        f=pd.concat(fs.values(),axis=1)
    if update:
        [f.update(upd(extern[i],i)) for i in extern]
        f=f.apply(pd.to_numeric,errors="coerce")
    if not local:
        f.to_csv(f"{PATH}data0.csv",encoding="utf-8-sig")
    os.system('cls||clear')
    print(f"getdata:: {time()-t0:.1f}s::{local=},{update=}")
    return f


def impt(f:pd.DataFrame,
        x:list=None,n=10)->pd.DataFrame:
    if x is None:f=f.columns
    return (pd.DataFrame(
        KNNImputer(n_neighbors=n,weights="distance").fit_transform(f[x]),
        index=f.index,columns=x))


def dtr(a,
        o:int=3):
    if any(np.isnan(a)):raise TypeError(f"nan in the array")
    x=np.arange(len(a))
    q=np.polyval(np.polyfit(x,a,deg=o),x)
    a-=q
    return a


def mm(a:pd.DataFrame)->pd.DataFrame:
    return (a-a.min())/(a.max()-a.min())


def zs(f:pd.DataFrame,save=False)->pd.DataFrame:
    fcache=[]
    for i in tqdm(f.columns,desc="z-score"):
        q=f[i].dropna()
        if i=="yt":
            q=dtr(q)
        q_=mm(q)
        lz=pd.DataFrame(
            scipy.stats.zscore(
            scipy.stats.yeojohnson(q_)[0]),
            index=q.index)
        zp=pd.DataFrame(
            scipy.stats.norm.cdf(lz,
            loc=np.mean(lz.to_numpy()),scale=np.std(lz)),
            index=q.index)
        w=(pd.concat([q,lz,zp],axis=1)
            .set_axis([f"{i}",f"{i}lz",f"{i}lzp"],axis=1))
        fcache.append(w)
    f=full_range_idx(f).join(pd.concat(fcache,axis=1),how="left")
    if save:
        f.to_csv(f"{PATH}data1.csv",encoding="utf-8-sig")
    return f


def rng(f:pd.DataFrame,i:str,
        rng=(.05,3),test=False)->pd.DataFrame:
    f=f.copy()
    if not i in f.columns:
        raise NameError(f"{i} does not exist")
    col=f"{i}lzp"
    if test:
        rng=np.delete(
            np.round(np.flip(np.geomspace(rng[0],1,rng[1])),
            2),
            2)
    else:
        rng=np.round(np.flip(np.percentile(f[col].dropna(),(2,15,30,100))),
            2)
    f.loc[:,f"{col}rng"]=None
    #heaviside
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
    q=f.iloc[rowidx,colidx:colidx+3]
    w=q[f"{i}lzp"]
    print(f"{w*100:.2f}%")
    return q,(rowidx,colidx)


def ns(f:pd.DataFrame,x:str,y:str):
    f=f[[x,y]].dropna()
    rowidx=np.amin((f.count()[x],f.count()[y]))
    return f[f.shape[0]-rowidx:]


def cx(f:pd.DataFrame,x:str,y:str,
    d=24,normed=True,save=True,test=False,dtr=None):
    f=ns(f,x,y)
    if save:
        plt.figure(figsize=(22,14))
        xc=plt.xcorr(f[x],f[y],
            detrend=dtr,maxlags=d,
            normed=normed)
        plt.suptitle(f"{x},{y},{d}")
        plt.savefig(f"e:/capt/{x}_{y}_{d}.png")
        plt.cla()
        plt.clf()
        plt.close()
        idx=xc[1].argmax()
        return xc[0][idx],xc[1][idx]
    fg,ax=plt.subplots(1,2)
    ac=ax[0].acorr(f[x],
        detrend=scipy.signal.detrend,maxlags=d)
    xc=ax[1].xcorr(f[x],f[y],
        detrend=scipy.signal.detrend,maxlags=d,
        normed=normed)
    fg.suptitle(f"{x},{y},{d}")
    ac_=ac[0][ac[1].argmax()]
    xc_=xc[0][xc[1].argmax()]
    if test:
        return (ac[0],ac[1]),(xc[0],xc[1])
    return ac_,xc_


def cx_(f:pd.DataFrame,x,
        d=12):
    f=f[x]
    cache=[]
    for col in tqdm(list(product(x,x)),desc=f"cx-rel"):
        if not col[0] is col[1]:
            rslt=cx(f[[col[0],col[1]]],
                col[0],col[1],d=d,save=True)
            cache.append((col[0],col[1],rslt[0],np.round(rslt[1],2)))
    rslt=(pd.DataFrame(cache,columns=["x","y","dur","coef"])
            .sort_values(by="coef",ascending=False)
            .set_index(keys=["x","y"]))
    rslt.to_csv(f"{PATH}cxr.csv")
    return rslt


def cx__(f:pd.DataFrame):
    from statsmodels.tsa.stattools import grangercausalitytests
    data=f[["yt","fr"]].bfill().diff().dropna()
    rslt=grangercausalitytests(data,[a for a in np.arange(12,21)])
    import statsmodels.api
    statsmodels.api.tsa.stattools.ccf(a0,a1,adjusted=False)
    ...


def regr(x,y,s=0.2,cv=5,typ="rf"):
    xi,xt,yi,yt=train_test_split(x,y,test_size=s)
    if typ=="rf":
        params={"max_depth":
                    [a for a in np.arange(8,65,16)],
                "n_estimators":
                    [a for a in np.arange(8,65,16)],
                "max_features":
                    ["sqrt"],
                "min_samples_leaf":
                    [a for a in np.arange(20,61,20)],
                "n_jobs":[-1],
                "random_state":[rnd]}
        regressor=rfr()
    elif typ=="gb":
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
    proc_f.update(mm(proc_f[[q for q in proc_f.columns if q!="ic"]]))
    proc_f.update(proc_f.ic.dropna().pct_change())
    proc_f=proc_f.interpolate("quadratic",limit=3).dropna(thresh=thresh)
    proc_f.update(proc_f.ic.ffill())
    proc_f=impt(proc_f,x=proc_f.columns,n=10)
    return (pd.DataFrame(
            [scipy.stats.yeojohnson(proc_f[q])[0] for q in proc_f.columns],
            index=proc_f.columns,columns=proc_f.index)
            .T.join(f[y].shift(-60).bfill()))


def regr_(f:pd.DataFrame,t,typ,
        x=["cb","yt","ys","ng","cl","zc","zw","uj","ue","ic"],
        y=["pi","ci","ii"],
        thresh=6,cv=5,test=1):
    f=proc(f,x=x,y=y,thresh=thresh)
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
    i=regr(x0,y0,cv=cv,typ=typ)
    return {f"{t}":[i,x1.assign(i_=i.best_estimator_.predict(x1))],
            f"{t}_stat":i.cv_results_}


# innermost visualisations
def hm(f,
    mm=(-1,1),ax=None,cbar=False,title=None):
    if title is None:
        title=f"{', '.join(f.columns)}"
    corr=f.corr("spearman")
    mask=np.triu(np.ones_like(corr,dtype=bool))
    cmap=sns.diverging_palette(240,10,as_cmap=True)
    if ax is None:
        plt.subplots(figsize=(12,12))
    plt.title(title)
    sns.heatmap(corr,
        mask=mask,cmap=cmap,ax=ax,cbar=cbar,
        vmin=mm[0],vmax=mm[1],
        annot=True,center=0,square=True,linewidths=.5,fmt=".2f")


def hm_(f):
    if len(f.columns)>20:
        q=input(f"{len(f.columns)} columns:: ")
        if not q:raise ValueError(f"{len(f.columns)} is too much")
    _,ax=plt.subplots(1,3,figsize=(22,16))
    hm(f,
        title=f"",ax=ax[0]),ax[0].title.set_text("org")
    hm(f.dropna(),
        title=f"",ax=ax[1]),ax[1].title.set_text("dropna")
    hm(impt(f,f.columns),
        title=f"",ax=ax[2]),ax[2].title.set_text("impt")


def pp(f,
    vars=None,l=False,hue=None):
    (sns.pairplot(data=f,vars=vars,hue=hue,
        dropna=False,kind="scatter",diag_kind="hist",palette=P)
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
