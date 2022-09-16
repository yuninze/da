from glob import glob
import requests
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from full_fred.fred import Fred
from bs4 import BeautifulSoup as bs
from time import time as t


# innermost params
pd.options.display.min_rows=6
pd.options.display.float_format=lambda q:f"{q:.5f}"


def truthy(*vals):
    for x in vals:
        if not x:
            raise SystemExit(f"{x}")


def full_range_idx(f:pd.DataFrame,
    test=False)->pd.DataFrame:
    if test: # frame with full-range date indices
        return np.arange(
            f.index.min(),f.index.max(),dtype="datetime64[D]")
    return (pd.DataFrame(
        pd.date_range(
            f.index.min(),f.index.max()),columns=["date"])
            .set_index("date"))


def messij(f:pd.DataFrame)->pd.DataFrame:
    return f.apply(pd.to_numeric,errors="coerce")


def upd(url:str,i:str):
    naiyou=bs(requests.get(url).text) # bs::parsable tags
    naiyou=naiyou.select("tbody")[1] # tbodies[1]
    naiyou=naiyou.find_all("tr",class_="datatable_row__2vgJl") # trs as iterable
    cache=[]
    for a in range(len(naiyou)): # each tr
        s=2 if a==0 else 1
        date=naiyou[a].select("time")[0]["datetime"]
        val =naiyou[a].select("td")[s].text
        cache.append((date,val))
    s=pd.DataFrame(cache,columns=["date",f"{i}"])
    s["date"]=pd.to_datetime(s["date"],yearfirst=True)
    s=((s.set_index("date").iloc[:,0].str.replace(",",""))
        .astype(float))
    return s


def getdata(local=True,update=True,roll=False)->pd.DataFrame:
    t0=t()
    if local:
        f=(pd.read_csv("c:/code/data0.csv",
            index_col="date",
            converters={"date":pd.to_datetime},
            na_filter=False)
            .apply(pd.to_numeric,errors="coerce"))
    else:
        int={
            "cys":"BAMLH0A0HYM2",
            "5yi":"T5YIE",
            "10yt":"DGS10",
            "ng":"DHHNGSP",
            "wti":"DCOILWTICO",
            "ffr":"DFF"
        }
        ext={
            "zs":"https://www.investing.com/commodities/us-soybeans-historical-data",
            "hg":"https://www.investing.com/commodities/copper-historical-data",
        }
        fed=Fred("c:/code/fed")
        fs={i:fed.get_series_df(int[i])
            .loc[:,"date":]
            .astype({"date":"datetime64[ns]"})
            .set_index("date")
            .rename(columns={"value":i}) for i in int}
        fsincsv=[pd.read_csv(q,
            index_col="date",
            converters={"date":pd.to_datetime},
            na_filter=False) for q in sorted(glob(r"c:/code/da_*.csv"))]
        for q in range(len((fsincsv))):
            fs[f"{fsincsv[q].columns[0]}"]=fsincsv[q]
        f=messij(pd.concat(fs.values(),axis=1))
        if update:
            [f.update(upd(ext[i],i)) for i in ext]
        f.to_csv("c:/code/data0.csv",encoding="utf-8-sig")
    if roll:
        try:
            f=f.rolling(5,min_periods=5).mean()
            print(f"min_periods=5")
        except:
            f=f.rolling(2,min_periods=2).mean()
            print(f"min_periods=2")
    print(f"elapsed {t()-t0:.2f}s (getdata): got")
    return f


def zs(f:pd.DataFrame,pctrng=(0.1,99.9),intp=False)->pd.DataFrame:
    t0=t()
    fcache=[]
    for i in f.columns:
        if intp:
            q=f[i].interpolate(method="cubic").interpolate("index").dropna()
        else:
            q=f[i].dropna()
        mm=np.percentile(q,pctrng)
        q[(q<=mm[0])|(q>=mm[1])]=np.nan
        q=q.dropna()

        nor_zs=scipy.stats.zscore(q)

        q_=(q-q.min())/(q.max()-q.min())
        log_zs=pd.DataFrame(scipy.stats.zscore(
            scipy.stats.yeojohnson(q_)[0]),index=q.index)
        
        prb_zs=pd.concat(
                [pd.DataFrame(w,index=q.index) for w in  
                [scipy.stats.norm.pdf(np.absolute(e)) for e in 
                [nor_zs,log_zs]]]
                ,axis=1)
        
        w=(pd.concat([q,nor_zs,log_zs,prb_zs],axis=1)
            .set_axis(
                [f"{i}",f"{i}nzs",f"{i}lzs",f"{i}nzsprb",f"{i}lzsprb"],
                axis=1))
        
        fcache.append(w)
        print(f"zs::col::{i}")
    f=full_range_idx(f).join(pd.concat(fcache,axis=1),how="left")
    f.to_csv("c:/code/data1.csv",encoding="utf-8-sig",)
    print(f"elapsed {t()-t0:.2f}s (zs): {pctrng}")
    return f


def prng(f:pd.DataFrame,i:str,rng=(.05,5),dropna=True,
    test=False)->pd.DataFrame:
    t0=t()
    if not i in f.columns:
        raise NameError(f"{i} not exist in columns")
    colp=f"{i}lzsprb"
    if test:
        rng=np.delete(np.round(np.flip(
                    np.geomspace(rng[0],1,rng[1])
                    ),2),2)
    else:
        rng=np.round(np.flip(np.percentile(
            f[colp].dropna(),(5,10,20,40,100)
            )),2)
    f.loc[:,f"{colp}rng"]=None
    for q in range(len(rng)):
        f.loc[:,f"{colp}rng"]=np.where(
            (~pd.isna(f[colp])) & (f[colp]<=rng[q]),
            rng[q],f[f"{colp}rng"])
    if dropna:
        f=f.dropna(subset=f"{colp}rng")
    f.update(f[f"{colp}rng"].astype("category"))
    print(f"elapsed {t()-t0:.2f}s (prng): {rng}")
    return f


def mon(f:pd.DataFrame,start:int,stop:int)->np.ndarray:
    a=sum([f.index.month==q for q in np.arange(start,stop,1)])
    return np.asarray(a,dtype="bool")


def locate(f,i:str,v:float,test=True):
    rowidx=np.abs(f[f"{i}"]-v).argmin()
    colidx=f.columns.get_indexer([f"{i}"])[0]
    q=f.iloc[rowidx,colidx:colidx+5]
    if test:
        print(f"{q[4]*100:.2f}% ({rowidx=}, {colidx=})")
    return q


def exec(i,prng_=(.03,5),local=False,roll=False,intp=False):
    hue=f"{i}zsprbrng"
    f=getdata(local=local,roll=roll)
    ff=zs(f,intp=intp)
    fff=prng(ff,i,rng=prng_)
    return f,ff,fff


def hm(q,title="heatmap [-1,1]",
    minmax=(-1,1),
    fmt=".2f",corner=True,
    ax=None,
    cbar=True):
    q_corr=q.corr()
    if corner:
        mask=np.triu(np.ones_like(q_corr,dtype=bool))
    else:
        mask=None
    cmap=sns.diverging_palette(240,10,as_cmap=True)
    sns.set_theme(font="monospace")
    sns.heatmap(q_corr,
        vmin=minmax[0],vmax=minmax[1],
        mask=mask,cmap=cmap,
        annot=True,center=0,square=True,linewidths=.5,fmt=fmt,
        ax=ax,cbar=cbar)
    plt.title(title)


def vis(f,c_="deep"):
    if len(f.columns)>20:return f"{len(f.columns)} columns are too much"

    sns.set_style("whitegrid")
    sns.set_context("talk")

    mm=(-1,1)
    fg,ax=plt.subplots(1,3)
    hm(f,
    minmax=mm,title=f"org",cbar=False,ax=ax[0])
    hm(f.dropna(),
    minmax=mm,title=f"dropna",cbar=False,ax=ax[1])
    hm(f.interpolate(method="time").dropna(),
    minmax=mm,title=f"interpolated, dropna",cbar=False,ax=ax[2])

    f_=f.interpolate(method="time")
    (sns.pairplot(f_,
        vars=f.columns,
        hue=None,
        dropna=False,
        kind="scatter",
        diag_kind="hist",
        palette=c_)
    .map_diag(sns.histplot,multiple="stack",element="step"))

    # sns.relplot(ff,x="30ym",y="zs",
    #     hue=None,palette=c_,
    #     size="wti",sizes=(1,150))

    # fg,ax=plt.subplots(1,3,figsize=(12,12))
    # fg.suptitle("da_ed")
        
    # sns.kdeplot(fs0Tot["ng"],x="ng")
    # sns.displot(fs0Tot["ng"],x="ng",bins=10)
