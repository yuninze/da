def upd(url:str,i:str):
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    options=webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-extensions")
    driver=webdriver.Chrome(options=options)
    driver.implicitly_wait(10)
    driver.get(url)
    cache=[]
    try:
        for x in range(1,27,1):
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
    except:
        driver.close()
        driver.quit()
        print(f"{url}")
        return None
    driver.close()
    driver.quit()
    s=pd.DataFrame(cache,columns=["date",f"{i}"])
    s["date"]=pd.to_datetime(s["date"])
    s=s.set_index("date").iloc[:,0].str.replace(",","")
    return s


def upd_(url:str,i:str):
    import requests
    import pandas as pd
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


def zs(f:pd.DataFrame)->pd.DataFrame:
    f=f.copy()
    fs=[]
    for i in f.columns:
        q=f[i].dropna()
        lz=pd.DataFrame(
            scipy.stats.zscore(scipy.stats.yeojohnson(q)[0]),
            index=q.index)
        lzp=pd.DataFrame(
            scipy.stats.norm.cdf(lz,loc=lz.mean(),scale=lz.std(ddof=1)),
            index=q.index)
        w=(pd.concat([q,lz,lzp],axis=1)
            .set_axis([f"{i}",f"{i}lz",f"{i}lzp"],axis=1))
        fs.append(w)
    f=index_full_range(f).join(pd.concat(fs,axis=1),how="left")
    return f


# rf regressor
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


class etp:
    # p=10000
    # c=c/250
    # i0=100
    # i1=85
    # f"{1-(i1/i0)}"
    def iv(self):
        delta=abs(self.bv1-self.bv0)/self.bv0
        if type>0:
            basis=1+delta
        else:
            basis=1-delta
        diff=self.lq*(basis-1)
        iv=(self.lq*basis)*(1-(self.etc/365))
        print((self.lq,
            self.bv0,
            self.bv1,
            self.etc,round(self.etc/365,5)))
        return {"delta":delta,
            "difference":diff,
            "iv":iv}