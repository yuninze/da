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