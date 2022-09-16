import requests
import pandas as pd
from bs4 import BeautifulSoup as bs

url="https://www.investing.com/commodities/us-soybeans-historical-data"

q=bs(requests.get(url).text) # bs::parsable tags
q=q.select("tbody")[1] # select tbodies[1]
q=q.find_all("tr",class_="datatable_row__2vgJl") # get trs as iterable

def mori(context)->pd.DataFrame:
    cache=[]
    for a in range(len(context)): # each tr
        s=2 if a==0 else 1
        date=context[a].select("time")[0]["datetime"]
        val =context[a].select("td")[s].text
        cache.append((date,val))
    s=pd.DataFrame(cache,columns=["date","zs"])
    s["date"]=pd.to_datetime(s["date"],yearfirst=True)
    s=s.set_index("date")
    return s