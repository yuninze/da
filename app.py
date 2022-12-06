import streamlit as st
from datetime import datetime
from PIL import Image
from da import *

# ornament and asset
st.set_page_config(page_title="yiz",layout="wide")
imgs={q.name:Image.open(f"asset/{q.name}") for q in os.scandir("asset") if q.name.endswith(".png")}
print(f"{datetime.now()}::initialized in {os.getcwd()}")

def get_data(path="c:/code/f.csv"):
    return pd.read_csv(path,index_col="date",converters={"date":pd.to_datetime})

def main():
    f=get_data()
    st.header("Product Snapshot")
    cat0=["bonds","indices","commodities"]
    cat0_element=st.sidebar.radio("Target Product",cat0)
    for label in cat0:
        if label==cat0_element:
            show(f,None,f"{label}")
            print(f"{datetime.now()}::something has been shown")
            break
        else:
            print(f"{datetime.now()}::wasn't {label}")
            continue

def show(f,col,cmd):
    if col is None:col=f.columns
    f=f[f.columns]
    if cmd=="bonds":
        f_bonds_latest_data=f[["iy","by","fr"]].dropna(thresh=2).tail(3).T

        st.subheader(f"Latest Quotes")
        st.dataframe(f_bonds_latest_data)

        st.subheader(f"Inflation-indexed US10Y")
        fg,ax=plt.subplots(figsize=(5,5))
        data=f.iy.dropna()
        ax.plot(data.loc["2016":],alpha=0.8,color="darkorange")
        ax.axhline(color="red",alpha=0.5,y=data.iloc[-1])
        st.pyplot(fg)

        st.subheader(f"US10Y Quote + FFR + Upcoming FFR")
        fg,ax=plt.subplots(figsize=(5,5))
        data=f[["fr","by"]].dropna()
        ax.plot(data.loc["2016":],alpha=.8)
        ax.axhline(color="red",alpha=0.5,y=data.iloc[-1,0]+.45)
        st.pyplot(fg)

        st.subheader(f"Long-Short Yield Spread")
        freq       ="2d"
        f_cols     =["hs","ic","cb","ys"]
        f_cols_name=["HSI","JC","CBYS","LSYS"]
        f0=f[f_cols]
        f1=f0.resample(freq).mean().dropna()
        f2=f1[["hs","ic","cb"]].apply(lambda q:scipy.stats.zscore(scipy.stats.yeojohnson(q)[0]))
        f2["ys"]=scipy.stats.zscore(f1["ys"])
        f3=f2.loc["2007":"2009"].set_axis(f_cols_name,axis=1)
        fg,ax=plt.subplots(figsize=(8,8))
        ax.plot(f3,alpha=.5)
        ax.legend(f_cols_name)
        ax.set_ylabel("log/z-score")
        ax.set_xlabel("2-day freq")
        plt.xticks(rotation=45)
        st.pyplot(fg)

        print(f"{datetime.now()}::{cmd}")
    elif cmd=="ng":
        fg,ax=plt.subplots(2,figsize=(3,6))
        ax[0].hist(scipy.stats.yeojohnson(f.ng.dropna())[0],alpha=.6)
        ax[1].hist(np.log(f.ng.dropna()),alpha=.6)
        st.pyplot(fg)
    elif cmd=="relevance":
        fg,ax=plt.subplots(1,figsize=(5,5))
        freq  ="2d"
        f_cols=["by","cb","hs","cl","ie"]
        f0=f[f_cols]
        f1=f0.resample(freq).mean().diff().dropna()
        f2=f1.apply(lambda q:scipy.stats.zscore(scipy.stats.yeojohnson(q)[0]))
        hm(f2.corr(),ax=ax)
        st.pyplot(fg)
    elif cmd=="roll":
        fg,ax=plt.subplots(1,figsize=(8,8))
        roll_pct(f,"ng",ax=ax)
        st.pyplot(fg)

if __name__=="__main__":
    main()