import streamlit as st
from PIL import Image
from da import *
from datetime import datetime

# ornament
st.set_page_config(page_title="yiz",layout="centered")
print(f"{datetime.now()}::initialized in {os.getcwd()}")

# yielding
def get_data(path="c:/code/f.csv"):
    return pd.read_csv(path,index_col="date",
        converters={"date":pd.to_datetime})

# head
st.header("Product Snapshot")
f=get_data()
prods=["Indices","Commodities","Images","Citations"]
t0,t1,t2,t3=st.tabs(prods)

# tabs
with t0:
    base=st.empty()
    with base.container():
        # row 0
        st.subheader("Latest Quotes")
        cols_nm=["US10Y","US10Y-I","C.B.Y.S.","L.S.Y.S.","F.F.R."]
        cols=["by","iy","cb","ys","fr"]
        data={q:f[q].dropna().iloc[-2:] for q in cols}
        for q in enumerate(st.columns(5)):
            q[1].metric(
                label=f"{cols_nm[q[0]]}",
                value=f"{data[cols[q[0]]][-1]:.3f}",
                delta=f"{data[cols[q[0]]].diff()[-1]:.3f} "
                      f"({data[cols[q[0]]].pct_change()[-1]*100:.2f}%)")
        st.subheader(f"Inflation-indexed US10Y")
        st.caption("인플레이션 반영 10년 국채 일드")
        fg,ax=plt.subplots(figsize=(10,6))
        data=f.iy.dropna().loc["2018":]
        ax.plot(data,alpha=.8,color="darkorange")
        ax.axhline(color="red",alpha=.5,y=data.iloc[-1])
        st.pyplot(fg)
        # row 1
        st.subheader(f"US10Y+FFR+Upcoming FFR")
        st.caption("10년 국채 일드+금리(익일담보대여금리)+예상 차기 금리")
        fg,ax=plt.subplots(figsize=(10,6))
        data=f[["fr","by"]].loc["2018":].dropna()
        ax.plot(data,alpha=.8)
        ax.axhline(color="red",alpha=.5,y=data.iloc[-1,0]+.5)
        st.pyplot(fg)
        # row 2
        st.subheader(f"Long-Short Yield Spread")
        st.caption("장단기 국채 일드 스프레드(10년물 일드-3개월물 일드)")
        freq="3d"
        f_cols=["hs","ic","cb","ys"]
        f_cols_name=["HSI","JC","CBYS","LSYS"]
        f0=f[f_cols]
        f1=f0.resample(freq).mean().dropna()
        f2=f1[["hs","ic","cb"]].apply(lambda q:scipy.stats.zscore(scipy.stats.yeojohnson(q)[0]))
        f2["ys"]=scipy.stats.zscore(f1["ys"])
        f3=f2.loc["2020":].set_axis(f_cols_name,axis=1)
        fg,ax=plt.subplots(figsize=(10,6))
        ax.plot(f3,alpha=.8)
        ax.legend(f_cols_name)
        ax.set_ylabel("log/z-score")
        ax.set_xlabel(f"{freq=}")
        plt.xticks(rotation=45)
        st.pyplot(fg)
        # row 3
        st.subheader(f"Correleation")
        st.caption(f"1차 변화량의 Pearson Co-coef")
        freq="2d"
        f_cols=["by","cb","hs","cl","ie"]
        f0=f[f_cols]
        f1=f0.resample(freq).mean().diff().dropna()
        f2=f1.apply(lambda q:scipy.stats.zscore(scipy.stats.yeojohnson(q)[0]))
        fg,ax=plt.subplots(figsize=(5,5))
        hm(f2.corr(),ax=ax)
        st.pyplot(fg)
        st.text("end of page")
with t1:
    # row 0
    st.subheader("Latest Quotes")
    cols_nm=["W.T.I.","Nat-Gas","Silver","Copper","Soybean"]
    cols=["cl","ng","si","hg","zs"]
    ie=deflator(f)
    data={q:act(f[q],ie).dropna().iloc[-5:] for q in cols}
    for q in enumerate(st.columns(5)):
        data_=data[cols[q[0]]]
        data_name=cols_nm[q[0]]
        data_vals=data_.iloc[-2:]
        q[1].metric(
            label=f"{data_name}",
            value=f"{data_vals.iat[-1,0]:.2f}",
            delta=f"{data_vals.iat[-1,2]*100:.1f}%",
            delta_color="off")
    # row 1
    st.subheader("x-day Rolled Standard Deviation")
    st.caption("일 평균 변화율의 표준편차의 표준점수 1, 2로, 이론적 근거 없음.")
    dur=st.slider(
        "Duration",
        min_value=1,
        max_value=200,
        value=10,
        step=1,)
    cols=["cl","ng","si","hg","zs"]
    for q in enumerate(cols):
        st.pyplot(roll_pct(f,q[1],dur=dur,figsize=(8,6),title=f"{cols_nm[q[0]]}"))
    st.text("end of page")
with t2:
    st.subheader("Images")
    imgs={q.name:Image.open(f"asset/{q.name}") for q in os.scandir("asset") if q.name.endswith(".png") or q.name.endswith(".jpg")}
    if not len(imgs)==0:
        img_sel=st.radio("images",imgs.keys(),label_visibility="hidden")
        for img_label in imgs.keys():
            if img_label==img_sel:
                st.image(imgs[img_label],)
    else:
        st.caption(f"No imagefiles")
with t3:
    st.subheader("Citations")
    with open(f"asset/cite.txt",encoding="utf-8-sig") as citefile:
        sents=citefile.readlines()
    for q in sents:
        st.markdown(q)