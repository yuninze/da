import streamlit as st
from datetime import datetime
from da import *

# ornament
dff_next_bp=.25
st.set_page_config(page_title="Product Snapshot",layout="centered")
print(f"{datetime.now()}::initialized")

# yielding st.experimental_memo
def getdata_(path="c:/code/f.csv"):
    return pd.read_csv(path,index_col="date",converters={"date":pd.to_datetime})
f=getdata_()

# head
st.header("Product Snapshot")

# tabs
prods=["Indices","Commodities","Macro","Utilities","Citations"]
t0,t1,t2,t3,t4=st.tabs(prods)

# tab
with t0:
    # row
    st.subheader("Latests")
    st.markdown("Innermost criterions")
    cols=["us02y","iy","cb","ys","fr"]
    cols_nm=["U.S.02Y","U.S.10Y-I","C.B.Y.S.","L.S.Y.S.","F.F.R."]
    data={q:f[q].dropna().iloc[-2:] for q in cols}
    for q in enumerate(st.columns(len(cols))):
        q[1].metric(
            label=f"{cols_nm[q[0]]} ({data[cols[q[0]]].index.max().strftime('%m-%d')})",
            value=f"{data[cols[q[0]]][-1]:.3f}",
            delta=f"{data[cols[q[0]]].diff()[-1]:.3f} "
                    f"({data[cols[q[0]]].pct_change()[-1]*100:.2f}%)")
    st.subheader("Inflation-indexed U.S.10Y")
    st.markdown("인플레이션 반영 10년 국채 일드는 국채 10년물 수익률을 CPI-deflating한 것으로 명목 국채 일드임. 10-Years TIPS-US10Y=10Y BIR")
    fg,ax=plt.subplots(figsize=(8,4))
    data=f.iy.loc["2010":].dropna()
    ax.plot(data,alpha=.7,color="darkorange")
    ax.axhline(color="orange",alpha=.7,y=data.iloc[-1])
    st.pyplot(fg)
    # row
    st.subheader("U.S.02Y, F.F.R.")
    st.markdown("2년 국채 일드, 담보대익일조달금리 및 예상 차기 금리.")
    fg,ax=plt.subplots(figsize=(8,4))
    data=f[["fr","us02y"]].loc["2010":].dropna()
    ax.plot(data,alpha=.8)
    ax.axhline(color="blue",alpha=.7,y=data.iloc[-1,0]+dff_next_bp)
    ax.axhline(color="orange",alpha=.7,y=data.iloc[-1,1])
    st.pyplot(fg)
    # row
    st.subheader("U.S.02Y-F.F.R.")
    st.markdown("Lower US02Y-SOFR means upcoming easing or stopping hiking.")
    fg,ax=plt.subplots(figsize=(8,4))
    data=f["us02y"]-f["fr"].loc["2010":].dropna()
    ax.plot(data,alpha=.8)
    st.pyplot(fg)
    # row
    st.subheader("Long-Short Yield Spread")
    st.markdown("10년물-3개월물 일드 스프레드, 기업채 일드 스프레드.")
    freq="2d"
    f_cols=["ic","cb","ys"]
    f_cols_name=["JC","CBYS","LSYS"]
    f0=f[f_cols]
    f1=f0.resample(freq).mean().dropna()
    f2=f1[["ic","cb"]].apply(lambda q:scipy.stats.zscore(scipy.stats.yeojohnson(q)[0],nan_policy="omit"))
    f2["ys"]=scipy.stats.zscore(f1["ys"],nan_policy="omit")*-1
    f3=f2.loc["2010":].set_axis(f_cols_name,axis=1)
    fg,ax=plt.subplots(figsize=(8,4))
    ax.plot(f3,alpha=.8)
    ax.legend(f_cols_name)
    ax.set_ylabel("log/z-score")
    ax.set_xlabel(f"{freq=}")
    plt.xticks(rotation=45)
    st.pyplot(fg)
    # row
    st.subheader("Correleation")
    st.markdown("몇 가지 계열의 1차 변화량의 Pearson 상관계수.")
    freq="3d"
    f_cols=["us02y","ys","sp","hs","hg","si","zs","zc"]
    f0=f[f_cols]
    f1=f0.resample(freq).mean().diff().dropna()
    fg,ax=plt.subplots(figsize=(6,6))
    hm(f1.corr(),ax=ax)
    st.pyplot(fg)
with t1:
    # param
    cols=["cl","ng","si","hg","zs"]
    cols_nm=["W.T.I.","Nat-Gas","Silver","Copper","Soybean"]
    # row 0
    st.subheader("Latests")
    st.markdown("주요 원자재의 가격, 명목 가격의 상위 %. ADF를 1% 신뢰구간에서 통과하는 가격의 상품도 있음.")
    ie=deflator(f)
    data={q:act(f[q],ie).dropna().iloc[-5:] for q in cols}
    for q in enumerate(st.columns(len(cols))):
        data_=data[cols[q[0]]]
        data_name=cols_nm[q[0]]+f" ({data_.index.max().strftime('%m-%d')})"
        data_vals=data_.iloc[-2:]
        q[1].metric(
            label=f"{data_name}",
            value=f"{data_vals.iat[-1,0]:.2f}",
            delta=None,delta_color="off")
    # row 1
    for q in enumerate(st.columns(len(cols))):
        data_=data[cols[q[0]]]
        data_name=cols_nm[q[0]]
        data_vals=data_.iloc[-2:]
        q[1].metric(
            label=f"{data_name} LP",
            value=f"{data_vals.iat[-1,2]*100:.2f}%",
            delta=None,delta_color="off")
    # row 2
    st.subheader("Rolled Standard Deviation")
    st.markdown("x일 변화율 평균의 표준편차의 1,2 표준점수(σ). 상품 가격과 같이 천변만화하는 숫자에 대한 평활법의 실효성은 없음.")
    dur=st.slider(
        "Duration (days)",
        min_value=5,
        max_value=200,
        value=200,
        step=5,)
    cols=["cl","ng","si","hg","zs"]
    for q in enumerate(cols):
        st.pyplot(roll_pct(f,q[1],dur=dur,figsize=(8,4),title=f"{cols_nm[q[0]]}"))
with t2:
    # params
    # row 0
    st.subheader("Coporate Bond Yield Spread vs. Labor Market")
    st.markdown("Coporate bond yield is a prominent forecaster of labor market.")
    cols=["cb","ic","pr","ur"]
    cols_name=["CBYS","ICSA","NFP","UNRATE"]
    data=(f[cols].resample("m").mean().dropna(how="all")
        .set_axis(cols_name,axis=1)
        .apply(lambda q:scipy.stats.zscore(q,nan_policy="omit")))
    dur=st.slider(
        "Duration (year)",
        min_value=data.index.min().year,
        max_value=data.index.max().year,
        value=(2021,2023),
        step=1,key="0")
    data_=data.loc[f"{dur[0]}":f"{dur[1]}"]
    fg,ax=plt.subplots(figsize=(8,4))
    ax.plot(data_,alpha=.8)
    ax.set_ylabel("z-score")
    ax.legend(cols_name)
    plt.xticks(rotation=60)
    st.pyplot(fg)
    # row 1
    st.subheader("Price Indices")
    st.markdown("Price indices are a deflator of given period. Those MoM-change percentages are shifted for 1 month.")
    cols=["pi","ci","ii"]
    cols_name=["PPI","CPI","PCE"]
    data=f[cols].resample("m").mean().pct_change().shift()[3:]
    dur=st.slider(
        "Duration (year)",
        min_value=data.index.min().year,
        max_value=data.index.max().year,
        value=(2020,2023),
        step=1,key="1")
    data_=data.loc[f"{dur[0]}":f"{dur[1]}"]
    fg,ax=plt.subplots(figsize=(8,4))
    ax.plot(data_,alpha=.8)
    ax.set_ylabel("MoM pct change")
    ax.legend(cols_name)
    plt.xticks(rotation=60)
    st.pyplot(fg)
with t3:
    # params
    examplars=[("cb","fs"),("cb","ic"),("cl","ie"),("ng","fert")]
    # row 0
    st.subheader("Linregress")
    st.markdown("단선형회귀 수행. 변환을 하지 않아 의사회귀가 생김. 계열간 상관계수는 sampling rate가 높을수록 낮음.")
    q0,q1=st.columns(2)
    x0_=q0.selectbox("x",f.columns)
    y0_=q1.selectbox("y",f.columns)
    if not x0_==y0_:
        # preparing
        data=f[[x0_,y0_]].dropna()
        obs_num=round(len(data)*.85)
        x0y0=data.sample(obs_num)
        x0,y0=x0y0.iloc[:,0],x0y0.iloc[:,1]
        x1y1=data.sample(len(data)-obs_num)
        x1=x1y1.iloc[:,0]
        # get equatation
        slope,intercept,rval,pval,stderr=scipy.stats.linregress(x0,y0)
        y1_guess=[(slope*x)+intercept for x in x1]
        y1_answer=x1y1.iloc[:,1]
        # plot guesses
        fg,ax=plt.subplots(2,1,figsize=(8,8))
        ax[0].scatter(x0.values,y0.values,
            color="tomato",alpha=.8)
        ax[0].plot(x1,y1_guess,
            color="navy",alpha=1,linewidth=2)
        # plot residues
        residue=y1_guess-y1_answer
        residue_devi=np.std(residue)*.5
        ax[1].scatter(y1_guess,y1_guess-y1_answer,
            color="tomato",alpha=.8)
        ax[1].hlines(
            y=0,
            xmin=min(y1_guess)-residue_devi,
            xmax=max(y1_guess)+residue_devi,
            color="navy",alpha=1)
        # render
        linreg_params=list(zip(
            ("slope","intercept","r-value","p-value"),
            (slope,intercept,rval,pval,stderr)))
        for q in enumerate(st.columns(len(linreg_params))):
            q[1].metric(
                linreg_params[q[0]][0],
                f"{linreg_params[q[0]][1]:.3f}")
        st.pyplot(fg)
    else:
        st.error("Select x (enobs, causes), y (exobs, results)")
    # row 1
    # with st.container():
    #     st.subheader("Tests")
    #     st.markdown("This section is solely for testing purpose.")
    #     imgs={q.name:Image.open(f"asset/{q.name}") for q in os.scandir("asset") if q.name.endswith(".png") or q.name.endswith(".jpg")}
    #     if not len(imgs)==0:
    #         img_sel=st.radio("images",imgs.keys(),label_visibility="hidden")
    #         for img_label in imgs.keys():
    #             if img_label==img_sel:
    #                 st.image(imgs[img_label],)
    #     else:
    #         st.markdown(f"No imagefiles")
with t4:
    st.subheader("Citations")
    with open(f"asset/cite.txt",encoding="utf-8-sig") as citefile:
        sents=citefile.readlines()
    for q in sents:
        st.markdown(q)
        