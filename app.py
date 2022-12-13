import streamlit as st
from PIL import Image
from da import *
from datetime import datetime

# ornament
st.set_page_config(page_title="Product Snapshot",layout="centered")
print(f"{datetime.now()}::initialized in {os.getcwd()}")

# yielding
def get_data(path="c:/code/f.csv"):
    return pd.read_csv(path,index_col="date",
        converters={"date":pd.to_datetime})
f=get_data()

# head
st.header("Product Snapshot")
prods=["Indices","Commodities","Utilities","Images","Citations"]
t0,t1,t2,t3,t4=st.tabs(prods)

# tabs
with t0:
    # param
    cols=["by","iy","cb","ys","fr"]
    cols_nm=["U.S.10Y","U.S.10Y-I","C.B.Y.S.","L.S.Y.S.","F.F.R."]
    data={q:f[q].dropna().iloc[-2:] for q in cols}
    # row 0
    st.subheader("Latests")
    st.markdown("국채 일드, 기업채 일드 스프레드 등 적시성과 대표성이 높은 상품과 지표를 표시한다.")
    for q in enumerate(st.columns(len(cols))):
        q[1].metric(
            label=f"{cols_nm[q[0]]}",
            value=f"{data[cols[q[0]]][-1]:.3f}",
            delta=f"{data[cols[q[0]]].diff()[-1]:.3f} "
                    f"({data[cols[q[0]]].pct_change()[-1]*100:.2f}%)")
    st.subheader(f"Inflation-indexed U.S.10Y")
    st.markdown("인플레이션 반영 10년 국채 일드는 국채 10년물 수익률을 CPI, CPCE로 인덱싱한 것으로 명목 국채 일드가 된다.")
    fg,ax=plt.subplots(figsize=(10,6))
    data=f.iy.dropna().loc["2018":]
    ax.plot(data,alpha=.8,color="darkorange")
    ax.axhline(color="red",alpha=.5,y=data.iloc[-1])
    st.pyplot(fg)
    # row 1
    st.subheader(f"U.S.10Y+F.F.R.+Upcoming F.F.R.")
    st.markdown("10년 국채 일드의 시장가, 기준금리(담보대익일조달금리, SOFR), 차기 예상 기준금리이다. CMA 등 익일물 대출금리가 SOFR임을 즉시 알 수 있다.")
    fg,ax=plt.subplots(figsize=(10,6))
    data=f[["fr","by"]].loc["2018":].dropna()
    ax.plot(data,alpha=.8)
    ax.axhline(color="red",alpha=.5,y=data.iloc[-1,0]+.5)
    st.pyplot(fg)
    # row 2
    st.subheader(f"Long-Short Yield Spread")
    st.markdown("10년물 일드-3개월물 일드 차이(스프레드)와 기업채 스프레드 등 몇 가지 거시지표 및 금융상품 가격이다. 안전자산으로서 국채는 장기물 일드가 단기물 일드보다 높게 유지되지만, 장기 경제 컨센서스가 불량해지면 단기물 일드가 높아진다. 일드 역전에의 도상에 있는 것을 일드 커브 플래트닝, 반대의 경우를 스티프닝이라고 한다. 일드 커브 역전으로부터 보통 20분기 이내에 경기침체가 발생하는 등, 장단기 스프레드 흐름은 금리정책에 대한 시장의 반응을 그대로 대변하므로 경제 사이클 유추에 기업채 스프레드와 함께 더할나위 없는 지표로 된다.")
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
    st.markdown(f"U.S.10Y, 기업채 스프레드, H.S.I., W.T.I., 10년 인플레이션 기대율의 1차 변화량의 Pearson Coef이다.")
    freq="2d"
    f_cols=["by","cb","hs","cl","ie"]
    f0=f[f_cols]
    f1=f0.resample(freq).mean().diff().dropna()
    f2=f1.apply(lambda q:scipy.stats.zscore(scipy.stats.yeojohnson(q)[0]))
    fg,ax=plt.subplots(figsize=(6,6))
    hm(f2.corr(),ax=ax)
    st.pyplot(fg)
    st.text("end of page")
with t1:
    # param
    cols=["cl","ng","si","hg","zs"]
    cols_nm=["W.T.I.","Nat-Gas","Silver","Copper","Soybean"]
    # row 0
    st.subheader("Latests")
    st.markdown("주요 원자재의 가격, 계산된 명목 가격(deflated/nominal price)이 현재 상위 몇 %에 해당하는지 표시한다. 금융상품 가격은 로그정규분포를 하는 때가 많고 비정상성이 심하다. 일반적으로 시중 유동성 총량의 증가가 원자재 가격의 기저를 형성하는 경향이 있고, 산업금속은 21세기 들어 특정 국가의 경제 상황과 동기화되고 random-walking하므로 쉽게 mean-reverting하지 않는다. 매우 예외적으로 ADF를 1% 신뢰구간에서 통과하는 비정상적 가격 시계열의 상품도 있으며, deflating이 필요하지 않을지도 모른다.")
    ie=deflator(f)
    data={q:act(f[q],ie).dropna().iloc[-5:] for q in cols}
    for q in enumerate(st.columns(len(cols))):
        data_=data[cols[q[0]]]
        data_name=cols_nm[q[0]]
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
    st.subheader("x-day Rolled Standard Deviation")
    st.markdown("x일 변화율 평균의 표준편차의 1,2 표준점수(σ)를 보여준다. 여러 파라메터가 있는 만큼, 금융상품 가격과 같이 천변만화하는 숫자에 대한 평활법의 이론적 근거와 실효성은 없다.")
    dur=st.slider(
        "Duration (days)",
        min_value=5,
        max_value=200,
        value=5,
        step=5,)
    cols=["cl","ng","si","hg","zs"]
    for q in enumerate(cols):
        st.pyplot(roll_pct(f,q[1],dur=dur,figsize=(8,8),title=f"{cols_nm[q[0]]}"))
    st.text("end of page")
with t2:
    # params
    examplars=[("cb","fs"),("cb","ic"),("cl","ie"),("ng","fert")]
    # row 0
    st.subheader("Utilities: Linregress")
    st.markdown("단선형회귀를 수행한다. 이 대시보드에서 다루는 데이터는 전부 비정상 시계열이다. 여기서는 아무런 변환을 수행하지 않으므로 의사회귀가 발생한다. 금융상품 가격은 시중 유동성 총량에 비례하므로 원자재와 지수 선물 가격은 양의 상관관계를 가지는 듯하다. 안전상품인 국채 일드와 위험상품인 항셍 가격 사이에는 음의 상관관계가 있는 듯하다. 그러나 사실 각 계열 사이의 선형성은, 특히 sampling rate가 높을 수록 일관적이지 않다. 10YIE, CPI 등 유력한 지표의 범위를 바탕으로 grouping해 조건하 상관성을 톺아볼 수 있을 것이다.")
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
        # equatation
        slope,intercept,rval,pval,stderr=scipy.stats.linregress(x0,y0)
        y1_predicted=[(slope*x)+intercept for x in x1]
        y1_answer=x1y1.iloc[:,1]
        fg,ax=plt.subplots(2,1,figsize=(10,7))
        #expected values
        ax[0].scatter(x0.values,y0.values,
            color="tomato",alpha=.3)
        ax[0].plot(x1,y1_predicted,
            color="navy",alpha=1,linewidth=2)
        # residue
        residue=y1_predicted-y1_answer
        ax[1].scatter(y1_predicted,y1_predicted-y1_answer,
            color="tomato",alpha=.3)
        ax[1].hlines(
            y=0,
            xmin=min(y1_predicted)-(np.std(residue)*.5),
            xmax=max(y1_predicted)+(np.std(residue)*.5),
            color="navy",alpha=1)
        # rendering
        st.metric("Coef",value=f"{slope:.3f}")
        st.metric("r-value",value=f"{rval:.3f}")
        st.metric("p-value",value=f"{pval:.3f}")
        st.pyplot(fg)
    else:
        st.markdown("Select enobs, exobs")
    # row 1
    st.text("end of page")
with t3:
    st.subheader("Images")
    st.markdown("Selecting something reloads whole script.")
    imgs={q.name:Image.open(f"asset/{q.name}") for q in os.scandir("asset") if q.name.endswith(".png") or q.name.endswith(".jpg")}
    if not len(imgs)==0:
        img_sel=st.radio("images",imgs.keys(),label_visibility="hidden")
        for img_label in imgs.keys():
            if img_label==img_sel:
                st.image(imgs[img_label],)
    else:
        st.markdown(f"No imagefiles")
with t4:
    st.subheader("Citations")
    with open(f"asset/cite.txt",encoding="utf-8-sig") as citefile:
        sents=citefile.readlines()
    for q in sents:
        st.markdown(q)

