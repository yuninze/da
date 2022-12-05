import streamlit as st
from PIL import Image
from da import *

print(os.getcwd())

# ornament
logo=Image.open("e:/dnld/kkk.jpg")
st.title("da::StreamLit Trial")
st.image(logo)

# control element example
# if st.checkbox(f"Show dropna-ed samples"):
#     st.subheader("Row samples")
#     st.write(f.dropna(thresh=20).sample(5))
# st.button("fuck0")
# st.selectbox("fuck1",
#     options=("選択肢1", "選択肢2", "選択肢3"))
# st.checkbox("checkbox")
# st.radio("radiobutton",
#     options=("選択肢1", "選択肢2", "選択肢3"))

@st.cache
def get_data(path="c:/code/f.csv"):
    return pd.read_csv(path,
        index_col="date",
        converters={"date":pd.to_datetime})

def main():
    features=["show samples","waratah","ng","relevance","rolling pct ch"]
    feature =st.sidebar.selectbox("features_listbox",features)
    st.header(f"selected_feature '{feature}'")
    if feature==features[0]:
        show("show samples")
    elif feature==features[1]:
        show("waratah")
    elif feature==features[2]:
        show("ng")
    elif feature==features[3]:
        show("relevance")
    elif feature==features[4]:
        show("roll")

def show(string):
    f=get_data()
    if string=="show samples":
        st.subheader("samples")
        st.dataframe(f.sample(5))
    elif string=="waratah":
        st.subheader("...go see the terminal")
        print(f"waratah selected")
    elif string=="ng":
        fg,ax=plt.subplots(2,figsize=(3,6))
        ax[0].hist(scipy.stats.yeojohnson(f.ng.dropna())[0],alpha=.6)
        ax[1].hist(np.log(f.ng.dropna()),alpha=.6)
        st.pyplot(fg)
    elif string=="relevance":
        fg,ax=plt.subplots(1,figsize=(5,5))
        freq  ="2d"
        f_cols=["by","cb","hs","cl","ie"]
        f0=f[f_cols]
        f1=f0.resample(freq).mean().diff().dropna()
        f2=f1.apply(lambda q:scipy.stats.zscore(scipy.stats.yeojohnson(q)[0]))
        hm(f2.corr(),ax=ax)
        st.pyplot(fg)
    elif string=="roll":
        fg,ax=plt.subplots(1,figsize=(8,8))
        roll_pct(f,"ng",ax=ax)
        st.pyplot(fg)

if __name__=="__main__":
    main()