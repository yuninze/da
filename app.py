import streamlit as st
from da import *

st.title("da::StreamLit Trial")

@st.cache
def get_data(path="c:/code/f.csv"):
    return pd.read_csv(path,
        index_col="date",
        converters={"date":pd.to_datetime})

def main():
    st.title("trial")
    features_list=["show samples","fuck","ng"]
    features=st.sidebar.selectbox("features_selectbox",features_list)
    st.header(f"{features=} yeonsoup")
    if features==features_list[0]:
        show("show samples")
    elif features==features_list[1]:
        show("fuck")
    elif features==features_list[2]:
        show("ng")

def show(string):
    f=get_data()
    if string=="show samples":
        st.subheader("samples")
        st.dataframe(f.sample(5))
    elif string=="fuck":
        st.subheader("...go see the terminal")
        print(f"fuck selected")
    elif string=="ng":
        fg,ax=plt.subplots(1,1)
        np.log(f.ng.dropna()).hist(alpha=.6,ax=ax)
        st.pyplot(fg)

# if st.checkbox(f"Show dropna-ed samples"):
#     st.subheader("Row samples")
#     st.write(f.dropna(thresh=20).sample(5))
# st.button("fuck0")
# st.selectbox("fuck1",
#     options=("選択肢1", "選択肢2", "選択肢3"))
# st.checkbox("checkbox")
# st.radio("radiobutton",
#     options=("選択肢1", "選択肢2", "選択肢3"))

if __name__=="__main__":
    main()