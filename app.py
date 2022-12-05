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
    features=st.sidebar.selectbox("")
    st.write(f.tail(5))
    st.subheader(f"Fuck1")
    fg,ax=plt.subplots(1,1)
    np.log(f.ng.dropna()).hist(alpha=.6,ax=ax)
    ax.set_title(f"f::ng")
    st.pyplot(fg)

def show():
    f=get_data()
    if st.sidebar.checkbox("show samples"):
        st.subheader("samples")
        st.dataframe(f.sample(10))
    if st.sidebar.checkbox("fuck"):
        print(f"fuck selected")


# if st.checkbox(f"Show dropna-ed samples"):
#     st.subheader("Row samples")
#     st.write(f.dropna(thresh=20).sample(5))
# st.button("fuck0")
# st.selectbox("fuck1",
#     options=("選択肢1", "選択肢2", "選択肢3"))
# st.checkbox("checkbox")
# st.radio("radiobutton",
#     options=("選択肢1", "選択肢2", "選択肢3"))

