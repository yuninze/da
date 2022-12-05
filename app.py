import streamlit as st
from da import *

st.title("da::StreamLit Trial")

state=st.text(f"Getting data")
f=pd.read_csv("c:/code/f.csv",
    index_col="date",
    converters={"date":pd.to_datetime})
state.text(f"Got")

st.subheader("Row Samples")
st.write(f.tail(10))

# if st.checkbox(f"Show dropna-ed samples"):
#     st.subheader("Row samples")
#     st.write(f.dropna(thresh=20).sample(5))

st.subheader(f"Plotting")
fg,ax=plt.subplots(1,1)
np.log(f.ng.dropna()).hist(alpha=.6,ax=ax)
ax.set_title(f"f::ng")
st.pyplot(fg)

