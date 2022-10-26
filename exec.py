from da import *

def exec(i,local=False,intp=None,roll=1,update=True):
    f=getdata(local=local,roll=roll,update=update)
    ff=zs(f,intp=intp)
    fff=rng(ff,i)
    return f,ff,fff