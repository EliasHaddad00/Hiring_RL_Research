import numpy as np, time, sys
T=60; LAM=1.5; K=3; C=0.05; B=0.6; NMAX=4
G=91; grid=np.linspace(-3,4,G)
xq,wq=np.polynomial.hermite_e.hermegauss(15); wq=wq/wq.sum()
xe,we=np.polynomial.hermite_e.hermegauss(9); we=we/we.sum()
prec0=1+1/2.25
s2=lambda n: 1/(prec0+n)
p_acc=lambda d: 0.9*np.exp(-0.03*d)
def build(h0, zs, ceff, nmax=NMAX, keep=False):
    Z=len(zs); zs=np.asarray(zs)[:,None]
    haz=lambda q: np.minimum(1.0,h0*np.exp(B*q))
    # per n precompute
    Q=[grid[:,None]+np.sqrt(s2(n))*xq[None,:] for n in range(nmax+1)]   # (G,J)
    idx=[];fr=[]
    for n in range(nmax):
        y=Q[n][:,:,None]+xe[None,None,:]
        mu2=(grid[:,None,None]/s2(n)+y)*s2(n+1)
        t=np.clip((mu2-grid[0])/(grid[1]-grid[0]),0,G-1-1e-9)
        i=np.floor(t).astype(int); idx.append(i); fr.append(t-i)
    V=np.zeros((T,T+1,nmax+2,Z,G),dtype=np.float32) if keep else None
    D=np.zeros((T,T+1,nmax+1,Z,G),dtype=np.int8) if keep else None
    prev={}  # (d,n)->(Z,G) for r-1
    out=[]
    for r in range(T):
        cur={}
        for d in range(0,T-r):
            for n in range(0,min(d,nmax)+1):
                q=Q[n]; tw=wq[None,:]*(1-haz(q))**d
                pi=tw/tw.sum(1,keepdims=True)
                M=(pi*q).sum(1)
                offer=p_acc(d)*(M[None,:]-zs)           # (Z,G)
                best=np.maximum(offer,0); dec=(offer>0).astype(np.int8)
                if r>=1 and n<nmax:
                    Vn=prev[(d+1,n+1)]
                    i=idx[n]; f=fr[n]
                    val=Vn[:,i]*(1-f)+Vn[:,i+1]*f        # (Z,G,J,Kk)
                    cont=-ceff+np.einsum('gj,zgjk,k->zg',pi*(1-haz(q)),val,we)
                    dec=np.where(cont>best,2,dec); best=np.maximum(best,cont)
                cur[(d,n)]=best
                if keep: V[r,d,n]=best; D[r,d,n]=dec
        prev=cur
        out.append(cur[(0,0)])
    # expected value of a fresh arrival with r days remaining
    xm,wm=np.polynomial.hermite_e.hermegauss(41); wm=wm/wm.sum()
    mu0=np.sqrt(1-s2(0))*xm
    F=np.array([[np.interp(mu0,grid,o[z])@wm for z in range(Z)] for o in out])  # (T,Z): r=0..T-1
    return F,V,D
if __name__=='__main__':
    zs=np.linspace(0.6,2.6,41)
    for h0 in (0.0,0.01,0.02,0.05):
        for lam in (0.0,0.02):
            t0=time.time()
            F,_,_=build(h0,zs,C+lam)
            for m in (3,1):
                ub=m*zs+LAM*F.sum(0)+lam*K*T
                j=ub.argmin()
                print(f'h0={h0} lam={lam} m={m}: UB={ub[j]:.4f} at z={zs[j]:.3f}  ({time.time()-t0:.1f}s)')
    # check nmax sensitivity
    for nm in (3,4,6):
        F,_,_=build(0.02,zs,C,nmax=nm); ub=3*zs+LAM*F.sum(0); print('nmax',nm,ub.min(),zs[ub.argmin()])
