import numpy as np, time, sys
from lag import *
h0=float(sys.argv[1]); m=int(sys.argv[2]); EP=int(sys.argv[3])
zs=np.linspace(0.0,2.4,25)
t0=time.time()
F,V,D=build(h0,zs,C,nmax=3,keep=True)
print('built',time.time()-t0, 'UB',(m*zs+LAM*F.sum(0)).min(), zs[(m*zs+LAM*F.sum(0)).argmin()])
Fcum=np.concatenate([np.zeros((1,len(zs))),np.cumsum(F,0)])  # Fcum[r]=sum_{r'<r}F[r']
haz=lambda q: np.minimum(1.0,h0*np.exp(B*q))
rng=np.random.default_rng(1)
def gidx(mu):
    t=np.clip((mu-grid[0])/(grid[1]-grid[0]),0,G-1-1e-9); i=np.floor(t).astype(int); return i,t-i
def run_price():
    k=m; R=0.0; nint=0
    q=np.zeros(0); mu=np.zeros(0); n=np.zeros(0,int); d=np.zeros(0,int)
    for t in range(1,T+1):
        r=T-t
        na=rng.poisson(LAM)
        if na:
            qn=rng.standard_normal(na); sig=qn+1.5*rng.standard_normal(na)
            q=np.r_[q,qn]; mu=np.r_[mu,sig*(1/2.25)/prec0]; n=np.r_[n,np.zeros(na,int)]; d=np.r_[d,np.zeros(na,int)]
        if len(q)==0: continue
        def price():
            i,f=gidx(mu)
            Vp=V[r,d,n][:, :, :]  # (P,Z,G)
            ar=np.arange(len(q))
            Vz=Vp[ar,:,i]*(1-f)[:,None]+Vp[ar,:,i+1]*f[:,None]  # (P,Z)
            obj=k*zs+LAM*Fcum[r]+Vz.sum(0)
            iz=obj.argmin()
            dec=D[r,d,np.minimum(n,3),iz,np.rint(i+f).astype(int)]
            return iz,Vz[:,iz],dec
        # offers
        while k>0 and len(q)>0:
            iz,v,dec=price()
            cand=np.where(dec==1)[0]
            if len(cand)==0: break
            j=cand[np.argmax(v[cand])]
            acc=rng.random()<p_acc(d[j])
            if acc: R+=q[j]; k-=1
            keep=np.ones(len(q),bool); keep[j]=False
            q,mu,n,d=q[keep],mu[keep],n[keep],d[keep]
        if k==0: break
        if len(q)==0: continue
        iz,v,dec=price()
        cand=np.where((dec==2)&(n<3))[0]
        if len(cand)>K: cand=cand[np.argsort(-v[cand])[:K]]
        for j in cand:
            y=q[j]+rng.standard_normal()
            mu[j]=(mu[j]/s2(n[j])+y)*s2(n[j]+1); n[j]+=1; R-=C; nint+=1
        stay=rng.random(len(q))>=haz(q)
        q,mu,n,d=q[stay],mu[stay],n[stay],d[stay]+1
    return R,m-k,nint
t0=time.time()
res=np.array([run_price() for _ in range(EP)])
print(f'price-directed policy h0={h0} m={m}: mean reward {res[:,0].mean():.4f} +- {res[:,0].std()/np.sqrt(EP):.4f}; hires {res[:,1].mean():.2f}; interviews {res[:,2].mean():.2f}; {time.time()-t0:.0f}s')
# static resume threshold with linear fall in the last L days
def run_thr(th,L):
    k=m;R=0.0
    for t in range(1,T+1):
        na=rng.poisson(LAM)
        if not na: continue
        qn=rng.standard_normal(na); mu0=(qn+1.5*rng.standard_normal(na))*(1/2.25)/prec0
        bar=th*min(1.0,(T-t)/L) if L>0 else th
        for j in np.argsort(-mu0):
            if k>0 and mu0[j]>=bar and mu0[j]>0:
                if rng.random()<0.9: R+=qn[j]; k-=1
    return R
best=max(((np.mean([run_thr(th,L) for _ in range(3000)]),th,L) for th in (0.7,0.8,0.9,1.0,1.1,1.2) for L in (0,15,30,60)))
print('best resume-threshold rule (mean,theta,L):',best)
