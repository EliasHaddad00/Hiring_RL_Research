import numpy as np, time, sys, itertools
from scipy.stats import norm
from functools import lru_cache
T=int(sys.argv[1]); P=int(sys.argv[2]); h0=float(sys.argv[3]); NI=2
pa=0.6; c=0.05; Bq=0.6
rcuts=[-0.8,0.8]; icuts=[0.0,1.0]; R=len(rcuts)+1; Bn=len(icuts)+1
qg=np.linspace(-8,8,4001); w=norm.pdf(qg); w/=w.sum()
def binp(cuts,sd):
    e=[-np.inf]+cuts+[np.inf]
    return np.array([norm.cdf((e[i+1]-qg)/sd)-norm.cdf((e[i]-qg)/sd) for i in range(len(e)-1)])
Lr=binp(rcuts,1.5); Li=binp(icuts,1.0)
surv=1-np.minimum(1,h0*np.exp(Bq*qg))
# types: (r, counts tuple)
counts=[cnt for cnt in itertools.product(range(NI+1),repeat=Bn) if sum(cnt)<=NI]
types=[(r,cnt) for r in range(R) for cnt in counts]; tid={t:i for i,t in enumerate(types)}
NT=len(types); DM=T+1
Z=np.zeros((NT,DM+1)); M=np.zeros((NT,DM+1))
for i,(r,cnt) in enumerate(types):
    L=w*Lr[r]*np.prod([Li[b]**cnt[b] for b in range(Bn)],0)
    for d in range(DM+1):
        ww=L*surv**d; Z[i,d]=ww.sum(); M[i,d]=(ww*qg).sum()/ww.sum()
nxt=-np.ones((NT,Bn),int)
for i,(r,cnt) in enumerate(types):
    if sum(cnt)<NI:
        for b in range(Bn):
            c2=list(cnt); c2[b]+=1; nxt[i,b]=tid[(r,tuple(c2))]
Pr=np.array([Z[tid[(r,(0,)*Bn)],0] for r in range(R)])
S=Z[:,1:]/Z[:,:-1]                       # survival prob given (type,d)
Pb=np.where(nxt[:,:,None]>=0, Z[np.maximum(nxt,0),:]/Z[:,None,:],0.0)  # (NT,Bn,d)
pacc=lambda d: 0.9*np.exp(-0.03*d)
S=S.tolist(); Mm=M.tolist(); Pbl=Pb.tolist(); nx=nxt.tolist(); Prl=Pr.tolist()
import sys; sys.setrecursionlimit(100000)
memo={}
def W(t,pool):            # start of day t, before arrival
    if t>T: return 0.0
    key=(t,pool)
    v=memo.get(key)
    if v is not None: return v
    v=(1-pa)*U(t,pool)
    for r in range(R):
        new=tuple(sorted(pool+((r*len(counts),0),)))
        if len(new)>P: best=max(U(t,new[:j]+new[j+1:]) for j in range(len(new)))
        else: best=U(t,new)
        v+=pa*Prl[r]*best
    memo[key]=v; return v
memoU={}
def U(t,pool):            # offer stage (sequential offers)
    key=(t,pool); v=memoU.get(key)
    if v is not None: return v
    v=I(t,pool)
    for j,(ty,d) in enumerate(pool):
        p=pacc(d); v=max(v,p*Mm[ty][d]+(1-p)*U(t,pool[:j]+pool[j+1:]))
    memoU[key]=v; return v
memoI={}
def I(t,pool):            # interview stage (K=1)
    key=(t,pool); v=memoI.get(key)
    if v is not None: return v
    v=Pc(t,pool)
    if t<T:
        for j,(ty,d) in enumerate(pool):
            if nx[ty][0]>=0:
                e=-c
                for b in range(Bn):
                    e+=Pbl[ty][b][d]*Pc(t,tuple(sorted(pool[:j]+((nx[ty][b],d),)+pool[j+1:])))
                v=max(v,e)
    memoI[key]=v; return v
memoP={}
def Pc(t,pool):           # poaching, then next day
    key=(t,pool); v=memoP.get(key)
    if v is not None: return v
    if t>=T: v=0.0
    else:
        v=0.0
        for mask in itertools.product((0,1),repeat=len(pool)):
            pr=1.0; new=[]
            for keep,(ty,d) in zip(mask,pool):
                s=S[ty][d]
                if keep: pr*=s; new.append((ty,d+1))
                else: pr*=1-s
            if pr>0: v+=pr*W(t+1,tuple(new))
    memoP[key]=v; return v
t0=time.time(); v=W(1,())
print(f'T={T} P={P} h0={h0}: types/candidate={NT}; optimal value={v:.5f}; states W={len(memo)} U={len(memoU)} I={len(memoI)} Pc={len(memoP)}; {time.time()-t0:.1f}s')
# value with no interviews allowed (c=inf) and hindsight for reference
if len(sys.argv)>4:
    # Monte Carlo check of DP policy
    rng=np.random.default_rng(0); tot=[]
    def argU(t,pool):
        best=I(t,pool); a=None
        for j,(ty,d) in enumerate(pool):
            p=pacc(d); e=p*Mm[ty][d]+(1-p)*U(t,pool[:j]+pool[j+1:])
            if e>best+1e-12: best=e;a=j
        return a
    def argI(t,pool):
        best=Pc(t,pool); a=None
        if t<T:
            for j,(ty,d) in enumerate(pool):
                if nx[ty][0]>=0:
                    e=-c+sum(Pbl[ty][b][d]*Pc(t,tuple(sorted(pool[:j]+((nx[ty][b],d),)+pool[j+1:]))) for b in range(Bn))
                    if e>best+1e-12: best=e;a=j
        return a
    for ep in range(int(sys.argv[4])):
        pool=[]; Rw=0.0; done=False   # entries [type,d,q]
        for t in range(1,T+1):
            if rng.random()<pa:
                q=rng.standard_normal(); r=int(np.searchsorted(rcuts,q+1.5*rng.standard_normal()))
                pool.append([r*len(counts),0,q]); pool.sort(key=lambda x:(x[0],x[1]))
                if len(pool)>P:
                    key=[(x[0],x[1]) for x in pool]
                    j=max(range(len(pool)),key=lambda j:U(t,tuple(key[:j]+key[j+1:]))); pool.pop(j)
            while True:
                key=tuple((x[0],x[1]) for x in pool); a=argU(t,key)
                if a is None: break
                x=pool.pop(a)
                if rng.random()<pacc(x[1]): Rw+=x[2]; done=True; break
            if done: break
            key=tuple((x[0],x[1]) for x in pool); a=argI(t,key)
            if a is not None:
                x=pool[a]; b=int(np.searchsorted(icuts,x[2]+rng.standard_normal())); x[0]=nx[x[0]][b]; Rw-=c
            pool=[[x[0],x[1]+1,x[2]] for x in pool if rng.random()>=min(1,h0*np.exp(Bq*x[2]))]
            pool.sort(key=lambda x:(x[0],x[1]))
        tot.append(Rw)
    print('MC of DP policy:',np.mean(tot),'+-',np.std(tot)/np.sqrt(len(tot)))
