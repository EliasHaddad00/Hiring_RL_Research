"""Hiring-as-a-race simulator, v3 (batched numpy). Written to verify an external review of the proposal.

Day order: arrivals -> offers -> interviews -> poaching. An interview result is usable from the next day.
Offers resolve the same day; a candidate who declines leaves. Days waited count from arrival.
Fill m roles in T days. q~N(0,1) hidden; resume = q+N(0,sig0^2); interview = q+N(0,sig^2), cost c, K slots/day.
Poaching: a waiting candidate leaves w.p. min(1,h0*exp(a*q)) per day. Acceptance: p0*exp(-b*days_waited).
Payoff = sum of hired q - c*interviews.

Generic rule: with n interviews done and posterior mean mu (closed-form Gaussian, ignores survival bias),
  offer if t>=S and mu >= o[n] - s_o[n]*t/T + r*hired ; interview if n<kmax and mu >= i[n] - s_i*t/T.
Families restrict this rule. All tuned on common random numbers (TUNE_E episodes), scored on EVAL_E fresh ones."""
import numpy as np, sys, itertools, time, json
from multiprocessing import Pool
T,m,lam,K,c,sig0,sig,a,p0=60,3,1.5,3,0.05,1.5,1.0,0.6,0.9
N=135; TUNE_E=600; EVAL_E=4000
def gen(E,seed):
    r=np.random.default_rng(seed); n=r.poisson(lam*T,E)
    d=dict(exists=np.arange(N)[None,:]<n[:,None],arr=r.integers(0,T,(E,N)),q=r.normal(size=(E,N)))
    d["resume"]=d["q"]+sig0*r.normal(size=(E,N)); d["inoise"]=sig*r.normal(size=(E,N,3))
    d["upoach"]=r.random((E,N,T),dtype=np.float32); d["uacc"]=r.random((E,N)); return d
def run(d,h0,b,kmax,o,so,i,si,r=0.0,S=0,oracle=False,stats=False):
    q=d["q"];E=q.shape[0];arr=d["arr"]
    alive=np.zeros((E,N),bool);prec=np.full((E,N),1+1/sig0**2);ssum=d["resume"]/sig0**2
    n=np.zeros((E,N),int);hired=np.zeros(E,int);val=np.zeros(E);nint=np.zeros(E);capdays=0
    o=np.asarray(o,float);so=np.asarray(so,float);i=np.asarray(i,float);ph=np.minimum(1,h0*np.exp(a*q))
    for t in range(T):
        alive|=d["exists"]&(arr==t)&(hired<m)[:,None]
        mu=q if oracle else ssum/prec; f=t/T
        if t>=S:
            elig=alive&(mu>=o[n]-so[n]*f+r*hired[:,None])
            if elig.any():
                idx=np.argsort(-np.where(elig,mu,-np.inf),axis=1)[:,:12]
                es=np.take_along_axis(elig,idx,1);p=p0*np.exp(-b*(t-np.take_along_axis(arr,idx,1)))
                acc=es&(np.take_along_axis(d["uacc"],idx,1)<p);cb=np.cumsum(acc,1)-acc
                off=es&(cb<(m-hired)[:,None]);h=off&acc
                val+=(np.take_along_axis(q,idx,1)*h).sum(1);hired+=h.sum(1)
                rows=np.nonzero(off);alive[rows[0],idx[rows]]=False
                alive&=(hired<m)[:,None]
        if kmax>0:
            want=alive&(n<kmax)&(mu>=i[np.minimum(n,len(i)-1)]-si*f)
            if want.any():
                pr=np.where(want,mu,-np.inf);idx=np.argpartition(-pr,K-1,axis=1)[:,:K]
                sel=np.take_along_axis(want,idx,1);rows=np.nonzero(sel);cols=idx[rows]
                e=rows[0];nn=n[e,cols]
                ssum[e,cols]+=(q[e,cols]+d["inoise"][e,cols,np.minimum(nn,2)])/sig**2;prec[e,cols]+=1/sig**2;n[e,cols]+=1
                np.add.at(nint,e,1);capdays+=(want.sum(1)>K).sum()
        alive&=~(d["upoach"][:,:,t]<ph)
    val-=c*nint
    return (val.mean(),val.std()/np.sqrt(E),nint.mean(),capdays/(E*T)) if stats else val.mean()
INF=99.0
FAM={ # name: (n_params, vector -> kwargs, init mean, init sd)
 "resume_falling_bar":(3,lambda v:dict(kmax=0,o=[v[0]],so=[v[1]],i=[INF],si=0,r=v[2]),[1.0,0.5,0.0],[0.6,0.6,0.3]),
 "one_interview":(7,lambda v:dict(kmax=1,o=[v[0],v[1]],so=[v[2],v[3]],i=[v[4]],si=v[5],r=v[6]),
                  [1.8,1.2,0.3,0.5,0.5,0.3,0.0],[0.7,0.6,0.5,0.5,0.6,0.5,0.3]),
 "up_to_two_interviews":(9,lambda v:dict(kmax=2,o=[v[0],v[1],v[2]],so=[v[3],v[4],v[4]],i=[v[5],v[6]],si=v[7],r=v[8]),
                  [1.8,1.4,1.2,0.3,0.5,0.5,0.8,0.3,0.0],[0.7,0.6,0.6,0.5,0.5,0.6,0.6,0.5,0.3]),
 "oracle_threshold":(3,lambda v:dict(kmax=0,o=[v[0]],so=[v[1]],i=[INF],si=0,r=v[2],oracle=True),[2.0,1.0,0.0],[0.7,0.7,0.4])}
def cem(d,h0,b,fam,seed,warm=None,pop=40,elite=8,iters=9):
    npar,f,mu0,sd0=FAM[fam];rng=np.random.default_rng(seed);mu=np.array(mu0,float);sd=np.array(sd0,float)
    best=(-1e9,None)
    for it in range(iters):
        X=mu+sd*rng.normal(size=(pop,npar))
        if it==0 and warm is not None: X[:len(warm)]=warm
        sc=np.array([run(d,h0,b,**f(x)) for x in X]);top=np.argsort(-sc)[:elite]
        if sc[top[0]]>best[0]: best=(sc[top[0]],X[top[0]].copy())
        mu=0.3*mu+0.7*X[top].mean(0);sd=0.3*sd+0.7*X[top].std(0)+0.02
    return best[1]
def setting(arg):
    h0,b=arg;d=gen(TUNE_E,1);out={}
    ev=[gen(1000,100+j) for j in range(EVAL_E//1000)]
    def score(**kw):
        r=[run(e,h0,b,stats=True,**kw) for e in ev];r=np.array(r)
        return dict(mean=round(r[:,0].mean(),3),se=round(np.sqrt((r[:,1]**2).sum())/len(r),3),interviews=round(r[:,2].mean(),1),cap_bind=round(r[:,3].mean(),4))
    # static threshold on resume estimate (offer on arrival)
    g=np.arange(0.2,2.01,0.1);tau=max(g,key=lambda x:run(d,h0,b,kmax=0,o=[x],so=[0],i=[INF],si=0))
    out["static_threshold"]=dict(par=[round(float(tau),2)],**score(kmax=0,o=[tau],so=[0],i=[INF],si=0))
    # patient: wait to the last day, no interviews, offer in rank order above a floor
    g=np.arange(-0.5,1.51,0.25);fl=max(g,key=lambda x:run(d,h0,b,kmax=0,o=[x],so=[0],i=[INF],si=0,S=T-1))
    out["patient_no_interview"]=dict(par=[float(fl)],**score(kmax=0,o=[fl],so=[0],i=[INF],si=0,S=T-1))
    # patient: interview (k times, those above an interview bar) all window, offer in the last 2 days
    g=list(itertools.product((1,2),(0.0,0.5,1.0,1.5),(0.5,1.0,1.5,2.0)))
    kw=lambda k,ib,fl:dict(kmax=k,o=[INF]*k+[fl],so=[0]*(k+1),i=[ib],si=0,S=T-2)
    bp=max(g,key=lambda p:run(d,h0,b,**kw(*p)));out["patient_interview"]=dict(par=list(bp),**score(**kw(*bp)))
    # as the first proposal built it: interview EVERYONE once, then offer from day S
    g=list(itertools.product((10,20,30,45,55),(0.0,0.5,1.0)))
    kw2=lambda S,fl:dict(kmax=1,o=[INF,fl],so=[0,0],i=[-INF],si=0,S=S)
    bp=max(g,key=lambda p:run(d,h0,b,**kw2(*p)));out["patient_interview_everyone(v2 style)"]=dict(par=list(bp),**score(**kw2(*bp)))
    warm1=np.array([[tau,INF,0,0,INF,0,0]]);warm2=None
    for fam in FAM:
        w={"one_interview":warm1,"up_to_two_interviews":warm2}.get(fam)
        x=cem(d,h0,b,fam,seed=5,warm=w);out[fam]=dict(par=[round(float(z),2) for z in x],**score(**FAM[fam][1](x)))
        if fam=="one_interview": warm2=np.array([[x[0],x[1],INF,x[2],x[3],x[4],INF,x[5],x[6]]])
    return (h0,b),out
if __name__=="__main__":
    S=[(0.0,0.0),(0.0,0.03),(0.02,0.03),(0.05,0.03),(0.10,0.03)]
    t0=time.time()
    with Pool(4) as p:
        for (h0,b),out in p.imap(setting,S):
            print(f"\n== h0={h0} b={b}  ({time.time()-t0:.0f}s)")
            for k,v in out.items(): print(f"  {k:38s} {v['mean']:.2f} ±{v['se']:.2f}  interviews/ep={v['interviews']:5.1f} cap_bind={v['cap_bind']:.3f} par={v['par']}")
            sys.stdout.flush()
