import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
Psi=lambda u: norm.pdf(u)-u*norm.sf(u)
def resv(mu,s,c):
    if c>=s*Psi(-40): return None
    return mu+s*brentq(lambda u: Psi(u)-c/s,-40,40)
c=0.05
prec=1+1/2.25
for n in range(0,5):
    v=1/prec; v2=1/(prec+1); st=np.sqrt(v-v2)
    u=brentq(lambda u: Psi(u)-c/st,-40,40)
    print(n,'post var',round(v,4),'sd',round(np.sqrt(v),4),'stilde',round(st,4),'c/st',round(c/st,4),'u*',round(u,4),'z-mu',round(st*u,4), 'full-reveal z-mu', round(np.sqrt(v)*brentq(lambda u: Psi(u)-c/np.sqrt(v),-40,40),4))
    prec+=1
print('resume weight',(1/2.25)/(1+1/2.25),'sd of resume post mean',np.sqrt(1-1/(1+1/2.25)))
# tilt check
def tilt(mu,s2,d,h0,b=0.6):
    x,w=np.polynomial.hermite_e.hermegauss(80)
    q=mu+np.sqrt(s2)*x
    g=np.clip(1-np.minimum(1,h0*np.exp(b*q)),0,1)**d
    W=w*g
    m=(W*q).sum()/W.sum()
    return m, ((W*q*q).sum()/W.sum()-m*m)
for h0 in (0.01,0.03,0.05):
  for mu,s2,d in ((0.5,0.6923,5),(1.0,0.6923,10),(1.5,0.4091,10),(1.5,0.4091,30)):
    m,v=tilt(mu,s2,d,h0)
    ap=mu-d*h0*0.6*s2*np.exp(0.6*mu+0.18*s2)
    print(h0,mu,s2,d,'exact mean',round(m,4),'var',round(v,4),'first-order',round(ap,4))
# hindsight bound
rng=np.random.default_rng(0)
for m_,T,lam in ((3,60,1.5),(1,60,1.5),(1,12,0.6),(1,10,0.6)):
    tot=[]
    for _ in range(200000):
        N=rng.poisson(lam*T*0.9)
        if N==0: tot.append(0); continue
        q=np.sort(rng.standard_normal(N))[::-1][:m_]
        tot.append(np.maximum(q,0).sum())
    print('hindsight',m_,T,lam,np.mean(tot),np.std(tot)/np.sqrt(len(tot)))
