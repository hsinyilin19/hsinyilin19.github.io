import numpy as np
rng=np.random.default_rng(3)
n=60; L=10.; xs=np.linspace(0,L,n); X1,X2=np.meshgrid(xs,xs,indexing='ij')
A=np.array([2.5,5.]); B=np.array([7.5,5.]); R=1.2
inA=(X1-A[0])**2+(X2-A[1])**2<R**2; inB=(X1-B[0])**2+(X2-B[1])**2<R**2
T=300; steps=2400; dt=1.
def E_at(t):
    good = inA if (t//T)%2==0 else inB
    return np.where(good,1.0,-1.0)
ch={'s':dict(k=1.0,d=0.02,u=0.15),'l':dict(k=1.0,d=0.0012,u=0.01)}
M={'s':np.zeros((n,n)),'l':np.where(inA|inB,1.0,0.0)}
w={'s':0.6,'l':0.4}
X=A.copy()+0.3; path=[]; r=2.6; speed=0.06
for t in range(steps):
    E=E_at(t)
    K=np.clip(1-np.hypot(X1-X[0],X2-X[1])/r,0,1)
    for c,p in ch.items():
        m=M[c]; m+=dt*p['k']*(-p['d']*np.sign(m)*np.sqrt(np.abs(m))+p['u']*K*(E-m))
    H=w['s']*M['s']+w['l']*M['l']
    i=np.clip(int(round(X[0]/L*(n-1))),0,n-1); j=np.clip(int(round(X[1]/L*(n-1))),0,n-1)
    # desire to move: low if local environment is good
    desire = 0.0 if E[i,j]>0 else 1.0
    tgt_idx=np.unravel_index(np.argmax(H),H.shape); tgt=np.array([xs[tgt_idx[0]],xs[tgt_idx[1]]])
    dvec=tgt-X; dist=np.linalg.norm(dvec)
    step = desire*speed*dvec/dist if dist>1e-6 else 0
    X=np.clip(X+step+0.04*rng.standard_normal(2),0,L); path.append(X.copy())
path=np.array(path); np.save('path.npy',path)
# report visits
side=np.where(path[:,0]<5,'A','B'); 
print(''.join(side[::100]))
print(''.join(['A' if (t//T)%2==0 else 'B' for t in range(0,steps,100)]))
