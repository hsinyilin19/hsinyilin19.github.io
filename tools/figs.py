import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import linear_sum_assignment
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'cm',
  'font.size':10,'axes.edgecolor':'#4a4560','axes.labelcolor':'#2a2640','xtick.color':'#4a4560','ytick.color':'#4a4560',
  'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220,'savefig.bbox':'tight','savefig.pad_inches':0.08})
INK='#221d3a'; VIO='#5b2a86'; AMB='#e39b2d'; cm=plt.cm.plasma
out='../assets/fig/'
import os; os.makedirs(out,exist_ok=True)
rng=np.random.default_rng(7)

# A. averaging lemma via free transport: f(t,x,v)=h(x-vt), rho(t,x)=int f phi dv = h * phi_t
nx,nv=1024,400; x=np.linspace(0,2*np.pi,nx,endpoint=False); v=np.linspace(-1,1,nv)
k=np.fft.fftfreq(nx,1/nx); hh=np.fft.fft(rng.standard_normal(nx))*np.exp(-(k/70)**2)
h=np.real(np.fft.ifft(hh)); h/=h.std()
phi=np.where(np.abs(v)<1,np.exp(-1/np.maximum(1-v**2,1e-12)),0); phi/=np.trapezoid(phi,v)
t1=0.9
f=np.real(np.fft.ifft(hh[:,None]*np.exp(-1j*k[:,None]*v[None,:]*t1),axis=0))/np.real(np.fft.ifft(hh)).std()
rho=np.trapezoid(f*phi[None,:],v,axis=1)
fig,ax=plt.subplots(1,2,figsize=(8.4,3.0),gridspec_kw={'width_ratios':[1.15,1]})
ax[0].imshow(f.T,origin='lower',extent=[0,2*np.pi,-1,1],aspect='auto',cmap='plasma',vmin=-2.4,vmax=2.4,interpolation='antialiased')
ax[0].set_xlabel(r'$x\in\mathbb{T}$'); ax[0].set_ylabel(r'$v$'); ax[0].set_title(r'$f(t,x,v)=f_0(x-vt)$: still rough',fontsize=10,color=INK)
ax[0].set_xticks([0,np.pi,2*np.pi]); ax[0].set_xticklabels(['0',r'$\pi$',r'$2\pi$'])
ax[1].plot(x,h,color=AMB,lw=0.6,alpha=0.85,label=r'$\rho_\varphi(0,x)$')
ax[1].plot(x,rho,color=VIO,lw=1.8,label=r'$\rho_\varphi(t,x)=\int f\,\varphi\,dv$')
ax[1].set_xticks([0,np.pi,2*np.pi]); ax[1].set_xticklabels(['0',r'$\pi$',r'$2\pi$']); ax[1].set_xlim(0,2*np.pi)
ax[1].set_yticks([]); ax[1].spines['left'].set_visible(False)
ax[1].set_title('velocity average is smooth',fontsize=10,color=INK); ax[1].legend(frameon=False,fontsize=8.5,loc='upper right',ncol=1)
ax[1].set_ylim(-4.2,5.2)
fig.savefig(out+'averaging.png'); plt.close(fig)

# B. unmixing on torus: random Fourier stream function
N=256; t=np.linspace(0,2*np.pi,N,endpoint=False); X,Y=np.meshgrid(t,t)
psi=np.zeros_like(X)
for kx in range(-4,5):
  for ky in range(-4,5):
    if kx==0 and ky==0: continue
    a=rng.standard_normal()/(kx*kx+ky*ky)**1.1; ph=rng.uniform(0,2*np.pi)
    psi+=a*np.cos(kx*X+ky*Y+ph)
psi/=np.abs(psi).max()
u=np.gradient(psi,t,axis=0); w=-np.gradient(psi,t,axis=1)   # a = grad^perp psi = (d_y psi, -d_x psi)
fig,ax=plt.subplots(figsize=(4.4,4.4))
ax.contourf(X,Y,psi,levels=24,cmap='plasma',alpha=0.92)
ax.contour(X,Y,psi,levels=24,colors='white',linewidths=0.35,alpha=0.6)
ax.streamplot(t,t,u,w,color=(1,1,1,0.55),linewidth=0.5,density=1.3,arrowsize=0.5)
from scipy.ndimage import maximum_filter, minimum_filter
for fil,mk in [(maximum_filter,'^'),(minimum_filter,'v')]:
  m=(fil(psi,size=41,mode='wrap')==psi)
  ii,jj=np.where(m); ax.scatter(X[ii,jj],Y[ii,jj],marker=mk,s=34,c='white',edgecolors=INK,linewidths=0.8,zorder=5)
ax.set_xticks([0,np.pi,2*np.pi]); ax.set_xticklabels(['0',r'$\pi$',r'$2\pi$']); ax.set_yticks([0,np.pi,2*np.pi]); ax.set_yticklabels(['0',r'$\pi$',r'$2\pi$'])
ax.set_aspect('equal'); ax.set_xlim(0,2*np.pi); ax.set_ylim(0,2*np.pi)
for s_ in ax.spines.values(): s_.set_visible(False)
fig.savefig(out+'torus.png'); plt.close(fig)


# C. migration
path=np.load('path.npy'); T=300; steps=len(path)
fig,ax=plt.subplots(1,2,figsize=(8.4,3.0),gridspec_kw={'width_ratios':[1,1.7]})
for c,lab in [((2.5,5),'A'),((7.5,5),'B')]:
  ax[0].add_patch(plt.Circle(c,1.2,fc='#efe9f7',ec=VIO,lw=0.9)); ax[0].text(c[0],c[1]+1.55,lab,ha='center',color=VIO,fontsize=11)
seg=np.stack([path[:-1],path[1:]],1); lc=LineCollection(seg,cmap='plasma',lw=0.7,alpha=0.9); lc.set_array(np.arange(steps-1)); ax[0].add_collection(lc)
ax[0].set_xlim(0.5,9.5); ax[0].set_ylim(2,8.5); ax[0].set_aspect('equal'); ax[0].set_xticks([]); ax[0].set_yticks([])
for s_ in ax[0].spines.values(): s_.set_visible(False)
ax[0].set_title('trajectory $X(t)$',fontsize=10,color=INK)
tt=np.arange(steps)
for p in range(0,steps,T):
  if (p//T)%2==0: ax[1].axvspan(p,p+T,color='#efe9f7',lw=0)
pts=np.stack([tt,path[:,0]],1); seg2=np.stack([pts[:-1],pts[1:]],1); lc2=LineCollection(seg2,cmap='plasma',lw=1.1); lc2.set_array(tt[:-1]); ax[1].add_collection(lc2)
ax[1].set_xlim(0,steps); ax[1].set_ylim(0.5,9.5); ax[1].set_yticks([2.5,7.5]); ax[1].set_yticklabels(['A','B'])
ax[1].set_xlabel('time  (shaded: resources in A)'); ax[1].set_title('periodic migration emerges from two memory channels',fontsize=10,color=INK)
fig.savefig(out+'migration.png'); plt.close(fig)

# D. optimal transport in 1D: monotone rearrangement T = F_nu^{-1} o F_mu
z=np.linspace(-4,4,2000)
mu=0.55*np.exp(-(z+1.2)**2/0.9)+0.45*np.exp(-(z-1.4)**2/1.6); mu/=np.trapezoid(mu,z)
nu=0.5*np.exp(-(z+0.9)**2/0.12)+0.5*np.exp(-(z-0.9)**2/0.12); nu/=np.trapezoid(nu,z)
Fm=np.cumsum(mu)*(z[1]-z[0]); Fn=np.cumsum(nu)*(z[1]-z[0])
qs=np.linspace(0.02,0.98,44); xm=np.interp(qs,Fm,z); xn=np.interp(qs,Fn,z)
fig,ax=plt.subplots(figsize=(6.4,3.0))
top,bot=1.0,0.0; sc=0.32
ax.fill_between(z,top,top+sc*mu/mu.max(),color=AMB,alpha=0.25,lw=0); ax.plot(z,top+sc*mu/mu.max(),color=AMB,lw=1.4)
ax.fill_between(z,bot,bot-sc*nu/nu.max(),color=VIO,alpha=0.18,lw=0); ax.plot(z,bot-sc*nu/nu.max(),color=VIO,lw=1.4)
for i,(a,b) in enumerate(zip(xm,xn)): ax.plot([a,b],[top,bot],color=cm(0.1+0.75*i/len(qs)),lw=0.8,alpha=0.8)
ax.text(-4,top+0.12,r'noisy $\mu$',color=AMB,fontsize=10); ax.text(-4,bot-0.22,r'clean $\nu$',color=VIO,fontsize=10)
ax.text(3.95,0.5,r'$T_\#\mu=\nu$',ha='right',color=INK,fontsize=11)
ax.set_xlim(-4.1,4.1); ax.axis('off')
fig.savefig(out+'ot.png'); plt.close(fig)
print('ok')
