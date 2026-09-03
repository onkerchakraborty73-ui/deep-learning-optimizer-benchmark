
import argparse, csv, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class BasicBlock(nn.Module):
    def __init__(self, a, b, s=1):
        super().__init__()
        self.c1=nn.Conv2d(a,b,3,s,1,bias=False); self.n1=nn.BatchNorm2d(b)
        self.c2=nn.Conv2d(b,b,3,1,1,bias=False); self.n2=nn.BatchNorm2d(b)
        self.sc=nn.Sequential() if (s==1 and a==b) else nn.Sequential(nn.Conv2d(a,b,1,s,bias=False),nn.BatchNorm2d(b))
    def forward(self,x):
        return F.relu(self.n2(self.c2(F.relu(self.n1(self.c1(x)))))+self.sc(x))

class ResNet18(nn.Module):
    def __init__(self,c,k):
        super().__init__(); self.inp=64
        self.stem=nn.Sequential(nn.Conv2d(c,64,3,1,1,bias=False),nn.BatchNorm2d(64),nn.ReLU())
        self.l1=self.layer(64,2,1); self.l2=self.layer(128,2,2); self.l3=self.layer(256,2,2); self.l4=self.layer(512,2,2)
        self.fc=nn.Linear(512,k)
    def layer(self,p,n,s):
        z=[BasicBlock(self.inp,p,s)]; self.inp=p
        z += [BasicBlock(p,p) for _ in range(n-1)]
        return nn.Sequential(*z)
    def forward(self,x):
        x=self.stem(x); x=self.l1(x); x=self.l2(x); x=self.l3(x); x=self.l4(x)
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x,1),1))

class ViTTiny(nn.Module):
    def __init__(self,c,k,size=32,patch=4,d=192,depth=4,heads=3,mlp=384):
        super().__init__()
        self.patch=nn.Conv2d(c,d,patch,patch); n=(size//patch)**2
        self.cls=nn.Parameter(torch.zeros(1,1,d)); self.pos=nn.Parameter(torch.zeros(1,n+1,d))
        enc=nn.TransformerEncoderLayer(d,heads,mlp,dropout=0,batch_first=True,norm_first=True,activation="gelu")
        self.enc=nn.TransformerEncoder(enc,depth); self.norm=nn.LayerNorm(d); self.fc=nn.Linear(d,k)
        nn.init.trunc_normal_(self.cls,std=.02); nn.init.trunc_normal_(self.pos,std=.02)
    def forward(self,x):
        x=self.patch(x).flatten(2).transpose(1,2); x=torch.cat([self.cls.expand(x.size(0),-1,-1),x],1)+self.pos
        return self.fc(self.norm(self.enc(x)[:,0]))

class Lion(torch.optim.Optimizer):
    def __init__(self,p,lr=3e-4,betas=(.9,.99),weight_decay=1e-2):
        super().__init__(p,dict(lr=lr,betas=betas,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure:
            with torch.enable_grad(): loss=closure()
        for g in self.param_groups:
            b1,b2=g["betas"]; lr,wd=g["lr"],g["weight_decay"]
            for p in g["params"]:
                if p.grad is None: continue
                s=self.state[p]
                if not s: s["m"]=torch.zeros_like(p)
                m=s["m"]; u=m*b1+p.grad*(1-b1)
                p.mul_(1-lr*wd).add_(torch.sign(u),alpha=-lr)
                m.mul_(b2).add_(p.grad,alpha=1-b2)
        return loss

class ShampooLite(torch.optim.Optimizer):
    """Diagonal Shampoo-style adaptive preconditioning; use a full validated Shampoo package for publication claims."""
    def __init__(self,p,lr=3e-4,beta=.999,eps=1e-8,weight_decay=1e-2):
        super().__init__(p,dict(lr=lr,beta=beta,eps=eps,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure:
            with torch.enable_grad(): loss=closure()
        for g in self.param_groups:
            lr,beta,eps,wd=g["lr"],g["beta"],g["eps"],g["weight_decay"]
            for p in g["params"]:
                if p.grad is None: continue
                s=self.state[p]
                if not s: s["v"]=torch.zeros_like(p)
                s["v"].mul_(beta).addcmul_(p.grad,p.grad,value=1-beta)
                p.mul_(1-lr*wd).addcdiv_(p.grad,s["v"].sqrt().add(eps),value=-lr)
        return loss

def optimizer(name,model,lr,wd):
    if name=="adam": return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    if name=="lion": return Lion(model.parameters(),lr=lr,weight_decay=wd)
    if name=="shampoo": return ShampooLite(model.parameters(),lr=lr,weight_decay=wd)
    if name=="muon":
        if not hasattr(torch.optim,"Muon"): raise RuntimeError("torch.optim.Muon unavailable")
        mu,other=[],[]
        for n,p in model.named_parameters():
            (mu if p.ndim==2 and "norm" not in n.lower() and "bias" not in n.lower() else other).append(p)
        groups=[]
        if mu: groups.append({"params":mu,"use_muon":True})
        if other: groups.append({"params":other,"use_muon":False})
        return torch.optim.Muon(groups,lr=lr,weight_decay=wd,momentum=.95,nesterov=True)

def loaders(ds,root,seed,batch):
    if ds=="mnist":
        tr=transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((.1307,),(.3081,))])
        base=datasets.MNIST(root,True,download=True,transform=tr); test=datasets.MNIST(root,False,download=True,transform=tr); c,k=1,10
    else:
        tr=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.247,.2435,.2616))])
        ev=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.247,.2435,.2616))])
        base=datasets.CIFAR10(root,True,download=True,transform=tr); test=datasets.CIFAR10(root,False,download=True,transform=ev); c,k=3,10
    n=len(base); a=int(.9*n); gen=torch.Generator().manual_seed(seed)
    train,val=random_split(base,[a,n-a],generator=gen)
    pin=True
    return (DataLoader(train,batch_size=batch,shuffle=True,num_workers=2,pin_memory=pin,persistent_workers=True),
            DataLoader(val,batch_size=batch*2,shuffle=False,num_workers=2,pin_memory=pin,persistent_workers=True),
            DataLoader(test,batch_size=batch*2,shuffle=False,num_workers=2,pin_memory=pin,persistent_workers=True),c,k)

@torch.no_grad()
def evaluate(model,loader,amp):
    model.eval(); ls=co=cnt=0
    for x,y in loader:
        x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
        with torch.autocast("cuda",dtype=torch.float16,enabled=amp):
            z=model(x); loss=F.cross_entropy(z,y)
        ls+=loss.item()*y.size(0); co+=(z.argmax(1)==y).sum().item(); cnt+=y.size(0)
    return ls/cnt,co/cnt

def run(a,ds,mn,on,seed):
    seed_all(seed); batch=a.vit_batch if mn=="vit" else a.resnet_batch
    tr,va,te,c,k=loaders(ds,a.data,seed,batch)
    model=(ResNet18(c,k) if mn=="resnet" else ViTTiny(c,k)).cuda()
    opt=optimizer(on,model,a.lr,a.wd); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
    amp=not a.no_amp; scaler=torch.amp.GradScaler("cuda",enabled=amp); out=Path(a.out)/ds/mn/on/f"seed_{seed}"; out.mkdir(parents=True,exist_ok=True)
    hist=[]; best=-1; t0=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for ep in range(1,a.epochs+1):
        model.train(); es=time.perf_counter(); total=correct=n=0; gns=[]
        for x,y in tr:
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=amp): z=model(x); loss=F.cross_entropy(z,y)
            if amp:
                scaler.scale(loss).backward(); scaler.unscale_(opt); gn=math.sqrt(sum((p.grad.detach().float()**2).sum().item() for p in model.parameters() if p.grad is not None)); nn.utils.clip_grad_norm_(model.parameters(),a.clip); scaler.step(opt); scaler.update()
            else:
                loss.backward(); gn=math.sqrt(sum((p.grad.detach().float()**2).sum().item() for p in model.parameters() if p.grad is not None)); nn.utils.clip_grad_norm_(model.parameters(),a.clip); opt.step()
            gns.append(gn); total+=loss.item()*y.size(0); correct+=(z.argmax(1)==y).sum().item(); n+=y.size(0)
        sch.step(); torch.cuda.synchronize(); vl,va=evaluate(model,va,amp); et=time.perf_counter()-es
        vram=torch.cuda.max_memory_allocated()/1024**3
        row={"epoch":ep,"train_loss":total/n,"train_acc":correct/n,"val_loss":vl,"val_acc":va,"lr":opt.param_groups[0]["lr"],"epoch_time_sec":et,"mean_grad_norm":float(np.mean(gns)),"peak_vram_gb":vram}; hist.append(row)
        print(f"[{ds}|{mn}|{on}|seed={seed}] ep {ep}/{a.epochs} train={correct/n:.4f} val={va:.4f} time={et:.1f}s VRAM={vram:.2f}GB")
        if va>best: best=va; torch.save(model.state_dict(),out/"best.pt")
    torch.cuda.synchronize(); total_time=time.perf_counter()-t0; tl,ta=evaluate(model,te,amp)
    result={"dataset":ds,"model":mn,"optimizer":on,"seed":seed,"test_loss":tl,"test_acc":ta,"best_val_acc":best,"epochs":a.epochs,"amp":amp,"time_sec":total_time,"peak_vram_gb":torch.cuda.max_memory_allocated()/1024**3,"params":sum(p.numel() for p in model.parameters()),"lr":a.lr,"weight_decay":a.wd,"batch_size":batch}
    with open(out/"history.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
    (out/"final.json").write_text(json.dumps(result,indent=2)); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--datasets",nargs="+",default=["mnist","cifar10"]); p.add_argument("--models",nargs="+",choices=["resnet","vit"],default=["resnet","vit"]); p.add_argument("--optimizers",nargs="+",choices=["adam","lion","shampoo","muon"],default=["adam","lion","shampoo","muon"]); p.add_argument("--seeds",nargs="+",type=int,default=[0]); p.add_argument("--epochs",type=int,default=2); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--wd",type=float,default=1e-2); p.add_argument("--resnet-batch",type=int,default=128); p.add_argument("--vit-batch",type=int,default=32); p.add_argument("--clip",type=float,default=1.0); p.add_argument("--data",default="./data"); p.add_argument("--out",default="./results"); p.add_argument("--no-amp",action="store_true"); p.add_argument("--smoke-test",action="store_true")
    a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA is not available.")
    if a.smoke_test: a.datasets=["mnist"]; a.models=["resnet"]; a.optimizers=["adam"]; a.seeds=[0]; a.epochs=1
    print("PyTorch:",torch.__version__," CUDA:",torch.version.cuda," GPU:",torch.cuda.get_device_name(0)," VRAM:",round(torch.cuda.get_device_properties(0).total_memory/1024**3,2),"GB")
    results=[]
    for ds in a.datasets:
        for mn in a.models:
            for on in a.optimizers:
                for s in a.seeds:
                    try: results.append(run(a,ds,mn,on,s))
                    except Exception as e: print("FAILED",ds,mn,on,s,repr(e))
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out)/"summary.json").write_text(json.dumps(results,indent=2))
    if results:
        with open(Path(a.out)/"summary.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print("DONE:",Path(a.out).resolve())

if __name__=="__main__": main()
