"""Relation training with explicit overfit/gradient/collapse gates."""
import copy
import math
import random
from collections import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from .pair import SpanPairRelationModel,encode_marked,mark_pair,PairTooWide,relation_metrics,relation_gate

class RelationDataset(Dataset):
    def __init__(self,items,tokenizer,max_length=256):
        self.items=[];self.rejected=[]
        for x in items:
            try:
                marked=x.get('text') or mark_pair(x['raw_text'],*x['spans'])
                enc=encode_marked(tokenizer,marked,max_length)
                self.items.append({**enc,'labels':int(x['label']),'sample_weight':float(x.get('sample_weight',1.0)),
                                   'record_id':x['record_id']})
            except PairTooWide as exc:
                self.rejected.append({'record_id':x['record_id'],'reason':str(exc)})
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return dict(self.items[i])

def collator(tokenizer):
    def collate(rows):
        copied=[dict(x) for x in rows]
        labels=torch.tensor([r.pop('labels') for r in copied]);weights=torch.tensor([r.pop('sample_weight') for r in copied])
        for r in copied:r.pop('record_id',None)
        encoded=tokenizer.pad(copied,return_tensors='pt');encoded['labels']=labels;encoded['sample_weights']=weights
        return encoded
    return collate

@torch.no_grad()
def evaluate(model,ds,tokenizer,batch_size=8,threshold=.5):
    if not len(ds):return None
    ys=[];scores=[];device=next(model.parameters()).device;model.eval()
    for batch in DataLoader(ds,batch_size=batch_size,collate_fn=collator(tokenizer)):
        y=batch.pop('labels');batch.pop('sample_weights');logits=model(**{k:v.to(device) for k,v in batch.items()}).logits
        ys+=y.tolist();scores+=logits.softmax(-1)[:,1].cpu().tolist()
    return relation_metrics(ys,scores,threshold)

def optimizer_for(model,encoder_lr,head_lr):
    opt=torch.optim.AdamW([{'params':list(model.encoder.parameters()),'lr':encoder_lr},
                          {'params':list(model.classifier.parameters()),'lr':head_lr}],weight_decay=.01)
    ids={id(p) for group in opt.param_groups for p in group['params']}
    if any(p.requires_grad and id(p) not in ids for p in model.parameters()):raise RuntimeError('Trainable parameter missing from optimizer')
    return opt

def step_loss(model,batch,class_weights,device):
    y=batch.pop('labels').to(device);sw=batch.pop('sample_weights').to(device)
    logits=model(**{k:v.to(device) for k,v in batch.items()}).logits
    if logits.shape!=(len(y),2):raise RuntimeError('Expected raw logits [batch,2]')
    w=sw*class_weights[y]
    return (nn.functional.cross_entropy(logits,y,reduction='none')*w).sum()/w.sum().clamp_min(1e-8)

def overfit_gate(factory,ds,tokenizer,device,config):
    rng=random.Random(config.get('seed',42));ids=sorted({r['record_id'] for r in ds.items});rng.shuffle(ids)
    ids=set(ids[:100]); subset=copy.copy(ds);subset.items=[r for r in ds.items if r['record_id'] in ids]
    if len({r['labels'] for r in subset.items})<2:raise RuntimeError('Sanity subset lacks both relation labels')
    model=factory().to(device); opt=optimizer_for(model,config.get('sanity_encoder_lr',5e-5),config.get('sanity_head_lr',1e-3))
    loader=DataLoader(subset,batch_size=config.get('batch_size',4),shuffle=True,collate_fn=collator(tokenizer))
    steps=0;passed=False;grad_verified=False;best=0.
    max_steps=config.get('sanity_steps',300)
    while steps<max_steps:
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss=step_loss(model,batch,torch.ones(2,device=device),device);loss.backward()
            gradients=[p.grad for p in model.classifier.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients) or sum(float(g.norm()) for g in gradients)==0:
                raise RuntimeError('Relation classifier gradients missing, zero or nonfinite')
            grad_verified=True;torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();steps+=1
            if steps>=max_steps:break
        metrics=evaluate(model,subset,tokenizer,config.get('batch_size',4))
        best=max(best,metrics['macro_f1']);print({'sanity_steps':steps,'train_subset_macro_f1':metrics['macro_f1']},flush=True)
        if metrics['macro_f1']>.95:passed=True;break
    del model,opt
    if torch.cuda.is_available():torch.cuda.empty_cache()
    result={'passed':passed,'n_reviews':len(ids),'n_pairs':len(subset),'steps':steps,'best_macro_f1':best,'head_gradient_verified':grad_verified}
    if not passed:raise RuntimeError(f'Relation sanity-overfit failed; full training stopped: {result}')
    return result

def train_relation(factory,train_items,tune_items,tokenizer,device,config):
    train=RelationDataset(train_items,tokenizer,config.get('max_length',256));tune=RelationDataset(tune_items,tokenizer,config.get('max_length',256))
    audit={'train_labels':dict(Counter(r['labels'] for r in train.items)),'tune_labels':dict(Counter(r['labels'] for r in tune.items)),
           'train_pair_too_wide':train.rejected,'tune_pair_too_wide':tune.rejected}
    if tune.rejected:raise RuntimeError('Held-out relation pairs exceed the token budget. Increase relation_max_length; do not silently reduce evaluation denominator.')
    if any(audit['train_labels'].get(i,0)<config.get('min_per_class',20) for i in [0,1]) or any(audit['tune_labels'].get(i,0)<5 for i in [0,1]):
        return None,.5,None,[],{**audit,'status':'not_trained_insufficient_safe_supervision'}
    audit['sanity']=overfit_gate(factory,train,tokenizer,device,config)
    # Reset after overfit; the diagnostic subset receives no extra full-training exposure.
    torch.manual_seed(config.get('seed',42));model=factory().to(device)
    opt=optimizer_for(model,config.get('encoder_lr',2e-5),config.get('head_lr',3e-4))
    sums=[sum(r['sample_weight'] for r in train.items if r['labels']==i) for i in [0,1]]
    if min(sums)<=0:raise ValueError('Both classes need positive effective weight')
    cw=torch.tensor([sum(sums)/(2*s) for s in sums],device=device)
    dl=DataLoader(train,batch_size=config.get('batch_size',4),shuffle=True,collate_fn=collator(tokenizer))
    best_state=None;best=-1.;history=[];best_metrics=None;best_threshold=.5
    for epoch in range(1,config.get('epochs',5)+1):
        model.train();losses=[];accum=config.get('grad_accum',1);opt.zero_grad(set_to_none=True)
        for i,batch in enumerate(dl):
            group_start=(i//accum)*accum;group_size=min(accum,len(dl)-group_start)
            loss=step_loss(model,batch,cw,device);(loss/group_size).backward();losses.append(float(loss))
            if (i+1)%accum==0 or i+1==len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();opt.zero_grad(set_to_none=True)
        raw=evaluate(model,tune,tokenizer,config.get('batch_size',4),.5)
        history.append({'epoch':epoch,'train_loss':sum(losses)/len(losses),**{k:v for k,v in raw.items() if k not in {'labels','scores','predictions'}}})
        print(history[-1],flush=True)
        if raw['predicted_related_fraction']>.95 or raw['predicted_related_fraction']<.05:
            audit['status']='stopped_collapsed_checkpoint';break
        # Scores computed once; threshold search never reruns the encoder.
        candidates=[relation_metrics(raw['labels'],raw['scores'],i/100) for i in range(20,81,2)]
        eligible=[(m,i/100) for m,i in zip(candidates,range(20,81,2)) if relation_gate(m,config.get('gate_f1',.75),config.get('gate_recall',.70))]
        if eligible:
            metrics,threshold=max(eligible,key=lambda x:x[0]['composite'])
            if metrics['composite']>best:
                best=metrics['composite'];best_threshold=threshold;best_metrics=metrics
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is None:
        return None,.5,None,history,{**audit,'status':audit.get('status','failed_relation_gate')}
    model.load_state_dict(best_state);model.eval()
    return model,best_threshold,best_metrics,history,{**audit,'status':'trained_gate_passed'}
