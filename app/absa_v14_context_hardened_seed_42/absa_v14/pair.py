"""Marker-aware token windows and explicit aspect/opinion pooled relation scorer."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

MARKERS = ['[ASP]', '[/ASP]', '[OPN]', '[/OPN]']
class PairTooWide(ValueError): pass

def mark_pair(text, a_s, a_e, o_s=None, o_e=None):
    text=str(text)
    spans=[(int(a_s),int(a_e),'[ASP]','[/ASP]')]
    if o_s is not None and o_e is not None: spans.append((int(o_s),int(o_e),'[OPN]','[/OPN]'))
    for s,e,_,_ in spans:
        if not 0<=s<e<=len(text): raise ValueError('Invalid pair offsets')
    if len(spans)==2 and max(spans[0][0],spans[1][0])<min(spans[0][1],spans[1][1]):
        raise ValueError('Overlapping aspect/opinion cannot be represented by nested marker pooling')
    # Literal special markers in user text would make the target ambiguous.
    if any(marker in text for marker in MARKERS): raise ValueError('Input contains reserved ABSA markers')
    for s,e,l,r in sorted(spans,reverse=True): text=text[:s]+l+' '+text[s:e]+' '+r+text[e:]
    return text

def encode_marked(tokenizer, text, max_length=256, target=None, require_opinion=True):
    """Crop around BOTH spans. No loss of markers to silent right truncation."""
    ids=tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
    needed=MARKERS if require_opinion else MARKERS[:2]
    mids=[tokenizer.convert_tokens_to_ids(m) for m in needed]
    if len(set(mids))!=len(mids) or tokenizer.unk_token_id in mids: raise ValueError('Register ABSA markers before training')
    positions=[]
    for mid in mids:
        found=[i for i,v in enumerate(ids) if v==mid]
        if len(found)!=1: raise ValueError('Each pair marker must appear exactly once')
        positions.extend(found)
    target_ids=tokenizer(str(target),add_special_tokens=False)['input_ids'] if target is not None else None
    special=tokenizer.num_special_tokens_to_add(pair=target_ids is not None)
    budget=max_length-special-(len(target_ids) if target_ids is not None else 0)
    left,right=min(positions),max(positions)+1
    if right-left>budget or budget<=0: raise PairTooWide('Pair hull exceeds token budget; queue explicitly, never truncate a target')
    start=max(0,left-(budget-(right-left))//2)
    end=min(len(ids),start+budget); start=max(0,end-budget)
    return tokenizer.prepare_for_model(ids[start:end],pair_ids=target_ids,add_special_tokens=True,
                                       return_attention_mask=True,truncation=False)

class SpanPairRelationModel(nn.Module):
    """Keeps V13 separate task encoders; only relation representation changes in V14.0."""
    def __init__(self, encoder, marker_ids, dropout=.1):
        super().__init__()
        self.encoder=encoder; self.config=encoder.config
        self.marker_ids=[int(x) for x in marker_ids]
        self.dropout_rate=dropout
        h=encoder.config.hidden_size
        self.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(h*5+2,h),nn.GELU(),nn.Dropout(dropout),nn.Linear(h,2))
    @classmethod
    def from_encoder(cls, name, tokenizer):
        from transformers import AutoModel
        encoder=AutoModel.from_pretrained(name)
        encoder.resize_token_embeddings(len(tokenizer))
        return cls(encoder,[tokenizer.convert_tokens_to_ids(m) for m in MARKERS])
    def gradient_checkpointing_enable(self):
        self.encoder.gradient_checkpointing_enable()
    def _pool(self,h,ids,open_id,close_id):
        open_mask=ids.eq(open_id); close_mask=ids.eq(close_id)
        if not torch.all(open_mask.sum(1)==1) or not torch.all(close_mask.sum(1)==1): raise ValueError('Missing/duplicate pair marker')
        lo=open_mask.long().argmax(1); hi=close_mask.long().argmax(1)
        if not torch.all(hi>lo+1): raise ValueError('Empty/inverted marked span')
        ar=torch.arange(ids.shape[1],device=ids.device)[None,:]
        mask=(ar>lo[:,None]) & (ar<hi[:,None])
        pooled=(h*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return pooled,lo,hi
    def forward(self,input_ids,attention_mask=None,**kwargs):
        kwargs.pop('labels',None)
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask,**kwargs).last_hidden_state
        a,al,ah=self._pool(h,input_ids,*self.marker_ids[:2]); o,ol,oh=self._pool(h,input_ids,*self.marker_ids[2:])
        denom=attention_mask.sum(1).clamp_min(1) if attention_mask is not None else input_ids.new_full((len(input_ids),),input_ids.shape[1])
        dist=torch.stack(((ol-al)/denom,torch.abs(ol-al)/denom),-1).to(h.dtype)
        representation=torch.cat((h[:,0],a,o,a*o,torch.abs(a-o),dist),-1)
        return SimpleNamespace(logits=self.classifier(representation))
    def save_pretrained(self,path,**kwargs):
        from safetensors.torch import save_file
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        self.config.save_pretrained(path/'encoder_config')
        (path/'config.json').write_text(json.dumps({'model_type':'v14_span_pair_relation','marker_ids':self.marker_ids,'dropout':self.dropout_rate,'label2id':{'not_related':0,'related':1}},indent=2))
        save_file({k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()},str(path/'model.safetensors'))
    @classmethod
    def from_pretrained(cls,path,device='cpu'):
        from transformers import AutoConfig,AutoModel
        from safetensors.torch import load_file
        path=Path(path); cfg=json.loads((path/'config.json').read_text())
        encoder=AutoModel.from_config(AutoConfig.from_pretrained(path/'encoder_config',local_files_only=True))
        obj=cls(encoder,cfg['marker_ids'],cfg['dropout'])
        obj.load_state_dict(load_file(str(path/'model.safetensors')),strict=True)
        return obj.to(device).eval()

def relation_metrics(labels,scores,threshold=.5):
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support,f1_score,balanced_accuracy_score,matthews_corrcoef,average_precision_score,confusion_matrix
    y=np.asarray(labels,dtype=int); s=np.asarray(scores,dtype=float);p=(s>=threshold).astype(int)
    if len(y)==0: return None
    pr,rc,f,_=precision_recall_fscore_support(y,p,labels=[0,1],zero_division=0)
    macro=float(f.mean());ba=float(balanced_accuracy_score(y,p)) if len(set(y))==2 else 0.
    mcc=float(matthews_corrcoef(y,p)) if len(set(y))==2 else 0.
    return {'precision':float(pr[1]),'recall':float(rc[1]),'f1':float(f[1]),'macro_f1':macro,'balanced_accuracy':ba,
            'mcc':mcc,'negative_recall':float(rc[0]),'positive_recall':float(rc[1]),
            'pr_auc':float(average_precision_score(y,s)) if len(set(y))==2 else None,
            'composite':.45*macro+.30*ba+.25*max(0,mcc),'scores':s.tolist(),'labels':y.tolist(),'predictions':p.tolist(),
            'confusion_matrix':confusion_matrix(y,p,labels=[0,1]).tolist(),'predicted_related_fraction':float(p.mean()),
            'score_min':float(s.min()),'score_max':float(s.max()),'score_std':float(s.std())}

def relation_gate(metrics, floor=.75, min_recall=.70):
    return bool(metrics and metrics['macro_f1']>=floor and metrics['balanced_accuracy']>=floor
                and min(metrics['negative_recall'],metrics['positive_recall'])>=min_recall
                and .05<=metrics['predicted_related_fraction']<=.95)


def encode_sentiment_example(tokenizer, example, max_length=384):
    """Retain SmSA document warm-up and protected aspect/pair windows."""
    text=example['marked_context']
    if example.get('target_type') == 'document_sentiment_warmup':
        return tokenizer(text,text_pair=example.get('target','DOCUMENT'),truncation=True,max_length=max_length,padding=False)
    return encode_marked(tokenizer,text,max_length,target=example['target'],require_opinion='[OPN]' in text)
