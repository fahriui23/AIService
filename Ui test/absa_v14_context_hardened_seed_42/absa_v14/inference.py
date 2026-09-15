"""One runtime for notebook, batch and exported bundle. No training at import time."""
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
import torch
from .pair import mark_pair,encode_marked,PairTooWide,SpanPairRelationModel

PROFILES = {
    'scientific_balanced':{'aspect_threshold':.5,'opinion_threshold':.5,'recovery':False,'dual_view':False},
    'production_precision':{'aspect_threshold':.6,'opinion_threshold':.65,'recovery':False,'dual_view':False},
    'maps_high_recall':{'aspect_threshold':.35,'opinion_threshold':.35,'recovery':True,'dual_view':True},
}

def coordination(text,a,b):
    left,right=sorted([a,b],key=lambda x:x['start'])
    gap=text[left['end']:right['start']].strip().lower()
    return bool(re.fullmatch(r'(?:dan|and|serta|&|と|و)',gap))

def global_decode(text,candidates,threshold=.5,max_opinions=8,sharing_threshold=.90,ambiguity_margin=.05):
    """Opinion-level competition; allow coordinated and confidently shared references."""
    by_op=defaultdict(list)
    for c in candidates:
        if c['score']>=threshold:by_op[(c['opinion']['start'],c['opinion']['end'])].append(c)
    kept=[];rejected=[]
    for rows in by_op.values():
        rows=sorted(rows,key=lambda x:(-x['score'],x['aspect']['start']))
        best=rows[0]
        competing=[r for r in rows[1:] if not coordination(text,best['aspect'],r['aspect']) and best['score']-r['score']<ambiguity_margin]
        kept.append({**best,'ambiguous_binding':bool(competing)})
        for row in rows[1:]:
            coordinated=coordination(text,best['aspect'],row['aspect'])
            shared=row['score']>=sharing_threshold and best['score']-row['score']<=ambiguity_margin
            if coordinated or shared:kept.append({**row,'shared_opinion':True,'ambiguous_binding':not coordinated})
            else:rejected.append({**row,'reason':'competing_aspect'})
    # Per-aspect cap prevents combinatorial output; overflow is explicit in review queue.
    final=[];counts=defaultdict(int)
    for row in sorted(kept,key=lambda x:-x['score']):
        a=(row['aspect']['start'],row['aspect']['end'])
        if counts[a]>=max_opinions:rejected.append({**row,'reason':'max_opinions_per_aspect'});continue
        counts[a]+=1;final.append(row)
    return final,rejected

def negation_scope(text,opinion):
    if opinion is None:return {'negated':False,'negation_scope':None,'negation_terms':[]}
    s,e=opinion['start'],opinion['end']
    # Opinion-local metadata only. No automatic negative -> neutral conversion.
    lo=max(0,s-32); prefix=text[lo:s]
    boundary=list(re.finditer(r'[,;.!?]|\b(?:tapi|tp|but|namun|melainkan|although)\b',prefix,re.I))
    if boundary:lo+=boundary[-1].end()
    scope=text[lo:e]
    matches=list(re.finditer(r'\b(?:tidak|tak|bukan|belum|ga|gak|nggak|not|never|no)\b',scope,re.I))
    return {'negated':bool(matches),'negation_scope':[lo,e] if matches else None,'negation_terms':[m.group() for m in matches]}

class TupleCalibrator:
    def __init__(self,artifact=None):self.artifact=artifact or {'status':'not_fitted'}
    @classmethod
    def fit(cls,scores,correct,metadata,min_samples=100):
        if metadata.get('split')!='calibration' or metadata.get('human_complete') is not True:
            raise ValueError('Tuple calibration requires disjoint human-complete calibration records')
        if len(scores)<min_samples or min(sum(correct),len(correct)-sum(correct))<10:
            return cls({'status':'not_fitted_insufficient_human_tuples','n':len(scores)})
        from sklearn.isotonic import IsotonicRegression
        model=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(scores,correct)
        return cls({'status':'fitted','x':model.X_thresholds_.tolist(),'y':model.y_thresholds_.tolist(),'metadata':metadata,'n':len(scores)})
    def predict(self,score,profile):
        a=self.artifact
        if a.get('status')!='fitted' or a['metadata'].get('profile')!=profile:return score,False
        import numpy as np
        return float(np.interp(score,a['x'],a['y'])),True

class Components:
    def __init__(self,tokenizer,models,config,labels,normalizer=None,recover_aspects=None,recover_opinions=None):
        self.tokenizer=tokenizer;self.models=models;self.config=config;self.labels=labels
        self.normalizer=normalizer;self.recover_aspects=recover_aspects;self.recover_opinions=recover_opinions
        for m in models.values():
            if m is not None:m.eval()
    def device(self,name):return next(self.models[name].parameters()).device
    @torch.inference_mode()
    def spans(self,text,kind,threshold,dual=False):
        model=self.models.get(kind)
        if model is None:return []
        views=[(text,None)]
        if dual and self.normalizer is not None:
            aligned=self.normalizer(text)
            if aligned['normalized_text']!=text:views.append((aligned['normalized_text'],aligned['norm_to_raw']))
        spans=[]
        for view,mapping in views:
            enc=self.tokenizer(view,truncation=True,max_length=self.config.get('span_max_length',256),stride=self.config.get('span_stride',96),
                               return_overflowing_tokens=True,return_offsets_mapping=True)
            for i,offsets in enumerate(enc['offset_mapping']):
                inputs={k:torch.tensor([enc[k][i]],device=self.device(kind)) for k in ['input_ids','attention_mask']}
                probs=model(**inputs).logits[0].softmax(-1).cpu();ids=probs.argmax(-1).tolist();cur=None;out=[]
                for j,((s,e),label_id) in enumerate(zip(offsets,ids)):
                    if e<=s:continue
                    tag=self.labels['bio'][str(label_id)]
                    if tag=='B':
                        if cur:out.append(cur)
                        cur={'start':s,'end':e,'scores':[float(probs[j,label_id])]}
                    elif tag=='I':
                        # A sliding window can start inside an entity; keep candidate with explicit boundary metadata.
                        if cur is None:cur={'start':s,'end':e,'scores':[],'window_fragment':True}
                        cur['end']=e;cur['scores'].append(float(probs[j,label_id]))
                    elif cur:out.append(cur);cur=None
                if cur:out.append(cur)
                for s in out:
                    s.pop('scores',None)
                    hits=[j for j,(lo,hi) in enumerate(offsets) if hi>lo and lo<s['end'] and hi>s['start']]
                    s['score']=sum(float(probs[j,ids[j]]) for j in hits)/len(hits) if hits else 0.
                    if s['score']<threshold:continue
                    if mapping:
                        raw=mapping[s['start']:s['end']]
                        if not raw:continue
                        s['start']=min(p[0] for p in raw);s['end']=max(p[1] for p in raw)
                    s['source_view']='normalized' if mapping else 'raw';s['is_recovery']=False;spans.append(s)
        return merge_spans(spans)
    @torch.inference_mode()
    def relation(self,text,a,o):
        model=self.models.get('relation')
        if model is None or not self.config.get('relation_usable'):return None
        enc=encode_marked(self.tokenizer,mark_pair(text,a['start'],a['end'],o['start'],o['end']),self.config.get('relation_max_length',384))
        inp={k:torch.tensor([v],device=self.device('relation')) for k,v in enc.items()}
        return float(model(**inp).logits[0].softmax(-1)[1])
    @torch.inference_mode()
    def classify(self,text,a,o=None,task='sentiment'):
        model=self.models.get(task)
        if model is None:return None,0.,{}
        marked=mark_pair(text,a['start'],a['end'],o['start'] if o else None,o['end'] if o else None)
        target=text[a['start']:a['end']] if task=='sentiment' else None
        enc=encode_marked(self.tokenizer,marked,self.config.get('sentiment_max_length',256),target=target,require_opinion=o is not None)
        inp={k:torch.tensor([v],device=self.device(task)) for k,v in enc.items()}
        temp=self.config.get('sentiment_temperature',1.) if task=='sentiment' else 1.
        p=(model(**inp).logits[0]/temp).softmax(-1);pid=int(p.argmax()); labels=self.labels[task]
        return labels[str(pid)],float(p[pid]),{labels[str(i)]:float(v) for i,v in enumerate(p)}

def merge_spans(spans):
    kept=[]
    for s in sorted(spans,key=lambda x:(-x['score'],-(x['end']-x['start']))):
        if any(max(s['start'],k['start'])<min(s['end'],k['end']) and
               (min(s['end'],k['end'])-max(s['start'],k['start']))/max(1,min(s['end']-s['start'],k['end']-k['start']))>=.8 for k in kept):continue
        kept.append(s)
    return sorted(kept,key=lambda x:(x['start'],x['end']))

class ABSAEngine:
    def __init__(self,components,profiles=None,calibrators=None):
        self.components=components;self.config=components.config
        self.profiles={k:{**v,**(profiles or {}).get(k,{})} for k,v in PROFILES.items()}
        self.calibrators=calibrators or {};self.last_diagnostics={}
    def predict(self,text,profile='production_precision'):
        if not isinstance(text,str):raise TypeError('review text must be a string')
        if profile not in self.profiles:raise ValueError(f'Unknown profile {profile}')
        if len(text)>self.config.get('max_input_chars',30000):raise ValueError('Review exceeds configured maximum input length; split upstream preserving offsets')
        self.last_diagnostics={}
        if not text.strip():return []
        if any(x in text for x in ['[ASP]','[/ASP]','[OPN]','[/OPN]']):raise ValueError('Input contains reserved ABSA markers')
        start_time=time.perf_counter();cfg=self.profiles[profile];c=self.components
        aspects=c.spans(text,'aspect',cfg['aspect_threshold'],cfg['dual_view'])
        opinions=c.spans(text,'opinion',cfg['opinion_threshold'],cfg['dual_view'])
        if cfg['recovery']:
            for kind,current,fn in [('aspect',aspects,c.recover_aspects),('opinion',opinions,c.recover_opinions)]:
                if fn:
                    for item in fn(text):current.append({**item,'is_recovery':True,'score':min(.60,float(item.get('score',.5)))})
        if cfg['recovery']:
            from .legacy_resources import split_coordinated_aspect
            repaired=[]
            for aspect in aspects:
                for fragment in split_coordinated_aspect(text,aspect):
                    repaired.append({**fragment,'is_recovery':True} if fragment.get('coordination_repair') else fragment)
            aspects=repaired
        aspects=merge_spans(aspects);opinions=merge_spans(opinions)
        candidates=[];issues=[];attempted=0;max_pairs=self.config.get('max_candidate_pairs',512)
        # Score nearest first for bounded latency; every skipped candidate is disclosed.
        proposals=sorted([(a,o) for a in aspects for o in opinions],key=lambda pair:abs(pair[0]['start']-pair[1]['start']))
        for a,o in proposals:
            if attempted>=max_pairs or time.perf_counter()-start_time>self.config.get('timeout_seconds',120):
                issues.append({'reason':'candidate_budget_or_timeout','skipped_pairs':len(proposals)-attempted});break
            attempted+=1
            try:
                score=c.relation(text,a,o)
                if score is not None:candidates.append({'aspect':a,'opinion':o,'score':score,'method':'learned_span_pair_v14'})
                elif cfg['recovery']:
                    # Recovery preserves V13 review assistance, but cannot produce accepted scientific/production tuples.
                    between=text[min(a['end'],o['end']):max(a['start'],o['start'])]
                    if not re.search(r'[.!?;]|\b(?:tapi|but|melainkan|namun)\b',between,re.I):
                        candidates.append({'aspect':a,'opinion':o,'score':.5/(1+abs(a['start']-o['start'])/100), 'method':'heuristic_review_only_v14'})
            except (PairTooWide,ValueError) as exc:issues.append({'aspect':[a['start'],a['end']],'opinion':[o['start'],o['end']],'reason':str(exc)})
        threshold=self.config.get('relation_threshold',.5) if self.config.get('relation_usable') else .1
        pairs,rejected=global_decode(text,candidates,threshold,self.config.get('max_opinions_per_aspect',8))
        overflow=[r for r in rejected if r.get('reason')=='max_opinions_per_aspect']
        if overflow:issues.append({'reason':'max_opinions_per_aspect','skipped_pairs':len(overflow)})
        results=[];paired_aspects=set()
        for pair in pairs:
            a,o=pair['aspect'],pair['opinion'];reasons=[]
            try:
                sentiment,sc,probs=c.classify(text,a,o,'sentiment');tax,tc,_=c.classify(text,a,o,'taxonomy')
            except (PairTooWide,ValueError) as exc:issues.append({'reason':str(exc),'aspect':[a['start'],a['end']]});continue
            if sentiment is None:continue
            taxonomy_method='learned_aspect_opinion_context_v14'
            if cfg['recovery'] and (tax in {None,'Other'} or tc<self.config.get('taxonomy_min_confidence',.60)):
                from .legacy_resources import taxonomy_mapper
                fallback=taxonomy_mapper(text[a['start']:a['end']],text)
                tax=fallback['normalized_category'];tc=fallback['taxonomy_confidence']
                taxonomy_method='legacy_taxonomy_review_hint'
                reasons.append('heuristic_taxonomy')
            raw_score=min(a['score'],o['score'],pair['score'],sc)
            calibrator=self.calibrators.get(profile,TupleCalibrator());conf,calibrated=calibrator.predict(raw_score,profile)
            heuristic=pair['method']!='learned_span_pair_v14' or a.get('is_recovery') or o.get('is_recovery')
            if heuristic:calibrated=False;reasons.append('heuristic_or_lexicon_evidence')
            if not calibrated:reasons.append('tuple_confidence_not_calibrated')
            if not self.config.get('relation_human_validated'):reasons.append('relation_not_human_validated')
            if pair.get('ambiguous_binding'):reasons.append('ambiguous_binding')
            if conf<self.config.get('auto_accept_threshold',.90):reasons.append('low_tuple_confidence')
            if tax in {None,'Other'} or tc<self.config.get('taxonomy_min_confidence',.60):reasons.append('taxonomy_needs_review')
            if issues:reasons.append('incomplete_candidate_processing')
            evidence={'term':text[o['start']:o['end']],'start':o['start'],'end':o['end'],'confidence':o['score'],'relation_probability':pair['score'],'relation_score_type':'uncalibrated_model_probability' if not heuristic else 'heuristic_strength'}
            results.append({'output_type':'triplet','aspect_term':text[a['start']:a['end']],'aspect_start':a['start'],'aspect_end':a['end'],
                            'aspect_confidence':a['score'],'aspect_source_view':a.get('source_view','raw'),
                            'opinion_term':evidence['term'],'opinion_start':o['start'],'opinion_end':o['end'],
                            'opinion_terms':[evidence['term']],'opinion_evidence':[evidence],
                            'sentiment':sentiment,'sentiment_confidence':sc,'sentiment_probabilities':probs,
                            'sentiment_method':'learned_per_pair_v14','sentiment_rule_applied':False,
                            'normalized_category':tax or 'Other','taxonomy_id':'v14.'+re.sub(r'[^a-z0-9]+','_',str(tax or 'Other').lower()),
                            'taxonomy_confidence':tc,'taxonomy_method':taxonomy_method,
                            'relation_method':pair['method'],'relation_model_usable':bool(self.config.get('relation_usable')),
                            'relation_scores':[pair['score']],'overall_confidence':conf,'raw_tuple_score':raw_score,
                            'confidence_is_calibrated':calibrated,'needs_human_review':bool(reasons),'review_reasons':reasons,
                            'profile':profile,**negation_scope(text,o)})
            paired_aspects.add((a['start'],a['end']))
        for a in aspects:
            if (a['start'],a['end']) in paired_aspects:continue
            # Retain the V13 aspect-conditioned capability as a typed review item, never fake an opinion span.
            try:sent,score,probs=c.classify(text,a,None,'sentiment');tax,tc,_=c.classify(text,a,None,'taxonomy')
            except (PairTooWide,ValueError):sent,score,probs,tax,tc=None,0.,{},'Other',0.
            results.append({'output_type':'aspect_only','aspect_term':text[a['start']:a['end']],'aspect_start':a['start'],'aspect_end':a['end'],
                            'opinion_term':None,'opinion_start':None,'opinion_end':None,'opinion_terms':[],'opinion_evidence':[],
                            'sentiment':sent,'sentiment_confidence':score,'sentiment_probabilities':probs,'normalized_category':tax,
                            'taxonomy_id':'v14.other' if tax in {None,'Other'} else 'v14.'+re.sub(r'[^a-z0-9]+','_',tax.lower()),
                            'taxonomy_confidence':tc,'overall_confidence':min(a['score'],score),'confidence_is_calibrated':False,
                            'relation_method':'no_valid_pair','relation_model_usable':bool(self.config.get('relation_usable')),
                            'needs_human_review':True,'review_reasons':['no_valid_pair'],'profile':profile})
        if issues:
            for row in results:
                row['needs_human_review']=True
                if 'incomplete_candidate_processing' not in row['review_reasons']:row['review_reasons'].append('incomplete_candidate_processing')
        self.last_diagnostics={'candidate_pairs':attempted,'issues':issues,'rejected_pair_count':len(rejected),'latency_seconds':time.perf_counter()-start_time}
        if cfg['recovery']:
            from .legacy_resources import add_alias_metadata
            results=add_alias_metadata(results)
        return sorted(results,key=lambda x:(x['aspect_start'],x['opinion_start'] if x['opinion_start'] is not None else -1))
    def predict_batch(self,texts,profile='production_precision'):
        rows=[];queue=[]
        for i,text in enumerate(texts):
            try:
                predictions=self.predict(text,profile)
                if not predictions:queue.append({'review_index':i,'output_type':'review','needs_human_review':True,'review_reasons':['no_triplets_detected']})
                for item in predictions:
                    row={'review_index':i,**item};rows.append(row)
                    if row['needs_human_review']:queue.append(row)
            except (ValueError,TypeError) as exc:queue.append({'review_index':i,'output_type':'error','needs_human_review':True,'review_reasons':[str(exc)]})
        return rows,queue
    def save_config(self,path):
        obj={'version':'14.0.0-candidate','config':self.config,'labels':self.components.labels,'profiles':self.profiles,
             'calibrators':{k:v.artifact for k,v in self.calibrators.items()},'components':{k:v is not None for k,v in self.components.models.items()}}
        Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
    @classmethod
    def from_pretrained(cls,path,device='cpu'):
        from transformers import AutoTokenizer,AutoModelForTokenClassification,AutoModelForSequenceClassification
        from . import legacy_resources as resources
        path=Path(path);cfg=json.loads((path/'inference_config.json').read_text());models={}
        for name,present in cfg['components'].items():
            if not present:models[name]=None;continue
            if name=='relation':models[name]=SpanPairRelationModel.from_pretrained(path/name,device)
            else:
                loader=AutoModelForTokenClassification if name in {'aspect','opinion'} else AutoModelForSequenceClassification
                models[name]=loader.from_pretrained(path/name,local_files_only=True).to(device).eval()
        tokenizer=AutoTokenizer.from_pretrained(path/'tokenizer',local_files_only=True,use_fast=True)
        c=Components(tokenizer,models,cfg['config'],cfg['labels'],resources.normalize_multilingual_text_with_alignment,
                     resources.recover_aspects,resources.recover_opinions)
        return cls(c,cfg['profiles'],{k:TupleCalibrator(v) for k,v in cfg.get('calibrators',{}).items()})
