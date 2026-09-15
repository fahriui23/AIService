"""Exact offsets, actual inference profile, denominator-safe evaluation and promotion gates."""
import math
from collections import Counter
from statistics import mean,stdev

FIELDS=('aspect_start','aspect_end','opinion_start','opinion_end')
def tuple_key(row, kind='triplet'):
    if kind=='aspect':return (row.get('aspect_start'),row.get('aspect_end'))
    if kind=='opinion':return (row.get('opinion_start'),row.get('opinion_end'))
    pair=tuple(row.get(k) for k in FIELDS)
    return pair if kind=='pair' else pair+(row.get('sentiment'),)

def prf(gold,pred):
    if len(gold)!=len(pred):raise ValueError('Gold/pred review alignment mismatch')
    tp=sum(len(g&p) for g,p in zip(gold,pred));fp=sum(len(p-g) for g,p in zip(gold,pred));fn=sum(len(g-p) for g,p in zip(gold,pred))
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'tp':tp,'fp':fp,'fn':fn,'support':tp+fn,'n_reviews':len(gold)}

def is_complete_human(r):
    return r.get('human_approved') is True and r.get('annotation_complete') is True and not r.get('is_synthetic',False)

def exact_evaluation(records,predict,profile='production_precision',human_only=True):
    records=[r for r in records if is_complete_human(r)] if human_only else list(records)
    if not records:return {'status':'not_available','profile':profile,'n_reviews':0,'human_gold':human_only}
    golds=[];preds=[];span_preds=[];methods=Counter();scores=[];correct=[];accepted=0;total=0;diagnostic=[]
    for r in records:
        g=[a for a in r['annotations'] if a.get('opinion_start') is not None and a.get('sentiment') in {'positive','negative','neutral'} and a.get('relation',True) is not False]
        rows=predict(r['text'],profile=profile)
        span_preds.append(rows)
        p=[x for x in rows if x.get('output_type','triplet')=='triplet' and x.get('opinion_start') is not None]
        # Same aspect/opinion/sentiment duplication must not inflate coverage/calibration denominators.
        p=list({tuple_key(x):x for x in p}.values())
        golds.append(g);preds.append(p);keys={tuple_key(x) for x in g}
        for x in p:
            methods[x.get('relation_method','unknown')]+=1;total+=1
            scores.append(float(x.get('overall_confidence',0)));correct.append(tuple_key(x) in keys)
            accepted+=int(not x.get('needs_human_review',True))
        diagnostic.append({'id':r['id'],'gold':[list(tuple_key(x)) for x in g],'predicted':[list(tuple_key(x)) for x in p]})
    result={'status':'evaluated','human_gold':human_only,'profile':profile,'n_reviews':len(records),'actual_methods':dict(methods)}
    for kind in ['aspect','opinion','pair','triplet']:
        # Span metrics use all span annotations, including factual/aspect-only mentions.
        gs=[r['annotations'] for r in records] if kind in {'aspect','opinion'} else golds
        result[kind]=prf([{tuple_key(x,kind) for x in g if not all(v is None for v in tuple_key(x,kind))} for g in gs],
                         [{tuple_key(x,kind) for x in p if not all(v is None for v in tuple_key(x,kind))} for p in (span_preds if kind in {'aspect','opinion'} else preds)])
    ng=[{tuple_key(x) for x in g if x.get('sentiment')=='negative'} for g in golds]
    np=[{tuple_key(x) for x in p if x.get('sentiment')=='negative'} for p in preds]
    result['negative_triplet']=prf(ng,np)
    result['tuple_calibration']=calibration_summary(scores,correct)
    ag=[];ap=[]
    for g,p in zip(golds,preds):
        ag.append({tuple_key(x) for x in g});ap.append({tuple_key(x) for x in p if not x.get('needs_human_review',True)})
    result['auto_accepted']=prf(ag,ap)
    result['auto_accepted']['prediction_coverage']=accepted/total if total else 0.
    result['auto_accepted']['gold_coverage']=result['auto_accepted']['recall']
    result['rows']=diagnostic
    return result

def calibration_summary(scores,correct,n_bins=10):
    if len(scores)!=len(correct): raise ValueError('Scores/labels length mismatch')
    if not scores:return {'n':0,'ece':None,'brier':None,'selective':[]}
    if any(not math.isfinite(s) or not 0<=s<=1 for s in scores):raise ValueError('Invalid tuple scores')
    ece=0.;n=len(scores)
    for i in range(n_bins):
        ids=[j for j,s in enumerate(scores) if i/n_bins<=s and (s<(i+1)/n_bins or i==n_bins-1)]
        if ids:ece+=len(ids)/n*abs(mean(scores[j] for j in ids)-mean(bool(correct[j]) for j in ids))
    selective=[]
    for threshold in [0.,.5,.6,.7,.8,.85,.9,.95]:
        ids=[j for j,s in enumerate(scores) if s>=threshold]
        selective.append({'threshold':threshold,'precision':mean(bool(correct[j]) for j in ids) if ids else None,'prediction_coverage':len(ids)/n,'n':len(ids)})
    return {'n':n,'ece':ece,'brier':mean((s-float(y))**2 for s,y in zip(scores,correct)),'selective':selective}

RESEARCH_TARGETS={'sentiment_macro_f1':.80,'positive_f1':.90,'negative_f1':.84,'neutral_f1':.68,
                  'aspect_exact_f1':.74,'aspect_sentiment_f1':.68,'relation_macro_f1':.75,
                  'relation_balanced_accuracy':.75,'macro_domain_f1':.72,'worst_domain_f1':.62}
PRODUCTION_TARGETS={'aspect_exact_f1':.80,'opinion_exact_f1':.85,'relation_macro_f1':.80,
                    'relation_negative_recall':.75,'relation_positive_recall':.75,
                    'positive_f1':.85,'negative_f1':.85,'neutral_f1':.65,'triplet_exact_f1':.75,
                    'negative_triplet_recall':.85,'auto_precision':.90,'auto_coverage':.70}

def promotion_report(candidate,baseline,required_metrics=None,tolerance=0.):
    """No-regression is paired evidence, not the old notebook's incomparable headline numbers."""
    required_metrics=required_metrics or list(RESEARCH_TARGETS)+['opinion_exact_f1','triplet_exact_f1','negative_triplet_recall']
    failures=[];rows=[]
    for key in ['dataset_fingerprint','split_fingerprint','profile','neutral_policy','metric_schema']:
        if not candidate.get(key) or candidate.get(key)!=baseline.get(key):failures.append('incomparable_'+key)
    for seed in [42,52,62]:
        if str(seed) not in candidate.get('seeds',{}) or str(seed) not in baseline.get('seeds',{}):failures.append(f'missing_seed_{seed}')
    for metric in required_metrics:
        values=[];deltas=[]
        for seed in ['42','52','62']:
            c=candidate.get('seeds',{}).get(seed,{}).get(metric); b=baseline.get('seeds',{}).get(seed,{}).get(metric)
            if c is None or b is None or not (math.isfinite(c) and math.isfinite(b)):continue
            values.append(c);deltas.append(c-b)
        if len(values)!=3:failures.append('missing_metric_'+metric);continue
        ok=all(d>=-tolerance for d in deltas)
        if not ok:failures.append('regression_'+metric)
        rows.append({'metric':metric,'mean':mean(values),'std':stdev(values),'paired_mean_delta':mean(deltas),'worst_seed_delta':min(deltas),'passed':ok})
    improvement=any(x['paired_mean_delta']>tolerance for x in rows if x['metric'] in {'relation_macro_f1','triplet_exact_f1','negative_triplet_recall'})
    if not improvement:failures.append('no_demonstrated_primary_improvement')
    return {'passed':not failures,'status':'eligible_for_further_production_gates' if not failures else 'not_promoted','failures':failures,'metrics':rows}
