"""V14.1 dataset contract, upload-safe bundle loader, and split audit.

Raw text is immutable. Dataset trust is read from fields, never from filenames.
A structural PASS is not a production-readiness claim.
"""
from __future__ import annotations
import csv, hashlib, json, math, re, unicodedata, zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

class ContractError(ValueError):
    pass

SENTIMENTS = {'positive', 'negative', 'neutral'}
GOLD_STATUSES = {'approved', 'human_approved', 'adjudicated', 'human_gold', 'gold'}
PROTECTED = {'validation', 'test', 'unseen_test', 'diagnostic', 'human_review_queue'}
TRAIN_ROLES = {'fit', 'inner_tune', 'calibration'}


def strict_bool(value, field):
    if isinstance(value, bool): return value
    if value in (0, 1) and isinstance(value, int): return bool(value)
    if isinstance(value, str) and value.strip().lower() in {'true','false','0','1'}:
        return value.strip().lower() in {'true','1'}
    raise ContractError(f'{field}: expected explicit boolean, received {value!r}')


def raw_exact_key(text):
    """Hard exact key: NFKC + whitespace collapse, with punctuation/case retained."""
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text))).strip()


def normalized_soft_key(text):
    """Language-aware soft audit key; never the identity for exact deduplication."""
    value=unicodedata.normalize('NFKC',str(text)).casefold()
    value=re.sub(r'[^\w\s-]', ' ', value, flags=re.UNICODE)
    value=re.sub(r'\s+', ' ', value).strip()
    soft_map={
        'gk':'tidak','ga':'tidak','gak':'tidak','nggak':'tidak','ngga':'tidak','tdk':'tidak',
        'bgt':'banget','bngt':'banget','tp':'tapi','tpi':'tapi','pdhl':'padahal',
        'udh':'sudah','sdh':'sudah','blm':'belum','krn':'karena','dgn':'dengan',
        'nunggu':'menunggu','mnt':'menit','mins':'menit','luv':'love','gud':'good','gr8':'great',
    }
    value=' '.join(soft_map.get(token,token) for token in value.split())
    value=re.sub(r'\b([\w-]+)\s+(nya|ku|mu)\b',r'\1\2',value)
    value=re.sub(r'([a-z])\1{2,}',r'\1\1',value)
    return value


def norm_key(text):
    """Compatibility alias; now conservative enough for grouping/debug output."""
    return raw_exact_key(text)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safe_extract(source,destination,max_bytes=12_000_000_000,max_files=50_000):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(source) as z:
        infos=z.infolist()
        if len(infos)>max_files:raise ContractError('ZIP exceeds configured member limit')
        if sum(i.file_size for i in infos)>max_bytes:raise ContractError('ZIP exceeds configured uncompressed size limit')
        for info in infos:
            name=PurePosixPath(info.filename.replace('\\','/'));mode=info.external_attr>>16
            if name.is_absolute() or '..' in name.parts or ':' in str(name) or mode&0o170000==0o120000:
                raise ContractError(f'Unsafe ZIP member: {info.filename}')
            target=(destination/str(name)).resolve()
            if not target.is_relative_to(destination):raise ContractError('ZIP path escaped destination')
        z.extractall(destination)
    return destination


def read_rows(path):
    path=Path(path);suffix=path.suffix.lower()
    if suffix=='.jsonl':return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if suffix=='.json':
        obj=json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(obj,list):return obj
        if isinstance(obj,dict) and isinstance(obj.get('records'),list):return obj['records']
        raise ContractError(f'{path.name}: JSON must be a list or contain records[]')
    if suffix=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        for row in rows:
            for key in ('annotations','phenomena','provenance','tasks','mentions','buckets'):
                if row.get(key):row[key]=json.loads(row[key])
        return rows
    raise ContractError(f'Unsupported dataset file: {path.name}')


def pick(row,names,default=None):
    for name in names:
        if name in row and row[name] not in (None,''):return row[name]
    return default


def _int_or_none(value,field,rid):
    if value in (None,''):return None
    try:
        if float(value)!=int(float(value)):raise ValueError()
        return int(float(value))
    except (TypeError,ValueError):raise ContractError(f'{rid}: {field} must be an integer or null')


def _annotation_signature(record):
    values=[(a.get('aspect_start'),a.get('aspect_end'),a.get('opinion_start'),a.get('opinion_end'),a.get('sentiment'),a.get('aspect_category')) for a in record.get('annotations',[])]
    return tuple(sorted(values,key=repr))


def canonical_record(row,neutral_policy,field_map=None):
    row=dict(row)
    for canonical,source in (field_map or {}).items():
        if source in row:row[canonical]=row[source]
    rid=pick(row,['id','record_id','review_id']);text=pick(row,['text','review_text_raw','raw_text'])
    if rid is None or not isinstance(text,str) or not text.strip():raise ContractError('Every record needs stable id and immutable nonempty raw text')
    rid=str(rid);split=pick(row,['split','official_split'])
    split={'dev':'validation','valid':'validation','val':'validation','frozen_diagnostic':'diagnostic'}.get(split,split)
    if split not in {'train'}|PROTECTED:raise ContractError(f'{rid}: unknown split {split!r}')
    source=pick(row,['source_dataset','source']);source_bundle=str(row.get('source_bundle','legacy'))
    source_human_status=str(row.get('human_status','')).lower()
    quality=str(pick(row,['annotation_status','quality_tier'],source_human_status or 'unlabeled_pending_human')).lower()
    if not source:raise ContractError(f'{rid}: source_dataset required')
    synthetic=strict_bool(row.get('is_synthetic',row.get('synthetic','synthetic' in quality or 'synthetic' in str(source).lower())),'is_synthetic')
    human=strict_bool(row.get('human_approved',False),'human_approved') or source_human_status in GOLD_STATUSES
    if human and (synthetic or 'silver' in quality or 'translated' in quality):raise ContractError(f'{rid}: synthetic/SILVER/translated record cannot be natural human GOLD')
    complete=strict_bool(row.get('annotation_complete',False),'annotation_complete')
    family=pick(row,['family_id','template_family_id','template_family','parent_review_id'])
    if synthetic and not family:raise ContractError(f'{rid}: synthetic record requires family/template/parent id')
    weight=float(row.get('sample_weight',1.0 if human else .25))
    if not math.isfinite(weight) or weight<=0:raise ContractError(f'{rid}: sample_weight must be finite and positive')
    raw_anns=row.get('annotations',row.get('triplets',[]))
    if isinstance(raw_anns,str):raw_anns=json.loads(raw_anns)
    if not isinstance(raw_anns,list):raise ContractError(f'{rid}: annotations must be a list')
    clean=[];mentions=list(row.get('mentions',[]));seen={};unsupported=False
    for original in raw_anns:
        a=dict(original)
        for kind in ('aspect','opinion'):
            s=_int_or_none(a.get(kind+'_start'),kind+'_start',rid);e=_int_or_none(a.get(kind+'_end'),kind+'_end',rid)
            if (s is None)!=(e is None):raise ContractError(f'{rid}: partial {kind} offsets')
            if s is None:
                a[kind+'_start']=a[kind+'_end']=None
                if a.get(kind+'_term') in ('',):a[kind+'_term']=None
                continue
            if not 0<=s<e<=len(text):raise ContractError(f'{rid}: invalid {kind} offsets {(s,e)}')
            term=a.get(kind+'_term')
            if term is not None and str(term)!=text[s:e]:raise ContractError(f'{rid}: exact {kind} term/offset mismatch; raw text was not changed')
            a.update({kind+'_start':s,kind+'_end':e,kind+'_term':text[s:e]})
        factual=(a.get('mention_type')=='factual_no_opinion' or a.get('annotation_type') in {'factual','no_opinion'} or strict_bool(a.get('factual',False),'factual'))
        sent=a.get('sentiment');sent=None if sent is None else {'pos':'positive','neg':'negative','neu':'neutral'}.get(str(sent).lower(),str(sent).lower())
        if factual:
            mention={**a,'annotation_type':'factual','mention_type':'factual_no_opinion','sentiment':None,'relation':None}
            mentions.append(mention)
            if neutral_policy=='factual_is_neutral':
                # Retained only for backward experiments; V14 release uses no-opinion.
                a={**a,'sentiment':'neutral','relation':False,'mention_type':'factual_no_opinion'}
            else:continue
        if sent in {'conflict','mixed','unknown','abstain','no_opinion','none',''}:
            if sent not in {None,'','none'}:mentions.append({**a,'sentiment_original':sent,'annotation_type':'unsupported'});unsupported=True;continue
            sent=None
        if sent not in SENTIMENTS:raise ContractError(f'{rid}: invalid sentiment {sent!r}')
        a['sentiment']=sent
        a['aspect_category']=pick(a,['issue_category','taxonomy_label','aspect_category','category'])
        a['entity_category']=a.get('entity_category')
        a['implicit']=strict_bool(a.get('implicit',a.get('aspect_start') is None),'implicit')
        rel=a.get('relation')
        if rel is None and a.get('relation_label') is not None:rel=int(a['relation_label'])==1
        if rel is None and a.get('relation_type') is not None:rel=str(a['relation_type']).lower()=='related'
        a['relation']=True if rel is None else strict_bool(rel,'relation')
        sig=tuple(a.get(k) for k in ('aspect_start','aspect_end','opinion_start','opinion_end'))
        if sig in seen and seen[sig]!=sent:raise ContractError(f'{rid}: contradictory sentiment for same exact pair')
        if sig in seen and seen[sig]==sent:continue
        seen[sig]=sent;clean.append(a)
    raw_tasks=row.get('tasks')
    if raw_tasks is None:
        raw_tasks=[]
        if any(a.get('sentiment') in SENTIMENTS for a in clean):raw_tasks.append('sentiment')
        if any(a.get('aspect_start') is not None for a in clean):raw_tasks.append('aspect')
        if any(a.get('opinion_start') is not None for a in clean):raw_tasks.append('opinion')
        if any(a.get('relation') is True and a.get('aspect_start') is not None and a.get('opinion_start') is not None for a in clean):raw_tasks.append('relation')
        if any(a.get('aspect_category') for a in clean):raw_tasks.append('taxonomy')
    if isinstance(raw_tasks,str):raw_tasks=json.loads(raw_tasks)
    allowed={'sentiment','aspect','opinion','relation','taxonomy','dapt'}
    if not isinstance(raw_tasks,list) or set(raw_tasks)-allowed:raise ContractError(f'{rid}: invalid task eligibility')
    return {**row,'id':rid,'text':text,'raw_text':text,'split':split,'source_dataset':str(source),'source_bundle':source_bundle,
            'source_record_id':str(row.get('source_record_id',rid)),'annotation_status':quality,'human_approved':bool(human),
            'human_status':'human_approved' if human else source_human_status or 'not_human_gold','is_synthetic':bool(synthetic),
            'annotation_complete':bool(complete and not unsupported),'source_annotation_complete':bool(complete),'family_id':family,
            'tasks':raw_tasks,'annotations':clean,'mentions':mentions,'language':row.get('language','unknown'),'domain':row.get('domain','unknown'),
            'sample_weight':weight,'neutral_policy':neutral_policy,'raw_exact_key':raw_exact_key(text),'normalized_soft_key':normalized_soft_key(text),
            'provenance':row.get('provenance',[{'source_bundle':source_bundle,'source_dataset':str(source),'source_record_id':str(row.get('source_record_id',rid))}])}


def read_record_table(path,field_map=None):
    rows=read_rows(path)
    for row in rows:
        for canonical,source in (field_map or {}).items():
            if source in row:row[canonical]=row[source]
    if not rows or any('annotations' in r or 'triplets' in r for r in rows):return rows
    if not any('aspect_start' in r for r in rows):return rows
    ann_keys={'aspect_start','aspect_end','aspect_term','opinion_start','opinion_end','opinion_term','sentiment','taxonomy_label','aspect_category','entity_category','issue_category','implicit','factual','mention_type','relation','relation_label','relation_type'}
    meta_keys=set(rows[0])-ann_keys;grouped={}
    for row in rows:
        rid=str(pick(row,['id','record_id','review_id']))
        if rid not in grouped:grouped[rid]={k:row.get(k) for k in meta_keys};grouped[rid]['annotations']=[]
        grouped[rid]['annotations'].append({k:row.get(k) for k in ann_keys if k in row})
    return list(grouped.values())


def _resolve_inside(base,name):
    p=(base/name).resolve()
    if not p.is_relative_to(base) or not p.is_file():raise ContractError(f'Missing/unsafe manifest path {name}')
    return p


def _verify_checksum_file(root):
    files=list(root.rglob('checksums.sha256'))
    if len(files)!=1:raise ContractError('Expected exactly one checksums.sha256 in canonical V14 bundle')
    checksum_file=files[0];base=checksum_file.parent.resolve();checked=[]
    for line in checksum_file.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        expected_hash, name=line.split(maxsplit=1);name=name.lstrip('* ')
        if not re.fullmatch(r'[0-9a-fA-F]{64}',expected_hash):raise ContractError(f'Invalid checksum row: {line[:80]}')
        p=_resolve_inside(base,name)
        if sha256(p)!=expected_hash.lower():raise ContractError(f'Checksum mismatch: {name}')
        checked.append(name)
    if not checked:raise ContractError('checksums.sha256 is empty')
    return checksum_file,checked


def _find_manifest(root):
    candidates=[]
    for name in ('manifest_v14.json','dataset_manifest_v14.json','MANIFEST.json'):
        for p in root.rglob(name):
            try:m=json.loads(p.read_text(encoding='utf-8-sig'))
            except Exception:continue
            version=str(m.get('dataset_version',m.get('bundle_version',m.get('version','')))).lower()
            if version.startswith(('v14','14')):candidates.append((p,m))
    unique={p.resolve():(p,m) for p,m in candidates}
    if len(unique)!=1:raise ContractError(f'Expected exactly one V14 manifest, found {len(unique)}')
    return next(iter(unique.values()))


def _load_canonical_v14(manifest_path,manifest,override):
    base=manifest_path.parent.resolve();checksum_file,checked=_verify_checksum_file(base)
    # Manifest hashes are independently checked when present.
    for name,expected in manifest.get('output_file_hashes',{}).items():
        p=_resolve_inside(base,name)
        if sha256(p)!=expected:raise ContractError(f'Manifest checksum mismatch: {name}')
    record_patterns=[
        'data/core/train.jsonl','data/core/validation.jsonl','data/core/test.jsonl','data/core/unseen_test.jsonl',
        'data/auxiliary/silver_train.jsonl','data/auxiliary/synthetic_train.jsonl','data/auxiliary/ai_repaired_pending_human.jsonl',
        'data/diagnostic/frozen_synthetic_diagnostic.jsonl','data/human_review/target_human_review_queue.jsonl']
    record_files=[base/x for x in record_patterns if (base/x).is_file()]
    if not record_files:raise ContractError('Canonical V14 bundle contains no supported record files')
    policy=(override or {}).get('neutral_policy','factual_is_no_opinion')
    if policy!='factual_is_no_opinion':raise ContractError('Canonical V14 release requires neutral_policy=factual_is_no_opinion')
    records=[]
    for p in record_files:records.extend(canonical_record(r,policy,(override or {}).get('record_field_map')) for r in read_record_table(p))
    for r in records:r['relation_table_authoritative']=True
    ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)):raise ContractError('Record IDs must be unique across canonical record files')
    relation_paths=[base/f'data/relation/relation_{x}.csv' for x in ('train','validation','test')]+[base/'data/diagnostic/relation_frozen_diagnostic.csv']
    rels=[];byid={r['id']:r for r in records};seen={}
    labels={'0':0,'1':1,'not_related':0,'related':1,'hard_negative':0,'false':0,'true':1}
    for p in relation_paths:
        if not p.is_file():continue
        for rel in read_rows(p):
            rid=str(pick(rel,['record_id','review_id','id']))
            if rid not in byid:raise ContractError(f'Orphan relation row: {rid}')
            value=str(pick(rel,['relation_label','label'])).lower()
            if value not in labels:raise ContractError(f'Ambiguous relation label {value!r}')
            item=dict(rel);item['record_id']=rid;item['label']=labels[value]
            for kind in ('aspect','opinion'):
                s=_int_or_none(item.get(kind+'_start'),kind+'_start',rid);e=_int_or_none(item.get(kind+'_end'),kind+'_end',rid)
                if s is None or e is None or not 0<=s<e<=len(byid[rid]['text']):raise ContractError(f'{rid}: invalid relation {kind} offsets')
                item[kind+'_start'],item[kind+'_end']=s,e
                term=item.get(kind+'_term')
                if term and term!=byid[rid]['text'][s:e]:raise ContractError(f'{rid}: relation {kind} term mismatch')
            sig=(rid,)+tuple(item[k] for k in ('aspect_start','aspect_end','opinion_start','opinion_end'))
            if sig in seen and seen[sig]!=item['label']:raise ContractError(f'Contradictory explicit relation labels: {sig}')
            if sig in seen:continue
            seen[sig]=item['label'];item['sample_weight']=float(item.get('sample_weight',byid[rid]['sample_weight']));rels.append(item)
    meta={**manifest,'resolved_neutral_policy':policy,'record_count':len(records),'relation_count':len(rels),
          'manifest_sha256':sha256(manifest_path),'checksum_file':str(checksum_file),'checksums_verified':len(checked),
          'record_files':[str(p.relative_to(base)) for p in record_files],
          'relation_files':[str(p.relative_to(base)) for p in relation_paths if p.is_file()],
          'dataset_root':str(base),'adapter':'canonical_v14_production_bundle'}
    return records,rels,meta


def _load_legacy_v14(manifest_path,manifest,override):
    cfg={**manifest,**(override or {})};base=manifest_path.parent.resolve()
    version=str(cfg.get('dataset_version',cfg.get('version',''))).lower()
    if not version.startswith(('v14','14')):raise ContractError('Manifest must declare dataset_version V14')
    policy=cfg.get('neutral_policy')
    if policy not in {'factual_is_no_opinion','factual_is_neutral'}:raise ContractError('Manifest neutral_policy is required')
    records_path=cfg.get('records_path')
    if not records_path:raise ContractError('Manifest/override requires records_path')
    checksums=cfg.get('checksums',{})
    if not isinstance(checksums,dict):raise ContractError('checksums must map relative path to SHA256')
    declared=[records_path]+([cfg['relations_path']] if cfg.get('relations_path') else [])
    if strict_bool(cfg.get('require_checksums',True),'require_checksums') and any(x not in checksums for x in declared):raise ContractError('Every consumed data file requires SHA256')
    for name,expected in checksums.items():
        if sha256(_resolve_inside(base,name))!=expected:raise ContractError(f'Checksum mismatch: {name}')
    records=[canonical_record(r,policy,cfg.get('record_field_map')) for r in read_record_table(_resolve_inside(base,records_path),cfg.get('record_field_map'))]
    ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)):raise ContractError('Record IDs must be unique')
    rels=[];byid={r['id']:r for r in records};seen={};labels={'0':0,'1':1,'not_related':0,'related':1,'hard_negative':0}
    if cfg.get('relations_path'):
        for rel in read_rows(_resolve_inside(base,cfg['relations_path'])):
            item=dict(rel);rid=str(pick(item,['record_id','review_id','id']))
            if rid not in byid:raise ContractError(f'Orphan relation row: {rid}')
            value=str(pick(item,['relation_label','label'])).lower()
            if value not in labels:raise ContractError(f'Ambiguous relation label {value!r}')
            item['record_id']=rid;item['label']=labels[value]
            for kind in ('aspect','opinion'):
                s=_int_or_none(item.get(kind+'_start'),kind+'_start',rid);e=_int_or_none(item.get(kind+'_end'),kind+'_end',rid)
                if s is None or e is None or not 0<=s<e<=len(byid[rid]['text']):raise ContractError('Invalid explicit relation offsets')
                item[kind+'_start'],item[kind+'_end']=s,e
            sig=(rid,)+tuple(item[k] for k in ('aspect_start','aspect_end','opinion_start','opinion_end'))
            if sig in seen and seen[sig]!=item['label']:raise ContractError('Contradictory explicit relation labels')
            seen[sig]=item['label']
            if not item['label'] and not (byid[rid]['is_synthetic'] or byid[rid]['human_approved']):raise ContractError('Unreviewed real negatives are forbidden')
            rels.append(item)
    return records,rels,{**manifest,'resolved_neutral_policy':policy,'record_count':len(records),'relation_count':len(rels),'manifest_sha256':sha256(manifest_path),'dataset_root':str(base),'adapter':'legacy_v14_contract'}


def load_bundle(path,destination=None,contract_override=None):
    path=Path(path).expanduser()
    if not path.exists():raise ContractError(f'Dataset path does not exist: {path}')
    root=safe_extract(path,destination or path.with_suffix('')) if path.is_file() else path
    manifest_path,manifest=_find_manifest(root)
    if manifest_path.name=='MANIFEST.json' and (manifest_path.parent/'data').is_dir():
        return _load_canonical_v14(manifest_path,manifest,contract_override or {})
    return _load_legacy_v14(manifest_path,manifest,contract_override or {})


def group_keys(record,place_policy='warn'):
    keys=[('raw_exact',raw_exact_key(record['text'])),('record_id',str(record.get('id',record.get('record_id'))))]
    for field in ('family_id','parent_review_id','translation_parent_id','template_family_id','template_key','group_id'):
        value=record.get(field)
        if value not in (None,'','nan','None'):keys.append((field,str(value)))
    if place_policy=='hard' and record.get('place_id') not in (None,'','nan','None'):keys.append(('place_id',str(record['place_id'])))
    return keys


def _label_conflicts(records):
    output=[]
    bytext=defaultdict(list)
    for r in records:bytext[raw_exact_key(r['text'])].append(r)
    for key,items in bytext.items():
        sigs=defaultdict(list)
        for r in items:sigs[_annotation_signature(r)].append(r['id'])
        if len(sigs)>1:
            output.append({'raw_exact_sha256':hashlib.sha256(key.encode()).hexdigest(),'record_ids':sorted(r['id'] for r in items),
                           'signatures':[{'record_ids':sorted(v),'signature':repr(k)} for k,v in sigs.items()]})
    return output


def _near_pairs(train,protected,threshold):
    if not train or not protected or threshold is None:return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    corpus=[normalized_soft_key(r['text']) for r in protected]+[normalized_soft_key(r['text']) for r in train]
    try:matrix=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1).fit_transform(corpus)
    except ValueError:return []
    n=len(protected);nn=NearestNeighbors(metric='cosine',algorithm='brute',n_jobs=-1).fit(matrix[:n]);out=[]
    for lo in range(0,len(train),256):
        distances,indices=nn.radius_neighbors(matrix[n+lo:n+lo+256],radius=1-float(threshold),return_distance=True)
        for i,(ds,js) in enumerate(zip(distances,indices)):
            for d,j in zip(ds,js):
                if 1-float(d)>=threshold:out.append({'train_id':train[lo+i]['id'],'protected_id':protected[int(j)]['id'],'similarity':float(1-d)})
    return out


def audit_partitions(records,seed=42,tune_fraction=.10,calibration_fraction=.10,mode='quarantine',place_policy='warn',near_threshold=.985):
    if mode not in {'audit','quarantine','strict'}:raise ContractError(f'Unknown audit mode {mode}')
    if place_policy not in {'warn','hard','ignore'}:raise ContractError(f'Unknown place policy {place_policy}')
    byid={}
    for r in records:
        rid=str(r['id'])
        if rid in byid:raise ContractError(f'Duplicate canonical id before audit: {rid}')
        byid[rid]=r
    parent={rid:rid for rid in byid}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    owners={}
    for r in records:
        rid=str(r['id'])
        for key in group_keys(r,place_policy):
            if key in owners:union(rid,owners[key])
            else:owners[key]=rid
    components=defaultdict(list)
    for r in records:components[find(str(r['id']))].append(r)
    roles={};hard=[];warnings=[];excluded={}
    for gid,items in components.items():
        train=[r for r in items if r['split']=='train'];protected=[r for r in items if r['split']!='train'];splits=sorted({r['split'] for r in items})
        if train and protected:
            event={'kind':'train_protected_hard_group','group_id':gid,'splits':splits,'train_ids':sorted(r['id'] for r in train),'protected_ids':sorted(r['id'] for r in protected)};hard.append(event)
            if mode in {'quarantine','strict'}:
                for r in train:excluded[r['id']]={'reason':'hard_train_protected_group','event':event}
        if len({r['split'] for r in protected})>1:warnings.append({'kind':'protected_protected_overlap','group_id':gid,'splits':splits,'record_ids':sorted(r['id'] for r in protected)})
        for r in protected:roles[r['id']]=r['split']
    place_events=[]
    if place_policy=='warn':
        places=defaultdict(list)
        for r in records:
            if r.get('place_id') not in (None,'','nan','None'):places[str(r['place_id'])].append(r)
        for place,items in places.items():
            if any(r['split']=='train' for r in items) and any(r['split']!='train' for r in items):
                place_events.append({'kind':'declared_place_overlap_warning','place_id':place,'record_ids':sorted(r['id'] for r in items),'text_count':len({raw_exact_key(r['text']) for r in items})})
    label_conflicts=_label_conflicts(records)
    conflicting_ids={rid for event in label_conflicts for rid in event['record_ids']}
    for rid in conflicting_ids:
        if byid[rid]['split']=='train' and mode in {'quarantine','strict'}:excluded[rid]={'reason':'cross_source_label_conflict'}
    protected=[r for r in records if r['split']!='train']
    train=[r for r in records if r['split']=='train' and r['id'] not in excluded]
    soft=[];protected_soft=defaultdict(list)
    for r in protected:protected_soft[normalized_soft_key(r['text'])].append(r['id'])
    for r in train:
        key=normalized_soft_key(r['text'])
        if key in protected_soft and key!=raw_exact_key(r['text']).casefold():
            event={'kind':'normalized_soft_duplicate','train_id':r['id'],'protected_ids':sorted(protected_soft[key])};soft.append(event)
            if mode in {'quarantine','strict'}:excluded[r['id']]={'reason':'soft_normalized_overlap','event':event}
    train=[r for r in records if r['split']=='train' and r['id'] not in excluded]
    near=_near_pairs(train,protected,near_threshold)
    for event in near:
        soft.append({'kind':'near_duplicate',**event})
        if mode in {'quarantine','strict'}:excluded[event['train_id']]={'reason':'soft_near_duplicate','event':event}
    if mode=='strict' and (hard or soft or label_conflicts):raise ContractError(f'Train/protected leakage or label conflict: hard={len(hard)}, soft={len(soft)}, labels={len(label_conflicts)}')
    # Internal train roles stay component-stable and are assigned only after quarantine.
    for gid,items in components.items():
        active=[r for r in items if r['split']=='train' and r['id'] not in excluded]
        if not active:continue
        u=int(hashlib.sha256(f'{seed}|{gid}'.encode()).hexdigest()[:12],16)/16**12
        role='inner_tune' if u<tune_fraction else ('calibration' if u<tune_fraction+calibration_fraction else 'fit')
        for r in active:roles[r['id']]=role
    for rid in excluded:roles[rid]='train_excluded_leakage'
    return {'mode':mode,'place_policy':place_policy,'roles':roles,'hard_conflicts':hard,'soft_conflicts':soft,
            'evaluation_warnings':warnings+place_events,'label_conflicts':label_conflicts,
            'train_excluded':[{'record_id':rid,**data} for rid,data in sorted(excluded.items())],
            'counts':dict(Counter(roles.values())),'passed_for_training':not any(roles.get(r['id'])=='train' for r in records),
            'near_duplicate_threshold':near_threshold,
            'key_policy':{'hard_exact':'NFKC + whitespace collapse, punctuation/case retained','soft':'NFKC + casefold + conservative punctuation spacing','near':'char_wb 3-5 TF-IDF cosine'}}


def partitions(records,seed=42,tune_fraction=.10,calibration_fraction=.10,mode='strict',place_policy='hard',near_threshold=None):
    """Compatibility wrapper. V14.1 notebook calls audit_partitions explicitly."""
    return audit_partitions(records,seed,tune_fraction,calibration_fraction,mode,place_policy,near_threshold)['roles']


def deduplicate_fit_records(records):
    grouped=defaultdict(list)
    for r in records:grouped[raw_exact_key(r['text'])].append(r)
    kept=[];aliases=[];conflicts=[]
    for key,items in grouped.items():
        signatures=defaultdict(list)
        for r in items:signatures[_annotation_signature(r)].append(r)
        if len(signatures)>1:
            conflicts.append({'raw_exact_sha256':hashlib.sha256(key.encode()).hexdigest(),'record_ids':sorted(r['id'] for r in items),'reason':'different_annotation_signatures'})
            continue
        ordered=sorted(items,key=lambda r:(not r.get('human_approved',False),r.get('is_synthetic',False),-float(r.get('sample_weight',0)),r['source_dataset'],r['id']))
        winner=dict(ordered[0]);winner['dedup_provenance']=[]
        for r in ordered:
            winner['dedup_provenance'].extend(r.get('provenance',[]))
        winner['dedup_provenance']=list({json.dumps(x,sort_keys=True,ensure_ascii=False):x for x in winner['dedup_provenance']}.values())
        kept.append(winner)
        for r in ordered[1:]:aliases.append({'kept_id':winner['id'],'alias_id':r['id'],'source_dataset':r['source_dataset'],'raw_exact_sha256':hashlib.sha256(key.encode()).hexdigest()})
    return kept,aliases,conflicts


def leakage_report(train,protected,near_threshold=.985,place_policy='warn'):
    exact=[];keys={}
    for r in protected:
        for key in group_keys(r,'hard' if place_policy=='hard' else 'ignore'):keys.setdefault(key,[]).append(r['id'])
    for r in train:
        for key in group_keys(r,'hard' if place_policy=='hard' else 'ignore'):
            if key in keys:exact.append({'train_id':r['id'],'protected_ids':keys[key],'kind':key[0]})
    soft=[];ps=defaultdict(list)
    for r in protected:ps[normalized_soft_key(r['text'])].append(r['id'])
    for r in train:
        if normalized_soft_key(r['text']) in ps and raw_exact_key(r['text']) not in {raw_exact_key(x['text']) for x in protected}:
            soft.append({'train_id':r['id'],'protected_ids':ps[normalized_soft_key(r['text'])],'kind':'normalized_soft'})
    near=_near_pairs(train,protected,near_threshold)
    return {'exact_or_group':exact,'soft_duplicates':soft,'near_duplicates':near,'passed':not exact and not soft and not near,'place_policy':place_policy}


def safe_relation_pairs(records,explicit_relations=(),max_neg_per_positive=2):
    explicit=defaultdict(list)
    for x in explicit_relations:explicit[str(x['record_id'])].append(x)
    out=[]
    for r in records:
        if 'relation' not in r.get('tasks',['relation']):continue
        pos={tuple(a[k] for k in ('aspect_start','aspect_end','opinion_start','opinion_end')) for a in r['annotations'] if a.get('aspect_start') is not None and a.get('opinion_start') is not None and a.get('relation',True) is not False}
        authoritative=bool(r.get('relation_table_authoritative',False))
        labels={} if authoritative else {p:1 for p in pos}
        sources={} if authoritative else {p:'canonical_positive' for p in pos}
        weights={} if authoritative else {p:r.get('sample_weight',.25) for p in pos}
        for e in explicit[str(r['id'])]:
            p=tuple(int(e[k]) for k in ('aspect_start','aspect_end','opinion_start','opinion_end'));label=int(e['label'])
            if label==0 and not (r.get('is_synthetic') or r.get('human_approved')):raise ContractError('Negative SILVER relations cannot enter supervised training')
            if p in labels and labels[p]!=label:raise ContractError(f'{r["id"]}: relation table contradicts annotations')
            labels[p]=label;sources[p]=e.get('source','explicit_relation_table');weights[p]=float(e.get('sample_weight',r.get('sample_weight',.25)))
        human=r.get('human_approved',False) and not r.get('is_synthetic',False)
        if human and r.get('annotation_complete') is True:
            aspects=sorted({p[:2] for p in pos});opinions=sorted({p[2:] for p in pos})
            negatives=[a+o for a in aspects for o in opinions if a+o not in pos]
            negatives.sort(key=lambda p:(min(abs(p[0]-p[3]),abs(p[2]-p[1])),p))
            if r['split']=='train':negatives=negatives[:max_neg_per_positive*max(1,len(pos))]
            for p in negatives:labels.setdefault(p,0);sources.setdefault(p,'derived_human_complete');weights.setdefault(p,r.get('sample_weight',1.0))
        for p,label in sorted(labels.items()):
            out.append({'raw_text':r['text'],'spans':p,'label':label,'split':r['split'],'record_id':r['id'],'place_id':r.get('place_id'),
                        'family_id':r.get('family_id'),'source_dataset':r['source_dataset'],'relation_source':sources[p],
                        'human_gold':bool(human and r.get('annotation_complete')),'sample_weight':weights[p]})
    return out


def example_raw_text(item):
    raw=item.get('raw_text') or item.get('review_text_raw')
    if raw is not None:return str(raw)
    raw=str(item.get('marked_context') or item.get('text') or '')
    if '[CTX]' in raw:raw=raw.split('[CTX]',1)[1]
    return re.sub(r'\[/?(?:ASP|OPN|DOC|CTX)\]','',raw).strip()


def ensure_example_identity(item):
    out=dict(item);raw=example_raw_text(out)
    if out.get('record_id') in (None,''):
        if not raw:raise ContractError('Auxiliary example needs stable ID or review text')
        digest=hashlib.sha256(raw_exact_key(raw).encode()).hexdigest();out['record_id']='aux::'+digest;out['group_id']=out.get('group_id') or 'text::'+digest
    return out
