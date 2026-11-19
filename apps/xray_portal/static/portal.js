/* Browser-only scan intake. No cloud credentials or runtime GPU dependencies. */
'use strict';
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const bytes = (n) => { n=Number(n || 0); const u=['B','KiB','MiB','GiB','TiB']; let i=0; while(n>=1024 && i<4){n/=1024;i++;} return `${n.toFixed(i ? 1 : 0)} ${u[i]}`; };
const timeText = (v) => {if(!v)return '';const d=new Date(typeof v==='number'?v*1000:v);return Number.isNaN(d.getTime())?String(v):d.toLocaleString('en-GB',{timeZone:'Asia/Tokyo',hour12:false})+' JST';};
const label = (v) => String(v || 'Draft').replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
let session={}, batch=null, batchId=null, queue=[], active=0, reviewDirty=false, reviewRows=[], pairPage=0, reportSignature='', polling=false;
const PAGE_SIZE=8;
function notice(message, error=false){ $('notice').textContent=message; $('notice').className=error?'error':''; $('notice').hidden=false; }
async function api(path, body, method){
  const options={method:method || (body===undefined?'GET':'POST'),credentials:'same-origin',headers:{}};
  if(options.method!=='GET') options.headers['X-CSRF-Token']=session.csrf_token || '';
  if(body!==undefined){ options.headers['Content-Type']='application/json'; options.body=JSON.stringify(body); }
  const response=await fetch(path, options); let value;
  try{value=await response.json();}catch{value={detail:`HTTP ${response.status}`};}
  if(!response.ok){const detail=value.detail || value.error || value; throw new Error(typeof detail==='string'?detail:JSON.stringify(detail));}
  return value;
}
async function refreshSession(){session=await api('/api/session');$('admin-open').hidden=!!session.admin;$('logout').hidden=!session.admin && !session.batch_ids?.length;}
function path(suffix=''){return `/api/batches/${encodeURIComponent(batchId)}${suffix}`;}
async function refreshList(){
  const data=await api('/api/batches'); const list=Array.isArray(data)?data:(data.batches || []);
  $('batch-list').innerHTML=list.length?list.map(b=>`<button class="batch-choice ${b.id===batchId?'active':''}" data-batch="${esc(b.id)}">${esc(b.title || b.id)}<span>${esc(label(b.status))}</span></button>`).join(''):'<p class="muted small">Create a batch or enter an access code to get started.</p>';
}
async function openBatch(id){
  if(reviewDirty && !confirm('Leave this batch and discard unsaved review changes?')) return;
  if(queue.some(q=>q.running) && id!==batchId){notice('Pause uploads before switching batches.',true);return;}
  if(id!==batchId) queue=[];
  batchId=id; reviewDirty=false;reportSignature='';pairPage=0;
  $('create-section').hidden=true;$('batch-content').hidden=false;$('access-code-panel').hidden=true;
  renderQueue();await refreshBatch(true);await refreshList();
}
async function refreshBatch(force=false){
  if(!batchId) return;const current=batchId;
  const query=!force && batch?.id===current && batch.updated_at!=null?`?after=${encodeURIComponent(batch.updated_at)}`:'';
  const data=await api(path()+query);if(current!==batchId || data.unchanged)return;
  batch=data;renderBatch();
}
function renderBatch(){
  $('batch-id').textContent=`BATCH ${batch.id}`;$('batch-title').textContent=batch.title;
  $('batch-meta').textContent=`${batch.workpiece || ''} · ${batch.scan_group || ''} · Submitted by ${batch.uploader || '—'}`;
  $('batch-status').textContent=label(batch.status);$('batch-status').className='badge'+(/fail|error|reject|correction/.test(batch.status)?' error':'');
  const frozen=!!batch.version || /froz|sync|cloud_ready/.test(batch.status);
  $('upload-section').hidden=frozen;
  const files=batch.files || [];
  $('stored-files').innerHTML=files.length?`<div class="table-scroll"><table class="file-table"><thead><tr><th>File</th><th>Role / condition</th><th>Size</th><th>Status</th><th></th></tr></thead><tbody>${files.map(f=>`<tr><td>${esc(f.filename)}</td><td>${esc(f.role)} · ${esc(f.condition)}</td><td>${bytes(f.size)}</td><td>${esc(label(f.status))}</td><td>${!frozen && f.status!=='excluded'?`<button class="quiet" data-exclude-file="${esc(f.id)}">Exclude</button>`:''}</td></tr>`).join('')}</tbody></table></div>`:'<p class="muted small">No files registered yet.</p>';
  $('validate').disabled=!files.length || active>0 || ['validating','freezing','syncing'].includes(batch.status) || files.filter(f=>f.status!=='excluded').some(f=>['registered','uploading'].includes(f.status));
  $('stored-files').querySelectorAll('[data-exclude-file]').forEach(el=>el.onclick=safely(async()=>{const reason=prompt('Why should this upload be excluded? The original will be retained.');if(reason?.trim()){await api(path(`/files/${encodeURIComponent(el.dataset.excludeFile)}/exclude`),{reason:reason.trim()});await refreshBatch();}}));
  if(batch.error) notice(typeof batch.error==='string'?batch.error:JSON.stringify(batch.error),true);
  const report=batch.report;
  if(!report && reportSignature){reviewDirty=false;reviewRows=[];reportSignature='';$('review-save-state').textContent='Files changed. Run validation again and review the new pairing report.';}
  $('validation-section').hidden=!report;
  $('review-section').hidden=!report || !session.admin;
  if(report){
    const images=report.images || [], pairs=report.candidate_pairs || [], issues=report.issues || [];
    $('validation-summary').innerHTML=`<div class="summary-grid"><div class="stat"><b>${images.length}</b><span>Recognized TIFF images</span></div><div class="stat"><b>${pairs.length}</b><span>Candidate pairs</span></div><div class="stat"><b>${issues.length}</b><span>Items to review</span></div></div>`;
    $('validation-issues').innerHTML=issues.slice(0,80).map(i=>`<div class="issue ${['error','fatal'].includes(i.severity)?'bad':''}">${esc(typeof i==='string'?i: (i.message || i.detail || JSON.stringify(i)))}</div>`).join('')+(issues.length>80?`<p class="small muted">Showing 80 of ${issues.length} issues. Correct the source files before proceeding.</p>`:'');
    const conditions=new Map();images.forEach(im=>{const key=`${im.role} · ${im.condition}`;conditions.set(key,(conditions.get(key)||0)+1);});
    $('condition-summary').innerHTML=`<div class="table-scroll"><table class="file-table"><thead><tr><th>Role / condition</th><th>Usable images</th></tr></thead><tbody>${[...conditions].map(([k,n])=>`<tr><td>${esc(k)}</td><td>${n}</td></tr>`).join('')}</tbody></table></div>`;
    const signature=JSON.stringify([pairs,batch.decisions]);
    if(!reviewDirty && signature!==reportSignature){
      reportSignature=signature; const saved=new Map((batch.decisions?.pairs || []).map(p=>[p.id,p]));
      reviewRows=pairs.map(p=>({...p,...saved.get(p.id),approved:!!saved.get(p.id)?.approved,split:saved.get(p.id)?.split || 'exclude'}));
      $('split-mode').value=batch.decisions?.split_mode || 'group';$('guard-slices').value=batch.decisions?.guard_slices || 16;$('review-rationale').value=batch.decisions?.rationale || '';
      $('acknowledge-issues').checked=!!batch.decisions?.acknowledge_issues;
      renderGroups();renderPairs();updateSplitGuidance();
    }
  }
  $('freeze').disabled=reviewDirty || !batch.decisions || ['validating','freezing','syncing'].includes(batch.status);
  $('save-review').disabled=['validating','freezing','syncing'].includes(batch.status);
  $('reject').disabled=['validating','freezing','syncing'].includes(batch.status);
  $('version-section').hidden=!batch.version;
  if(batch.version){
    const v=typeof batch.version==='object'?batch.version:{id:batch.version};const manifest=v.manifest || {};const cloud=v.cloud || {};
    const entries=[['Version',v.id || v.version_id || v.version],['Status',label(batch.status)],['Archive SHA256',v.sha256 || v.archive_sha256 || manifest.sha256 || cloud.sha256],['Cloud data URI',v.gcs_uri || v.data_uri || v.archive_uri || cloud.data_uri || cloud.archive_uri || cloud.gcs_uri || cloud.uri],['Cloud generation',v.generation || cloud.generation],['Train / monitor / test',(v.splits || manifest.counts)?JSON.stringify(v.splits || manifest.counts):null]];
    $('version-info').innerHTML=`<dl class="version-kv">${entries.filter(([,v])=>v!=null).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>`;
    $('sync').hidden=!session.admin;$('sync').disabled=/syncing|cloud_ready/.test(batch.status);
    $('sync').textContent=batch.status==='cloud_ready'?'Cloud copy verified':'Sync approved version to GCS';
  }
  applyReviewLock();
  $('activity').innerHTML=(batch.events || []).slice(-20).reverse().map(e=>`<div class="event">${esc(e.message || e.action || e.event || JSON.stringify(e))}<time>${esc(timeText(e.created_at || e.timestamp || e.time))}</time></div>`).join('') || '<p class="muted small">No activity yet.</p>';
}
function reviewLocked(){return !!batch?.version || /froz|sync|cloud_ready/.test(batch?.status || '');}
function applyReviewLock(){
  const locked=reviewLocked();const busy=['validating','freezing','syncing'].includes(batch?.status);
  $('review-mode-label').textContent=locked?'Frozen review · read-only':'Reviewer only';
  $('review-heading').textContent=locked?'3. Frozen pairing review':'3. Review pairs and assign splits';
  $('review-section').querySelectorAll('input,select,textarea').forEach(el=>{if(el.id!=='pair-filter')el.disabled=locked || busy || (el.id==='guard-slices' && $('split-mode').value!=='spatial');});
  $('group-controls').querySelectorAll('button').forEach(el=>el.disabled=locked || busy);
  $('apply-range').disabled=locked || busy;
  for(const id of ['save-review','freeze','reject']){$(id).hidden=locked;if(locked)$(id).disabled=true;}
  $('group-controls').hidden=locked;
  $('range-controls').hidden=locked || $('split-mode').value!=='spatial';
  if(locked)$('review-save-state').textContent='Frozen review: mappings, split assignments and notes are read-only. Browse pairs or download original TIFFs to audit this version.';
}
function dirty(){if(reviewLocked())return;reviewDirty=true;$('freeze').disabled=true;$('review-save-state').textContent='Unsaved changes. Save the review before freezing.';}
function updateSplitGuidance(){const spatial=$('split-mode').value==='spatial';$('range-controls').hidden=!spatial;$('guard-slices').disabled=!spatial;$('split-guidance').textContent=spatial?'Use continuous slice blocks with an exclusion gap across every condition. This does not create an independent workpiece test. The server checks guard bands.':'Keep repeated scans and all capture conditions of the same workpiece in the same split. The server checks group and content overlap.';applyReviewLock();}
function renderGroups(){
  const groups=[...new Set(reviewRows.map(p=>p.group || 'Unspecified'))];
  $('range-group').innerHTML=groups.map(g=>`<option value="${esc(g)}">${esc(g)}</option>`).join('');
  $('group-controls').innerHTML=groups.map((g,i)=>`<div class="group-control"><strong>${esc(g)}</strong><select data-group-index="${i}" aria-label="Split for ${esc(g)}"><option value="">Assign entire group…</option><option value="train">Train</option><option value="monitor">Monitor</option><option value="test">Test</option><option value="exclude">Exclude</option></select><button class="secondary" data-approve-group="${i}">Approve mapped pairs</button></div>`).join('');
  $('group-controls').querySelectorAll('select').forEach(el=>el.onchange=()=>{const group=groups[Number(el.dataset.groupIndex)];reviewRows.filter(p=>(p.group || 'Unspecified')===group).forEach(p=>p.split=el.value);dirty();renderPairs();});
  $('group-controls').querySelectorAll('button').forEach(el=>el.onclick=()=>{const group=groups[Number(el.dataset.approveGroup)];if(!confirm(`Approve mapped pairs in ${group}? Confirm you have reviewed their alignment and GT quality.`))return;reviewRows.filter(p=>(p.group || 'Unspecified')===group && p.gt_id).forEach(p=>p.approved=true);dirty();renderPairs();});
  applyReviewLock();
}
function imagePanel(im,role){
  if(!im) return `<div class="image-panel"><strong>${role}</strong><div class="missing-image">No matching reference selected</div></div>`;
  return `<div class="image-panel"><strong>${role}</strong><a href="${path(`/preview/${encodeURIComponent(im.id)}`)}" target="_blank" rel="noopener"><img loading="lazy" src="${path(`/preview/${encodeURIComponent(im.id)}`)}" alt="${esc(role+' '+im.filename)}"></a><p><a href="${path(`/image/${encodeURIComponent(im.id)}`)}">Download original TIFF</a></p><p>${esc(im.filename)} · ${esc(im.condition)}<br>${esc(im.width)} × ${esc(im.height)} · ${esc(im.dtype)}<br>Range ${esc(im.min)}–${esc(im.max)} · Mean ${Number(im.mean || 0).toFixed(1)}<br>Axis ${esc(im.axis || 'unknown')} · Saturated ${(100*Number(im.saturated_fraction || 0)).toFixed(3)}%</p></div>`;
}
function renderPairs(){
  const filter=$('pair-filter').value;const filtered=reviewRows.filter(p=>filter==='all' || (filter==='pending'&&!p.approved) || (filter==='approved'&&p.approved) || (filter==='excluded'&&p.split==='exclude'));
  pairPage=Math.max(0,Math.min(pairPage,Math.ceil(filtered.length/PAGE_SIZE)-1));
  const images=new Map((batch.report?.images || []).map(im=>[im.id,im]));
  $('pair-count').textContent=`${filtered.length} pairs · Page ${pairPage+1} / ${Math.max(1,Math.ceil(filtered.length/PAGE_SIZE))}`;
  $('prev-pairs').disabled=pairPage===0;$('next-pairs').disabled=(pairPage+1)*PAGE_SIZE>=filtered.length;
  $('pair-list').innerHTML=filtered.slice(pairPage*PAGE_SIZE,(pairPage+1)*PAGE_SIZE).map(p=>{
    const lq=images.get(p.lq_id),gt=images.get(p.gt_id);const refs=[...images.values()].filter(im=>im.role==='GT' && im.workpiece===lq?.workpiece && im.scan_group===lq?.scan_group && im.axis===lq?.axis && im.slice_id===lq?.slice_id && im.width===lq?.width && im.height===lq?.height);
    const options=refs.map(im=>`<option value="${esc(im.id)}" ${im.id===p.gt_id?'selected':''}>${esc(im.filename+' · '+im.condition)}</option>`).join('');
    return `<article class="pair-card" data-pair="${esc(p.id)}"><div class="pair-title"><span><b>Slice ${esc(p.slice_id ?? lq?.slice_id ?? '?')}</b> · ${esc(p.group || '')}</span><span class="badge neutral">${esc(label(p.status || 'candidate'))}</span></div><div class="pair-images">${imagePanel(lq,'Noisy image (LQ)')}${imagePanel(gt,'Reference image (GT)')}</div><label>Reference mapping<select class="pair-image-select" data-field="gt_id"><option value="">Select reference…</option>${options}</select></label><div class="pair-actions"><label><input type="checkbox" data-field="approved" ${p.approved?'checked':''}>Pair checked and approved</label><label>Split<select data-field="split">${['exclude','train','monitor','test'].map(s=>`<option value="${s}" ${p.split===s?'selected':''}>${label(s)}</option>`).join('')}</select></label></div></article>`;
  }).join('') || '<p class="muted">No pairs match this filter.</p>';
  $('pair-list').querySelectorAll('[data-field]').forEach(el=>el.onchange=()=>{const row=reviewRows.find(p=>p.id===el.closest('[data-pair]').dataset.pair);row[el.dataset.field]=el.type==='checkbox'?el.checked:el.value;if(el.dataset.field==='gt_id'){row.approved=false;renderPairs();}dirty();});
  applyReviewLock();
}
function renderQueue(){
  $('upload-queue').innerHTML=queue.map(q=>`<div class="upload-item" data-queue="${q.key}"><div class="filename">${esc(q.file.name)} <span class="muted">${bytes(q.file.size)}</span></div><div class="upload-fields"><label>Role<select data-upload-field="role" ${q.started?'disabled':''}><option value="LQ" ${q.role==='LQ'?'selected':''}>Noisy (LQ)</option><option value="GT" ${q.role==='GT'?'selected':''}>Reference (GT)</option></select></label><label>Capture condition<input data-upload-field="condition" value="${esc(q.condition)}" ${q.started?'disabled':''} placeholder="300 kV / 40 µA" required></label><button class="secondary" data-upload-action="${q.running?'pause':'resume'}" ${q.done?'disabled':''}>${q.running?'Pause':q.done?'Transferred':'Resume / start'}</button></div><details class="upload-group-fields"><summary>Workpiece / scan group: ${esc(q.workpiece)} · ${esc(q.scan_group)}</summary><div class="grid2"><label>Workpiece ID<input data-upload-field="workpiece" value="${esc(q.workpiece)}" ${q.started?'disabled':''}></label><label>Scan group<input data-upload-field="scan_group" value="${esc(q.scan_group)}" ${q.started?'disabled':''}></label></div></details><div class="upload-progress"><progress value="${q.progress || 0}" max="100"></progress><span>${esc(q.state || 'Ready')}</span></div></div>`).join('');
  $('start-uploads').disabled=!queue.some(q=>!q.running&&!q.done);$('pause-all').disabled=!queue.some(q=>q.running);
  $('upload-queue').querySelectorAll('[data-upload-field]').forEach(el=>el.onchange=()=>{const q=queue.find(q=>q.key===el.closest('[data-queue]').dataset.queue);q[el.dataset.uploadField]=el.value;});
  $('upload-queue').querySelectorAll('[data-upload-action]').forEach(el=>el.onclick=async()=>{const q=queue.find(q=>q.key===el.closest('[data-queue]').dataset.queue);if(q.running)await pause(q);else{q.wanted=true;pump();}});
}
function updateProgress(q,state,progress){q.state=state;q.progress=progress;const row=$('upload-queue').querySelector(`[data-queue="${q.key}"]`);if(row){row.querySelector('progress').value=progress;row.querySelector('.upload-progress span').textContent=state;}}
async function fingerprint(q){
  const hash=XrayUpload.sha256.create();const chunk=8*1024*1024;
  for(let offset=0;offset<q.file.size;offset+=chunk){if(q.paused)throw new Error('Upload paused');hash.update(new Uint8Array(await q.file.slice(offset,offset+chunk).arrayBuffer()));updateProgress(q,`Checking original file · ${Math.round(Math.min(offset+chunk,q.file.size)/q.file.size*100)}%`,0);}
  return Array.from(hash.digest(),x=>x.toString(16).padStart(2,'0')).join('');
}
async function pause(q){q.paused=true;q.wanted=false;if(q.upload){await q.upload.abort();q.finish?.();}q.state='Paused — resume with this original file';renderQueue();}
function pump(){for(const q of queue){if(active>=2)break;if(q.wanted&&!q.running&&!q.done){q.wanted=false;runUpload(q);}}}
async function runUpload(q){
  if(!q.condition.trim()){notice(`Enter a capture condition for ${q.file.name}.`,true);return;}
  q.running=true;q.started=true;q.paused=false;active++;renderQueue();
  try{
    if(!q.fingerprint)q.fingerprint=await fingerprint(q);
    if(q.paused)return;
    const registered=await api(path('/files'),{filename:q.file.name,size:q.file.size,role:q.role,condition:q.condition.trim(),workpiece:q.workpiece,scan_group:q.scan_group,fingerprint:q.fingerprint});
    if(['uploaded','received'].includes(registered.status)){q.done=true;updateProgress(q,'Transferred — request file checks when all files are ready',100);return;}
    await new Promise(async(resolve,reject)=>{
      let settled=false;q.finish=()=>{if(!settled){settled=true;resolve();}};
      q.upload=new XrayUpload.Upload(q.file,{
        endpoint:'/uploads/',uploadUrl:registered.upload_url || undefined,chunkSize:32*1024*1024,retryDelays:[0,1000,3000,5000,10000],withCredentials:true,
        headers:{'X-CSRF-Token':session.csrf_token},metadata:{filename:q.file.name,filetype:q.file.type || 'application/octet-stream',file_id:registered.id,batch_id:batchId,fingerprint:q.fingerprint},
        fingerprint:()=>Promise.resolve(`xray:${batchId}:${registered.id}:${q.fingerprint}`),removeFingerprintOnSuccess:true,
        onProgress:(sent,total)=>updateProgress(q,`${bytes(sent)} / ${bytes(total)} · ${Math.round(sent/total*100)}%`,sent/total*100),
        onError:(err)=>{if(!settled){settled=true;reject(err);}},
        onSuccess:()=>{q.done=true;updateProgress(q,'Transferred — checksum and TIFF checks pending',100);q.finish();}
      });
      try{if(!registered.upload_url){const previous=await q.upload.findPreviousUploads();if(previous.length)q.upload.resumeFromPreviousUpload(previous[0]);}if(q.paused)q.finish();else q.upload.start();}catch(err){reject(err);}
    });
  }catch(err){q.state=q.paused?'Paused':`Transfer stopped: ${err.message}`;if(!q.paused)notice(`${q.file.name}: ${err.message}`,true);}
  finally{q.running=false;active--;q.finish=null;renderQueue();await refreshBatch().catch(()=>{});pump();}
}
async function action(name,body={}){const result=await api(path('/'+name),body);await refreshBatch();await refreshList();return result;}
function safely(fn){return async(e)=>{try{await fn(e);}catch(err){notice(err.message,true);}};}
$('batch-list').onclick=safely(async e=>{const button=e.target.closest('[data-batch]');if(button)await openBatch(button.dataset.batch);});
$('new-batch').onclick=()=>{$('create-section').hidden=false;$('create-section').scrollIntoView({behavior:'smooth'});};
$('create-form').onsubmit=safely(async e=>{e.preventDefault();const values=Object.fromEntries(new FormData(e.target));const created=await api('/api/batches',values);await refreshSession();await openBatch(created.id);$('access-code').textContent=created.access_code;$('access-code-panel').hidden=false;notice('Batch created. Save its access code, then add your scan files.');});
$('access-form').onsubmit=safely(async e=>{e.preventDefault();const values=Object.fromEntries(new FormData(e.target));values.batch_id=values.batch_id.trim().toLowerCase();values.access_code=values.access_code.trim();await api('/api/access',values);e.target.reset();await refreshSession();await openBatch(values.batch_id);});
$('copy-access').onclick=safely(async()=>{const text=`Batch: ${batchId}\nAccess code: ${$('access-code').textContent}`;if(navigator.clipboard && window.isSecureContext){await navigator.clipboard.writeText(text);notice('Batch ID and access code copied.');}else{const range=document.createRange();range.selectNodeContents($('access-code'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);notice(`Access code selected. Copy it manually and save the batch ID: ${batchId}`);}});
$('admin-open').onclick=()=>{if(active){notice('Pause uploads before switching to reviewer access.',true);return;}$('admin-dialog').showModal();};$('admin-cancel').onclick=()=>$('admin-dialog').close();
$('admin-form').onsubmit=safely(async e=>{e.preventDefault();await api('/api/admin/login',Object.fromEntries(new FormData(e.target)));e.target.reset();$('admin-dialog').close();await refreshSession();await refreshList();if(batchId)await refreshBatch(true);notice('Reviewer access enabled.');});
$('logout').onclick=safely(async()=>{if(active){notice('Pause uploads before signing out.',true);return;}await api('/api/logout',{});location.reload();});
$('file-select').onchange=()=>{for(const file of $('file-select').files){if(!/\.(tif|tiff|zip)$/i.test(file.name)){notice(`${file.name}: choose TIFF or ZIP.`,true);continue;}if(file.size===0){notice(`${file.name}: the file is empty.`,true);continue;}if(file.size>32*1024**3){notice(`${file.name}: exceeds the 32 GiB limit.`,true);continue;}queue.push({key:String(Date.now())+Math.random().toString(36).slice(2),file,role:/\bgt\b|reference/i.test(file.name)?'GT':'LQ',condition:file.name.replace(/\.(zip|tiff?)$/i,''),workpiece:batch.workpiece,scan_group:batch.scan_group,progress:0});}renderQueue();$('file-select').value='';};
$('start-uploads').onclick=()=>{queue.filter(q=>!q.done).forEach(q=>q.wanted=true);pump();};
$('pause-all').onclick=safely(async()=>{queue.forEach(q=>q.wanted=false);await Promise.all(queue.filter(q=>q.running).map(pause));});
$('validate').onclick=safely(async()=>{await action('validate');notice('File checks queued. This page will update when review is ready.');});
$('split-mode').onchange=()=>{updateSplitGuidance();dirty();};$('guard-slices').onchange=dirty;$('review-rationale').oninput=dirty;
$('pair-filter').onchange=()=>{pairPage=0;renderPairs();};$('prev-pairs').onclick=()=>{pairPage--;renderPairs();};$('next-pairs').onclick=()=>{pairPage++;renderPairs();};
$('apply-range').onclick=safely(async()=>{const start=Number($('range-start').value),end=Number($('range-end').value);if(!Number.isInteger(start)||!Number.isInteger(end)||start<0||end<start)throw new Error('Enter a valid inclusive slice range.');const group=$('range-group').value,split=$('range-split').value;const affected=reviewRows.filter(p=>(p.group || 'Unspecified')===group && Number(p.slice_id)>=start && Number(p.slice_id)<=end);affected.forEach(p=>p.split=split);dirty();renderPairs();$('range-summary').textContent=`Assigned ${affected.length} pairs across all conditions to ${label(split)}. Mapping approval is separate; excluded guard slices stay out of the dataset.`;});
$('save-review').onclick=safely(async()=>{const body={pairs:reviewRows.map(p=>({id:p.id,gt_id:p.gt_id || null,approved:!!p.approved,split:p.split})),split_mode:$('split-mode').value,guard_slices:Number($('guard-slices').value),rationale:$('review-rationale').value,acknowledge_issues:$('acknowledge-issues').checked};await api(path('/review'),body);reviewDirty=false;reportSignature='';await refreshBatch();$('review-save-state').textContent='Review saved. Freeze when the approved pairs and splits are final.';notice('Review saved. No cloud upload or training has started.');});
$('freeze').onclick=safely(async()=>{if(reviewDirty)throw new Error('Save the review before freezing.');if(!confirm('Freeze this reviewed dataset version? Later changes require a new version. No training will start.'))return;await action('freeze');notice('Dataset preparation queued. The server will check split isolation and package only training and monitoring images.');});
$('sync').onclick=safely(async()=>{if(!confirm('Copy this approved training version to the existing private Tokyo GCS bucket? Originals stay on NFS.'))return;await action('sync');notice('Cloud sync queued. Wait for the verified Cloud ready state.');});
$('reject').onclick=safely(async()=>{const reason=prompt('What should the uploader correct or add?');if(reason?.trim()){await action('reject',{reason:reason.trim()});notice('Correction request recorded.');}});
// Issue acknowledgement is separate from approving individual pair mappings.
const acknowledgement=document.createElement('label');acknowledgement.className='small';acknowledgement.innerHTML='<input id="acknowledge-issues" type="checkbox"> I reviewed the file-check issues and excluded or resolved affected pairs.';$('review-rationale').parentElement.before(acknowledgement);$('acknowledge-issues').onchange=dirty;
window.addEventListener('beforeunload',e=>{if(active || reviewDirty){e.preventDefault();e.returnValue='';}});
(async()=>{try{await refreshSession();await refreshList();}catch(err){notice(`Cannot load portal: ${err.message}`,true);}})();
setInterval(async()=>{if(polling || document.hidden)return;polling=true;try{if(batchId)await refreshBatch();}catch(err){notice(`Connection interrupted: ${err.message}. Uploads can be resumed.`,true);}finally{polling=false;}},5000);
