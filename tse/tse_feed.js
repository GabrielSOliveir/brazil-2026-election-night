window.TSE = window.TSE || {data:{}, list_:{}};
TSE.pad=(x,n)=>String(x).padStart(n,'0');
TSE.list=async(ele)=>{const e6=TSE.pad(ele,6); const cm=await (await fetch(`/oficial/ele2026/${ele}/config/mun-e${e6}-cm.json`,{cache:'no-cache'})).json(); const L=[];
 for(const uf of cm.abr) for(const m of uf.mu) for(const z of (m.z||[])) L.push([uf.cd,m.cd,z,m.nm]); return L;};
TSE.parse=(j)=>{const c=j.carg[0]; const v={}; for(const a of c.agr) for(const p of a.par) for(const k of p.cand) v[k.n]=+k.vap;
 return {ts:+j.s.ts, st:+j.s.st, te:+j.e.te, comp:+j.e.c, abst:+j.e.a, vb:+j.v.vb, vn:+j.v.tvn, vv:+j.v.vv, hg:j.hg, c:v};};
TSE.run=async(ele, conc=24)=>{const e6=TSE.pad(ele,6); TSE.list_[ele]=TSE.list_[ele]||await TSE.list(ele); const L=TSE.list_[ele]; const D=TSE.data[ele]=TSE.data[ele]||{}; let i=0, err=0, changed=0; const t0=Date.now();
 async function worker(){ while(i<L.length){ const [uf,mu,z,nm]=L[i++]; const key=`${uf}|${mu}|${z}`; const old=D[key]; if(old && old.st>=old.ts && old.ts>0) continue;
   try{ const r=await fetch(`/oficial/ele2026/${ele}/dados/${uf}/${uf}${mu}-z${z}-c0001-e${e6}-u.json`,{cache:'no-cache'}); if(!r.ok){err++; continue;} const p=TSE.parse(await r.json()); p.nm=nm;
     if(!old || old.st!==p.st || old.hg!==p.hg){D[key]=p; changed++;} }catch(e){err++;} } }
 await Promise.all(Array.from({length:conc},worker)); return {n:L.length, have:Object.keys(D).length, changed, err, secs:(Date.now()-t0)/1000};};
TSE.start=(ele)=>{TSE.job={done:false,started:Date.now()}; TSE.run(ele).then(r=>{TSE.job={done:true,...r}}).catch(e=>{TSE.job={done:true,error:String(e)}}); return 'started';};
TSE.csv=(ele,from,to)=>{const D=TSE.data[ele]; const ks=Object.keys(D).sort(); const out=[];
 for(const k of ks.slice(from,to)){const p=D[k]; const c=p.c; const main=['13','22','70','14','55']; let o=0; for(const n in c) if(!main.includes(n)) o+=c[n];
  out.push([k.replace(/\|/g,','),p.ts,p.st,p.te,p.comp,p.abst,p.vb,p.vn,p.vv,c['13']||0,c['22']||0,c['70']||0,c['14']||0,c['55']||0,o,p.hg].join(','));} return out.join('\n');};
TSE.save=(ele,name)=>{const hdr='uf,mun,zona,ts,st,te,comp,abst,vb,vn,vv,c13,c22,c70,c14,c55,cout,hg'; const s=hdr+'\n'+TSE.csv(ele,0,1e9); const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'})); a.download=name; document.body.appendChild(a); a.click(); a.remove(); return s.length;};
'ok'
TSE.sleep=(ms)=>new Promise(r=>setTimeout(r,ms));
TSE.get=async(u)=>{for(let k=0;k<5;k++){const r=await fetch(u,{cache:'no-cache'}); if(r.status===429||r.status>=500){await TSE.sleep(2000*(k+1)); continue;} if(!r.ok) return null; return await r.json();} return null;};
TSE.ufs=['ac','al','am','ap','ba','ce','df','es','go','ma','mg','ms','mt','pa','pb','pe','pi','pr','rj','rn','ro','rr','rs','sc','se','sp','to','zz'];
TSE.runAll=async(ele,conc=6)=>{const e6=TSE.pad(ele,6); TSE.list_[ele]=TSE.list_[ele]||await TSE.list(ele); const L=TSE.list_[ele]; const D=TSE.data[ele]=TSE.data[ele]||{}; let i=0,err=0; const t0=Date.now();
 async function w(){ while(i<L.length){ const [uf,mu,z,nm]=L[i++]; const k=`${uf}|${mu}|${z}`; if(D[k]&&D[k].ts>0&&D[k].st>=D[k].ts) continue; const j=await TSE.get(`/oficial/ele2026/${ele}/dados/${uf}/${uf}${mu}-z${z}-c0001-e${e6}-u.json`); if(!j){err++;continue;} const p=TSE.parse(j); p.nm=nm; D[k]=p; await TSE.sleep(40);} }
 await Promise.all(Array.from({length:conc},w)); return {n:L.length,have:Object.keys(D).length,err,secs:(Date.now()-t0)/1000};};
TSE.start=(ele)=>{TSE.job={done:false}; TSE.runAll(ele).then(r=>TSE.job={done:true,...r}).catch(e=>TSE.job={done:true,error:String(e)}); return 'started';};
TSE.der=(u8,s=0,e=u8.length,depth=0)=>{const out=[]; let i=s; while(i<e){ const tag=u8[i++]; let len=u8[i++]; if(len&0x80){const n=len&0x7f; len=0; for(let k=0;k<n;k++) len=len*256+u8[i++];} const vs=i, ve=i+len; if(ve>e) return null; const node={tag,vs,ve}; if(tag&0x20){ node.kids=TSE.der(u8,vs,ve,depth+1); if(node.kids===null) return null; } else if(tag===0x04 && len>2 && depth<12){ const k=TSE.der(u8,vs,ve,depth+1); if(k && k.length) node.kids=k; } out.push(node); i=ve;} return out;};
TSE.int=(u8,n)=>{let v=0; for(let i=n.vs;i<n.ve;i++) v=v*256+u8[i]; return v;};
TSE.bu=async(p,uf,mu,z,s)=>{const b=`/oficial/ele2026/arquivo-urna/${p}/dados/${uf}/${mu}/${z}/${s}/`; const j=await TSE.get(b+`p00${p}-${uf}-m${mu}-z${z}-s${s}-aux.json`); if(!j||!j.hashes) return null; const h=j.hashes[j.hashes.length-1]; const f=(h.arq||[]).find(x=>(x.tp||'')==='bu'); if(!f) return null; for(let k=0;k<4;k++){const r=await fetch(b+h.hash+'/'+f.nm); if(r.status===429){await TSE.sleep(2000*(k+1)); continue;} if(!r.ok) return null; return {u8:new Uint8Array(await r.arrayBuffer()), nh:j.hashes.length, st:h.st};} return null;};
TSE.extract=(u8,ele)=>{const t=TSE.der(u8); if(!t) return null; let res=null;
 const walk=(nodes,inEle)=>{ for(const n of nodes){ if(!n.kids) continue; const k=n.kids; let here=inEle;
   if(k.length>3 && k[0].tag===0x02 && TSE.int(u8,k[0])===ele && k[1].tag===0x02){ here=true; res=res||{aptos:TSE.int(u8,k[1]),c:{},b:0,n:0,comp:null}; }
   if(here && k.length>=2 && k[0].tag===0x0a && k[1].tag===0x02 && res && res.comp===null) res.comp=TSE.int(u8,k[1]);
   if(here && k.length>=3 && k[0].tag===0x81 && TSE.int(u8,k[0])===1 && k[1].tag===0x02 && k[2].tag===0x30){ for(const v of k[2].kids){ const vk=v.kids||[]; const tp=vk.find(x=>x.tag===0x81), q=vk.find(x=>x.tag===0x82), id=vk.find(x=>x.tag===0xa3); if(!tp||!q) continue; const tpv=TSE.int(u8,tp), qv=TSE.int(u8,q); if(tpv===1&&id) res.c[TSE.int(u8,id.kids[1])]=qv; else if(tpv===2) res.b+=qv; else if(tpv===3) res.n+=qv; } }
   walk(k,here);} };
 walk(t,false); return res;};
'ok-full'
