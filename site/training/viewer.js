'use strict';
const el=id=>document.getElementById(id), pageSize=8;
let snapshot, tags=[], pageIndex=0, generation=0;
const resizeObserver=new ResizeObserver(entries=>{for(const {target} of entries){if(target.isConnected&&target.data)Plotly.Plots.resize(target);}});
const colors={'024':'#176bb0','035':'#c25a24'};
function smooth(points,weight){
  if(weight===0)return points.map(p=>p[1]);
  let accum=0,mass=0;
  return points.map(([,value])=>{
    if(value===null){accum=0;mass=0;return null;}
    accum=weight*accum+(1-weight)*value;mass=weight*mass+(1-weight);
    return accum/mass;
  });
}
function render(){
  const token=++generation;
  const query=el('search').value.trim().toLowerCase(),group=el('group').value;
  const runs=['024','035'].filter(r=>el('run'+r).checked);
  const filtered=tags.filter(t=>(!group||t.split('/')[0]===group)&&t.toLowerCase().includes(query)&&runs.some(r=>snapshot.runs[r][t]));
  const pages=Math.ceil(filtered.length/pageSize);pageIndex=Math.min(pageIndex,Math.max(0,pages-1));
  el('count').textContent=`${filtered.length} of ${tags.length} metrics · ${runs.length} runs selected`;
  el('page').textContent=pages?`${pageIndex+1} / ${pages}`:'0 / 0';
  el('prev').disabled=pageIndex===0;el('next').disabled=pageIndex+1>=pages;
  const weight=Number(el('smoothing').value);
  el('smoothValue').textContent=weight.toFixed(2)+(weight===0?' · raw':' · EMA');
  resizeObserver.disconnect();
  for(const node of document.querySelectorAll('.plot'))Plotly.purge(node);
  el('charts').replaceChildren();
  if(!filtered.length){el('charts').textContent='No metrics match. Select a run or change the filters.';return;}
  for(const tag of filtered.slice(pageIndex*pageSize,(pageIndex+1)*pageSize)){
    const card=document.createElement('article');card.className='card';
    const heading=document.createElement('h2');heading.textContent=tag;card.append(heading);
    const available=['024','035'].filter(r=>snapshot.runs[r][tag]);
    if(available.length===1){const badge=document.createElement('span');badge.className='badge';badge.textContent=`${available[0]} only · not logged by the other run`;card.append(badge);}
    const plot=document.createElement('div');plot.className='plot';plot.setAttribute('aria-label',tag);card.append(plot);el('charts').append(card);
    const traces=runs.filter(r=>snapshot.runs[r][tag]).map(r=>{
      const points=snapshot.runs[r][tag];
      return {name:r,x:points.map(p=>p[0]),y:smooth(points,weight),customdata:points.map(p=>p[1]),type:'scatter',mode:'lines',connectgaps:false,line:{color:colors[r],width:1.5},hovertemplate:'Step %{x}<br>Value %{y:.5g}<br>Raw %{customdata:.5g}<extra>'+r+'</extra>'};
    });
    Plotly.newPlot(plot,traces,{margin:{l:65,r:20,t:20,b:52},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'system-ui, sans-serif',size:11,color:'#526674'},xaxis:{title:{text:'Logged training step'},gridcolor:'#edf1f4',zeroline:false},yaxis:{gridcolor:'#edf1f4',zeroline:false,automargin:true},legend:{orientation:'h',x:0,y:1.12},hovermode:'x unified',dragmode:'zoom'},{responsive:true,displaylogo:false,scrollZoom:false,toImageButtonOptions:{format:'png',filename:tag.replaceAll('/','_'),scale:2}}).then(()=>{if(token===generation&&plot.isConnected)resizeObserver.observe(plot);}).catch(error=>{if(token===generation){el('error').hidden=false;el('error').textContent=error.message;}});
  }
}
async function init(){
 try{
  const response=await fetch('scalars.json');if(!response.ok)throw Error('Could not load scalar snapshot: '+response.status);
  snapshot=await response.json();if(Object.keys(snapshot.runs).sort().join(',')!=='024,035')throw Error('Unexpected run selection');
  tags=[...new Set(Object.values(snapshot.runs).flatMap(s=>Object.keys(s)))].sort();
  for(const g of [...new Set(tags.map(t=>t.split('/')[0]))].sort()){const option=document.createElement('option');option.value=g;option.textContent=g;el('group').append(option);}
  el('summary').textContent=`${tags.length} metrics · 024: ${Object.keys(snapshot.runs['024']).length} · 035: ${Object.keys(snapshot.runs['035']).length} · Snapshot ${new Date(snapshot.exported_at).toISOString().slice(0,10)}`;
  for(const run of ['024','035']){let min=Infinity,max=-Infinity,count=0;for(const points of Object.values(snapshot.runs[run]))for(const [step] of points){min=Math.min(min,step);max=Math.max(max,step);count++;}const p=document.createElement('p');p.textContent=`${run}: steps ${min.toLocaleString()}–${max.toLocaleString()} · ${count.toLocaleString()} displayed points across all metrics`;el('coverage').append(p);}
  for(const id of ['search','group','run024','run035'])el(id).addEventListener(id==='search'?'input':'change',()=>{pageIndex=0;render();});
  el('smoothing').addEventListener('input',()=>{const w=Number(el('smoothing').value);el('smoothValue').textContent=w.toFixed(2)+(w===0?' · raw':' · EMA');});
  el('smoothing').addEventListener('change',render);
  el('prev').addEventListener('click',()=>{pageIndex--;render();});el('next').addEventListener('click',()=>{pageIndex++;render();});
  render();
 }catch(error){el('summary').textContent='Snapshot could not be loaded.';el('error').hidden=false;el('error').textContent=error.message;}
}
init();
