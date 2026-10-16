
  (() => {
    const root = document.getElementById('pcb-component-results');
    const data = JSON.parse(root.querySelector('#pcb-component-data').textContent);
    const get = id => root.querySelector('#' + id);
    const labels = {type:'类型',Name:'Name',value:'value',bbox:'框位置'};
    const typeName = type => data.typeLabels[type] || type;
    const esc = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    const value = item => item === null || item === undefined ? '<code>null</code>' : esc(item);
    const pct = ratio => (100 * ratio).toFixed(2) + '%';
    const state = {caseId:'1',type:'all',page:0,selected:null,tab:'images'};
    const saved = window.openai?.widgetState;
    const savedState = saved?.privateContent || saved;
    if (savedState && typeof savedState === 'object') {
      if (['all', ...data.cases.map(c => c.id)].includes(savedState.caseId)) state.caseId = savedState.caseId;
      if (['all', ...data.types.map(t => t.type)].includes(savedState.type)) state.type = savedState.type;
      if (['images','types','items'].includes(savedState.tab)) state.tab = savedState.tab;
    }
    const remember = () => {
      if (window.openai?.setWidgetState) window.openai.setWidgetState({
        modelContent:{submission:8422,view:state.tab,caseId:state.caseId,type:state.type,component:state.selected},
        privateContent:{caseId:state.caseId,type:state.type,tab:state.tab}
      }).catch(() => {});
    };
    function activate(tab, save=true) {
      state.tab = tab;
      for (const name of ['images','types','items']) {
        const button = get('pcb-tab-' + name);
        button.classList.toggle('active', name === tab);
        button.setAttribute('aria-selected', String(name === tab));
        get('pcb-' + name).hidden = name !== tab;
      }
      if (save) remember();
    }
    for (const tab of ['images','types','items']) get('pcb-tab-' + tab).addEventListener('click', () => activate(tab));
    get('pcb-image-rows').innerHTML = data.cases.map(c => `<tr>
      <td><button class="btn btn-ghost" type="button" data-case="${c.id}">第 ${c.id} 图</button></td>
      <td class="numeric">${c.pass}</td><td class="numeric">${c.mismatch}</td><td class="numeric">${c.ordinaryMissing}</td>
      <td class="numeric">${c.powerMissing}</td><td class="numeric">${pct(c.f1)}</td></tr>`).join('');
    get('pcb-type-rows').innerHTML = data.types.map(t => `<tr>
      <td><button class="btn btn-ghost" type="button" data-type="${esc(t.type)}">${esc(typeName(t.type))}</button>
      <div class="meter" role="img" aria-label="${esc(typeName(t.type))}提交${t.pred}个，通过${t.pass}个，失配${t.mismatch}个，多出${t.extra}个">
        <span class="pass" style="width:${100*t.pass/t.pred}%"></span><span class="mismatch" style="width:${100*t.mismatch/t.pred}%"></span>
        <span class="ordinary-missing" style="width:${100*t.extra/t.pred}%"></span></div></td>
      <td class="numeric">${t.pred}</td><td class="numeric">${t.pass}</td><td class="numeric">${t.mismatch}</td>
      <td class="numeric">${t.extra}</td><td class="numeric">${pct(t.pass/t.pred)}</td></tr>`).join('');
    get('pcb-case-select').innerHTML = '<option value="all">全部 13 张图</option>' + data.cases.map(c => `<option value="${c.id}">第 ${c.id} 图</option>`).join('');
    get('pcb-type-select').innerHTML = '<option value="all">全部类别</option>' + data.types.map(t => `<option value="${esc(t.type)}">${esc(typeName(t.type))}</option>`).join('');
    function detail(index) {
      const row = data.items[index];
      state.selected = index;
      const panel = get('pcb-selected-detail');
      panel.hidden = false;
      const boxText = row.bbox.map(x => Number(x).toFixed(2)).join(', ');
      const boxResult = row.status === 'extra' ? '位号未匹配，无对应框误差' : row.maxError !== null
        ? `最大坐标误差 ${Number(row.maxError).toFixed(3)} px；阈值 20 px`
        : '框通过本轮 20 px 检查；报告未列具体误差';
      panel.innerHTML = `<div>第 ${esc(row.caseId)} 图 · ${esc(row.id)} · ${row.status==='pass'?'完全通过':row.status==='extra'?'多出位号':'字段或框失配'}</div>
        <dl class="detail-fields"><div><dt>预测类型</dt><dd>${esc(row.type)}</dd></div><div><dt>Name</dt><dd>${value(row.name)}</dd></div>
        <div><dt>value</dt><dd>${value(row.value)}</dd></div><div><dt>失败项</dt><dd>${row.failed.length?row.failed.map(k=>esc(labels[k]||k)).join('、'):row.status==='extra'?'key 未匹配':'无'}</dd></div></dl>
        <div class="box-values">bbox [${boxText}]</div><div class="text-muted">${boxResult}</div>`;
      remember();
    }
    function render() {
      get('pcb-case-select').value = state.caseId;
      get('pcb-type-select').value = state.type;
      const filtered = data.items.map((row,index)=>({...row,index})).filter(row =>
        (state.caseId==='all'||row.caseId===state.caseId)&&(state.type==='all'||row.type===state.type));
      const pass=filtered.filter(x=>x.status==='pass').length;
      const mismatch=filtered.filter(x=>x.status==='mismatch').length;
      const extra=filtered.filter(x=>x.status==='extra').length;
      get('pcb-filter-summary').textContent=`提交 ${filtered.length} 个 · 完全通过 ${pass} · 字段/框失配 ${mismatch} · 多出位号 ${extra}`;
      const pages=Math.max(1,Math.ceil(filtered.length/10));
      state.page=Math.max(0,Math.min(state.page,pages-1));
      const shown=filtered.slice(state.page*10,state.page*10+10);
      get('pcb-item-rows').innerHTML=shown.length?shown.map(row=>`<tr>
        <td><span class="text-small text-muted">图 ${esc(row.caseId)}</span> <button class="btn btn-ghost" type="button" data-item="${row.index}">${esc(row.id)}</button></td>
        <td>${esc(typeName(row.type))}</td><td>${value(row.name)}</td><td>${value(row.value)}</td>
        <td>${row.status==='pass'?'通过':row.status==='extra'?'位号多出':row.failed.map(k=>esc(labels[k]||k)).join('、')}</td></tr>`).join(''):'<tr><td colspan="5">当前选择没有提交元件</td></tr>';
      get('pcb-page-label').textContent=`${state.page+1} / ${pages} 页`;
      get('pcb-previous').disabled=state.page===0;
      get('pcb-next').disabled=state.page===pages-1;
      get('pcb-selected-detail').hidden=true;
    }
    get('pcb-case-select').addEventListener('change',event=>{state.caseId=event.target.value;state.page=0;state.selected=null;render();remember();});
    get('pcb-type-select').addEventListener('change',event=>{state.type=event.target.value;state.page=0;state.selected=null;render();remember();});
    get('pcb-previous').addEventListener('click',()=>{state.page--;render();});
    get('pcb-next').addEventListener('click',()=>{state.page++;render();});
    root.addEventListener('click',event=>{
      const element=event.target.closest('button');
      if(!element)return;
      if(element.dataset.case){state.caseId=element.dataset.case;state.type='all';state.page=0;state.selected=null;render();activate('items');}
      if(element.dataset.type){state.caseId='all';state.type=element.dataset.type;state.page=0;state.selected=null;render();activate('items');}
      if(element.dataset.item!==undefined)detail(Number(element.dataset.item));
    });
    window.addEventListener('openai:set_globals',event=>{
      const next=event.detail?.globals?.widgetState?.privateContent;
      if(!next)return;
      if(next.caseId===state.caseId && next.type===state.type && next.tab===state.tab)return;
      if(['all',...data.cases.map(c=>c.id)].includes(next.caseId))state.caseId=next.caseId;
      if(['all',...data.types.map(t=>t.type)].includes(next.type))state.type=next.type;
      if(['images','types','items'].includes(next.tab))state.tab=next.tab;
      state.page=0;render();activate(state.tab,false);
    });
    render();activate(state.tab,false);
  })();
  