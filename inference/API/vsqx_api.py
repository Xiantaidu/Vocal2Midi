from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import numpy as np
from inference.LyricFA.tools.ZhG2p import ZhG2p
from inference.API.ustx_api import RmvpeResult, _to_ticks, _PitchPoint, _append_smoothed_points
PPQ=480
_PITCH_MAX=8191
_DEFAULT_PBS=2
_MIN_SECTION_BREAK=480
_TICK_STEP=5
_NOTE_PITCH_LEAD_IN=40
_G2P=None
def _t(s,b): return int(round(s*b*8.0))
def _vsqx_lyric(value):
    global _G2P
    value=(value or 'a').strip()
    if any(0x4E00<=ord(ch)<=0x9FFF for ch in value):
        if _G2P is None: _G2P=ZhG2p('mandarin')
        value=_G2P.convert(value,include_tone=False,convert_number=True)
    return value.split()[0] if value.split() else 'a'


def _build_pitch_data(notes,rmvpe,tempo):
    if not notes or rmvpe.midi_pitch.size==0:
        return []
    ns=sorted(notes,key=lambda n:n.onset)
    xs=[]; ys=[]
    pending: list[_PitchPoint]=[]
    pending_ni=-1
    ni=0

    def flush_pending(note_index):
        if not pending or note_index < 0:
            return
        local_xs=[]; local_ys=[]
        _append_smoothed_points(local_xs,local_ys,pending)
        if not local_xs:
            return
        for x,y in zip(local_xs,local_ys):
            if xs and xs[-1]==x:
                ys[-1]=y
            else:
                xs.append(x); ys.append(y)

    for i,mp in enumerate(rmvpe.midi_pitch):
        t=i*rmvpe.time_step_seconds
        while ni+1<len(ns) and ns[ni].offset<=t: ni+=1
        if ni>=len(ns): break
        n=ns[ni]
        if pending and pending_ni!=ni:
            flush_pending(pending_ni)
            pending.clear()
            pending_ni=-1
        if not(n.onset<=t<n.offset): continue
        if rmvpe.voiced_mask is not None:
            if i>=rmvpe.voiced_mask.size or not bool(rmvpe.voiced_mask[i]):
                continue
        if np.isnan(mp): continue
        dur=max(0.0,n.offset-n.onset)
        off=t-n.onset
        et=min(0.025,dur*0.15)
        if dur>et*2 and(off<et or dur-off<=et): continue
        x=int(round(_to_ticks(t,tempo)/_TICK_STEP)*_TICK_STEP)
        note_num=int(np.clip(round(n.pitch),0,127))
        y=int(round(np.clip((float(mp)-note_num)*100.0,-1200,1200)))
        pending_ni=ni
        if pending and pending[-1].x==x:
            pending[-1]=_PitchPoint(x=x,y=y)
        else:
            pending.append(_PitchPoint(x=x,y=y))
    flush_pending(pending_ni)
    if not xs: return []
    first_note_tick=_to_ticks(ns[0].onset,tempo)
    last_note_tick=max(_to_ticks(n.offset,tempo) for n in ns)
    leading_zero_ticks={
        max(0,((first_note_tick-_NOTE_PITCH_LEAD_IN)//_TICK_STEP)*_TICK_STEP),
        max(0,xs[0]-_TICK_STEP),
    }
    for tick in sorted((t for t in leading_zero_ticks if t<xs[0]),reverse=True):
        xs.insert(0,tick)
        ys.insert(0,0)
    trailing_zero_ticks={xs[-1]+_TICK_STEP,last_note_tick}
    for tick in sorted(t for t in trailing_zero_ticks if t>xs[-1]):
        xs.append(tick)
        ys.append(0)
    events=[]
    sections=[]
    for idx in range(len(xs)):
        if sections and xs[idx]-xs[idx-1]>=_MIN_SECTION_BREAK:
            sections.append([])
        if not sections: sections.append([])
        sections[-1].append((xs[idx],ys[idx]))
    for sec in sections:
        max_abs=max(abs(v) for _,v in sec)
        pbs=max(_DEFAULT_PBS,int(np.ceil(max_abs/100.0)))
        if pbs>_DEFAULT_PBS:
            events.append(('S',sec[0][0],pbs))
        for tick,cents in sec:
            pit=int(round(cents*_PITCH_MAX/(pbs*100.0)))
            pit=max(-_PITCH_MAX,min(_PITCH_MAX,pit))
            events.append(('P',tick,pit))
        if pbs>_DEFAULT_PBS:
            reset_tick=sec[-1][0]+_MIN_SECTION_BREAK//2
            events.append(('S',reset_tick,_DEFAULT_PBS))
    return events


def save_vsqx(notes:list[Any],filepath:Path,tempo:float=120.0,language:str='zh',rmvpe_result:RmvpeResult|None=None)->None:
    filepath=Path(filepath)
    valid=[]
    for n in notes:
        vals=[getattr(n,k,math.nan) for k in ('onset','offset','pitch')]
        if all(np.isfinite(v) for v in vals) and vals[1]>vals[0]: valid.append(n)
    valid.sort(key=lambda n:n.onset)
    end=max((_t(n.offset,tempo) for n in valid),default=PPQ)
    pitch_events=_build_pitch_data(valid,rmvpe_result,tempo) if rmvpe_result is not None else []
    print(f'[VSQX] rmvpe_result={rmvpe_result is not None}, pitch_events={len(pitch_events)}')
    P=[]
    P.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    P.append('<vsq4 xmlns="http://www.yamaha.co.jp/vocaloid/schema/vsq4/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.yamaha.co.jp/vocaloid/schema/vsq4/ vsq4.xsd">')
    P.append('\t<vender><![CDATA[Yamaha corporation]]></vender>')
    P.append('\t<version><![CDATA[4.0.0.3]]></version>')
    P.append('\t<vVoiceTable>')
    P.append('\t\t<vVoice>')
    P.append('\t\t\t<bs>0</bs>')
    P.append('\t\t\t<pc>0</pc>')
    P.append('\t\t\t<id><![CDATA[BCXDC6CZLSZHZCB4]]></id>')
    P.append('\t\t\t<name><![CDATA[VY2V3]]></name>')
    P.append('\t\t\t<vPrm>')
    for x in ('bre','bri','cle','gen','ope'): P.append(f'\t\t\t\t<{x}>0</{x}>')
    P.append('\t\t\t</vPrm>')
    P.append('\t\t</vVoice>')
    P.append('\t</vVoiceTable>')
    P.append('\t<mixer>')
    P.append('\t\t<masterUnit>')
    P.append('\t\t\t<oDev>0</oDev>')
    P.append('\t\t\t<rLvl>0</rLvl>')
    P.append('\t\t\t<vol>0</vol>')
    P.append('\t\t</masterUnit>')
    P.append('\t\t<vsUnit>')
    P.append('\t\t\t<tNo>0</tNo>')
    P.append('\t\t\t<iGin>0</iGin>')
    P.append('\t\t\t<sLvl>-898</sLvl>')
    P.append('\t\t\t<sEnable>0</sEnable>')
    P.append('\t\t\t<m>0</m>')
    P.append('\t\t\t<s>0</s>')
    P.append('\t\t\t<pan>64</pan>')
    P.append('\t\t\t<vol>0</vol>')
    P.append('\t\t</vsUnit>')
    P.append('\t\t<monoUnit>')
    P.append('\t\t\t<iGin>0</iGin>')
    P.append('\t\t\t<sLvl>-898</sLvl>')
    P.append('\t\t\t<sEnable>0</sEnable>')
    P.append('\t\t\t<m>0</m>')
    P.append('\t\t\t<s>0</s>')
    P.append('\t\t\t<pan>64</pan>')
    P.append('\t\t\t<vol>0</vol>')
    P.append('\t\t</monoUnit>')
    P.append('\t\t<stUnit>')
    P.append('\t\t\t<iGin>0</iGin>')
    P.append('\t\t\t<m>0</m>')
    P.append('\t\t\t<s>0</s>')
    P.append('\t\t\t<vol>-129</vol>')
    P.append('\t\t</stUnit>')
    P.append('\t</mixer>')
    P.append('\t<masterTrack>')
    P.append('\t\t<seqName><![CDATA[Untitled0]]></seqName>')
    P.append('\t\t<comment><![CDATA[Generated by Vocal2Midi]]></comment>')
    P.append(f'\t\t<resolution>{PPQ}</resolution>')
    P.append('\t\t<preMeasure>1</preMeasure>')
    P.append('\t\t<timeSig>')
    P.append('\t\t\t<m>0</m>')
    P.append('\t\t\t<nu>4</nu>')
    P.append('\t\t\t<de>4</de>')
    P.append('\t\t</timeSig>')
    P.append('\t\t<tempo>')
    P.append('\t\t\t<t>0</t>')
    P.append(f'\t\t\t<v>{int(round(tempo*100))}</v>')
    P.append('\t\t</tempo>')
    P.append('\t</masterTrack>')
    P.append('\t<vsTrack>')
    P.append('\t\t<tNo>0</tNo>')
    P.append('\t\t<name><![CDATA[Track 1]]></name>')
    P.append('\t\t<comment><![CDATA[Track]]></comment>')
    P.append('\t\t<vsPart>')
    start=PPQ*4
    P.append(f'\t\t\t<t>{start}</t>')
    P.append(f'\t\t\t<playTime>{end}</playTime>')
    P.append(f'\t\t\t<name><![CDATA[{filepath.stem}]]></name>')
    P.append('\t\t\t<comment><![CDATA[New Musical Part]]></comment>')
    P.append('\t\t\t<sPlug>')
    P.append('\t\t\t\t<id><![CDATA[ACA9C502-A04B-42b5-B2EB-5CEA36D16FCE]]></id>')
    P.append('\t\t\t\t<name><![CDATA[VOCALOID2 Compatible Style]]></name>')
    P.append('\t\t\t\t<version><![CDATA[3.0.0.1]]></version>')
    P.append('\t\t\t</sPlug>')
    P.append('\t\t\t<pStyle>')
    for k,x in [('accent',50),('bendDep',8),('bendLen',0),('decay',50),('fallPort',0),('opening',127),('risePort',0)]:
        P.append(f'\t\t\t\t<v id="{k}">{x}</v>')
    P.append('\t\t\t</pStyle>')
    P.append('\t\t\t<singer>')
    P.append('\t\t\t\t<t>0</t>')
    P.append('\t\t\t\t<bs>0</bs>')
    P.append('\t\t\t\t<pc>0</pc>')
    P.append('\t\t\t</singer>')
    s_events=[(t,v) for k,t,v in pitch_events if k=='S']
    p_events=[(t,v) for k,t,v in pitch_events if k=='P']
    for tick,val in s_events:
        P.append('\t\t\t<cc>')
        P.append(f'\t\t\t\t<t>{tick}</t>')
        P.append(f'\t\t\t\t<v id="S">{val}</v>')
        P.append('\t\t\t</cc>')
    for tick,val in p_events:
        P.append('\t\t\t<cc>')
        P.append(f'\t\t\t\t<t>{tick}</t>')
        P.append(f'\t\t\t\t<v id="P">{val}</v>')
        P.append('\t\t\t</cc>')
    note_lines=[]
    for n0 in valid:
        z=_t(n0.onset,tempo); dur=max(1,_t(n0.offset,tempo)-z)
        lyric=_vsqx_lyric(getattr(n0,'lyric',''))
        tone=int(np.clip(round(n0.pitch),0,127))
        note_lines.append('\t\t\t<note>')
        note_lines.append(f'\t\t\t\t<t>{z}</t>')
        note_lines.append(f'\t\t\t\t<dur>{dur}</dur>')
        note_lines.append(f'\t\t\t\t<n>{tone}</n>')
        note_lines.append('\t\t\t\t<v>64</v>')
        note_lines.append(f'\t\t\t\t<y><![CDATA[{lyric}]]></y>')
        note_lines.append('\t\t\t\t<p><![CDATA[a]]></p>')
        note_lines.append('\t\t\t\t<nStyle>')
        for k,x in [('accent',50),('bendDep',0),('bendLen',0),('decay',50),('fallPort',0),('opening',127),('risePort',0),('vibLen',0),('vibType',0)]:
            note_lines.append(f'\t\t\t\t\t<v id="{k}">{x}</v>')
        note_lines.append('\t\t\t\t</nStyle>')
        note_lines.append('\t\t\t</note>')
    raw_notes='\n'.join(note_lines)
    raw_notes=raw_notes.replace('</note>\n\t\t\t<note>','</note><note>')
    P.append(raw_notes)
    P.append('\t\t\t<plane>0</plane>')
    P.append('\t\t</vsPart>')
    P.append('\t</vsTrack>')
    P.append('\t<monoTrack>')
    P.append('\t</monoTrack>')
    P.append('\t<stTrack>')
    P.append('\t</stTrack>')
    P.append('\t<aux>')
    P.append('\t\t<id><![CDATA[AUX_VST_HOST_CHUNK_INFO]]></id>')
    P.append('\t\t<content><![CDATA[VlNDSwAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=]]></content>')
    P.append('\t</aux>')
    P.append('</vsq4>')
    filepath.parent.mkdir(parents=True,exist_ok=True)
    raw=P[0]+P[1]+'\n'+'\n'.join(P[2:])
    raw=raw.replace('</cc>\n\t\t\t<cc>','</cc><cc>')
    filepath.write_text(raw,encoding='utf-8')
    print(f'Saved VSQX file: {filepath}')
