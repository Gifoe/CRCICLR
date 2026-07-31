from __future__ import annotations

from pathlib import Path
import re
import numpy as np

from hsc_tta.preprocessing.annotations import map_mi_event, map_sleep_label
from hsc_tta.preprocessing.channels import select_sleep_channels
from hsc_tta.preprocessing.signal import preprocess_signal


def _run_id(path: Path) -> int:
    match = re.search(r"R(\d{2})", path.stem, re.I)
    if not match: raise ValueError(f"cannot parse run from {path.name}")
    return int(match.group(1))


def preprocess_sleep_recording(path: str | Path, dataset: str, target_rate: float, bandpass: tuple[float,float], epoch_seconds: float = 30.0) -> dict[str,np.ndarray]:
    import mne
    path=Path(path); raw=mne.io.read_raw_edf(path,preload=True,verbose="ERROR")
    selection=select_sleep_channels(list(raw.ch_names),dataset)
    if not selection["eligible"]: raise ValueError("required central channels missing")
    raw.pick(selection["selected"])
    source_rate=float(raw.info["sfreq"])
    signal=preprocess_signal(raw.get_data(),source_rate,target_rate,bandpass)
    windows=[]; labels=[]; starts=[]; ends=[]
    n_samples=int(round(epoch_seconds*target_rate))
    for annotation in raw.annotations:
        mapped=map_sleep_label(annotation["description"],dataset)
        if mapped is None or float(annotation["duration"])+1e-6 < epoch_seconds: continue
        count=int(np.floor(float(annotation["duration"])/epoch_seconds))
        for offset in range(count):
            onset=float(annotation["onset"])+offset*epoch_seconds
            start=int(round(onset*target_rate)); end=start+n_samples
            if start>=0 and end<=signal.shape[1]:
                windows.append(signal[:,start:end]); labels.append(mapped); starts.append(onset); ends.append(onset+epoch_seconds)
    if not windows: raise ValueError("no complete mapped sleep epochs")
    return {"signal":np.stack(windows),"label":np.asarray(labels,np.int16),"window_start":np.asarray(starts),"window_end":np.asarray(ends),"channel_names":np.asarray(selection["selected"],dtype="S32"),"channel_mask":np.asarray(selection["channel_mask"],bool),"sampling_rate":np.asarray(target_rate),"recording_id":np.asarray([path.stem]*len(windows),dtype="S64"),"run_id":np.full(len(windows),-1,np.int16)}


def preprocess_mi_recordings(paths: list[str | Path], target_rate: float, bandpass: tuple[float,float], event_seconds: float = 4.0) -> dict[str,np.ndarray]:
    import mne
    windows=[]; labels=[]; starts=[]; ends=[]; recs=[]; runs=[]; channel_names=None
    for item in sorted(map(Path,paths)):
        run=_run_id(item); raw=mne.io.read_raw_edf(item,preload=True,verbose="ERROR")
        source_rate=float(raw.info["sfreq"]); processed=preprocess_signal(raw.get_data(),source_rate,target_rate,bandpass)
        current=list(raw.ch_names)
        if channel_names is None: channel_names=current
        elif current!=channel_names: raise ValueError("EEGMMIDB official channel order changed across runs")
        n_samples=int(round(event_seconds*target_rate))
        for annotation in raw.annotations:
            mapped=map_mi_event(run,str(annotation["description"]))
            if mapped is None: continue
            duration=min(float(annotation["duration"]),event_seconds)
            if duration+1e-6<event_seconds: continue
            onset=float(annotation["onset"]); start=int(round(onset*target_rate)); end=start+n_samples
            if start>=0 and end<=processed.shape[1]:
                windows.append(processed[:,start:end]); labels.append(mapped); starts.append(onset); ends.append(onset+event_seconds); recs.append(item.stem); runs.append(run)
    if not windows: raise ValueError("no complete mapped MI events")
    return {"signal":np.stack(windows),"label":np.asarray(labels,np.int16),"window_start":np.asarray(starts),"window_end":np.asarray(ends),"channel_names":np.asarray(channel_names,dtype="S32"),"channel_mask":np.ones(len(channel_names),bool),"sampling_rate":np.asarray(target_rate),"recording_id":np.asarray(recs,dtype="S64"),"run_id":np.asarray(runs,np.int16)}

