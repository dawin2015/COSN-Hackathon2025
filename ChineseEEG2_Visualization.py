import os, glob, pandas as pd
import mne

# 1) 指向 .vhdr（不要把 .eeg 传给 MNE）
root = "./ChineseEEG"  # 改成你的数据根目录

# print(os.path.join(
#     root, "sub-01_ses-littleprince_task-lis_run-110*_eeg.vhdr"
# ))

vhdr = glob.glob(os.path.join(
    root, "sub-01_ses-littleprince_task-lis_run-110*_eeg.vhdr"
))[0]

raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=True)

# 2) 设置电极位置（若 .vhdr 未自带通道位置信息，用标准模板兜底）
try:
    if raw.get_montage() is None:
        raw.set_montage("standard_1020", on_missing="warn")
except Exception as e:
    print("[WARN] set_montage:", e)

# 3) 叠加 BIDS 事件（若有 *_events.tsv）
events_tsv = vhdr.replace("_eeg.vhdr", "_events.tsv")
if os.path.exists(events_tsv):
    ev = pd.read_csv(events_tsv, sep="\t")
    on = ev["onset"].values
    du = ev["duration"].fillna(0).values if "duration" in ev else [0]*len(ev)
    desc = ev["trial_type"].astype(str).values if "trial_type" in ev else ["event"]*len(ev)
    ann = mne.Annotations(onset=on, duration=du, description=desc)
    raw.set_annotations(ann)

# 4) 交互式原始波形浏览（可标记坏段/坏道）
raw.plot(block=True, scalings="auto")         # 关闭窗口后脚本继续
raw.plot_psd(average=True, fmax=80)           # 功率谱密度(PSD)

# 5) 可选：轻滤波 + 陷波，便于观察
raw_filt = raw.copy().filter(l_freq=1., h_freq=40., picks="eeg")
raw_filt.notch_filter(freqs=[50, 100])        # 若在 60Hz 地区改为 [60, 120]
raw_filt.plot_psd(average=True, fmax=80)

# 6) 快速事件锁定可视化（若上面已设置 Annotations）
if raw.annotations is not None and len(raw.annotations) > 0:
    events, event_id = mne.events_from_annotations(raw, event_id="auto")
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=-0.2, tmax=0.8, baseline=(None, 0),
                        preload=True, detrend=1)
    epochs.plot_image(combine="mean")         # 各通道 trial×time 图
    epochs.average().plot(time_unit="s")      # ERP
    epochs.average().plot_topomap(times=[0.1, 0.2, 0.3, 0.4],
                                  ch_type="eeg", time_unit="s")  # 头皮地形图
