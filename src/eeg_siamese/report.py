"""Offline, dependency-free QC HTML; plots are diagnostics, not neural SNR."""
import html
from pathlib import Path

import numpy as np


def _plot(x, y, title, color="#146c94", highlight=None):
    x, y = np.asarray(x), np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return f"<p>{html.escape(title)}：无可用数据</p>"
    xmin, xmax = float(x[finite].min()), float(x[finite].max())
    ymin, ymax = float(y[finite].min()), float(y[finite].max())
    sx = 630 / max(xmax - xmin, 1e-9)
    sy = 130 / max(ymax - ymin, 1e-9)
    shade = ""
    if highlight is not None:
        left, right = max(xmin, highlight[0]), min(xmax, highlight[1])
        if right > left:
            shade = (f'<rect x="{50+(left-xmin)*sx:.2f}" y="25" '
                     f'width="{(right-left)*sx:.2f}" height="135" fill="#fff0c2"/>')
    parts, drawing = [], False
    for a, b, good in zip(x, y, finite):
        if not good:
            drawing = False
            continue
        parts.append(f"{'L' if drawing else 'M'}{50+(a-xmin)*sx:.2f},{160-(b-ymin)*sy:.2f}")
        drawing = True
    return (f'<figure><figcaption>{html.escape(title)}</figcaption>'
            f'<svg viewBox="0 0 710 200" role="img" aria-label="{html.escape(title)}">'
            f'{shade}'
            '<path d="M50,25 V160 H680" fill="none" stroke="#ccc"/>'
            f'<path d="{" ".join(parts)}" fill="none" stroke="{color}" stroke-width="1.2"/>'
            f'<text x="8" y="32">{ymax:.2g}</text><text x="8" y="165">{ymin:.2g}</text>'
            f'<text x="50" y="185">{xmin:.2g}</text><text x="630" y="185">{xmax:.2g}</text>'
            '</svg></figure>')


def _trace(signal, channel, rate, start, end):
    """Peak-preserving display reduction; ordinary strides can hide spikes."""
    a, b = max(0, int(start * rate)), min(signal.shape[1], int(np.ceil(end * rate)))
    indices = np.arange(a, b)
    if len(indices) > 2000:
        chosen = []
        for chunk in np.array_split(indices, 1000):
            values = signal[channel, chunk]
            if np.isfinite(values).all():
                chosen.extend(sorted({int(chunk[np.argmin(values)]), int(chunk[np.argmax(values)])}))
            else:
                chosen.extend([int(chunk[0]), int(chunk[-1])])
        indices = np.asarray(chosen, dtype=int)
    return indices / rate, signal[channel, indices]


def window_qc_sections(result, raw_eeg_uv=None, raw_rate=None):
    diagnostic = result["qc"].get("window_qc")
    if diagnostic is None:
        return []
    reference = diagnostic["reference"]
    scope = "训练数据固定标准" if reference and reference["scope"] == "training" else "本记录内描述性标准"
    sections = ['<h2>窗口级异常检查（仅标记）</h2>',
                f'<p>标准：{scope}；可疑窗口 {diagnostic["suspect_window_count"]}/'
                f'{diagnostic["base_valid_window_count"]} 个基础有效窗口 '
                f'（{100 * diagnostic["suspect_fraction_of_valid"]:.2f}%）。'
                f'若全部排除可疑窗口，剩余 {diagnostic["retained_if_excluded_seconds"]:.1f} 秒。</p>',
                '<p>峰峰值和 RMS 在滤波、重采样后、模型缩放前计算。'
                '分数 = (ln(指标) − 中位数) / max(1.4826 × MAD, 0.1)，按通道分别计算。'
                '任一指标高于阈值就标记；幅值较大不等于已确认的眼电或其他伪迹。'
                '标记不会改变有效窗口掩码，训练排除需单独启用。</p>']
    if reference:
        sections.append(f'<p>告警阈值：{reference["score_threshold"]:.2f}；'
                        f'参考窗口数：{reference["reference_window_count"]}。</p>')
    else:
        sections.append('<p>少于 20 个有效窗口，无法建立本记录稳健参考；指标仍保存，未标记不代表干净。</p>')
    details = diagnostic["suspect_details"]
    sections.append('<table><tr><th>窗口编号（从 0 开始）</th><th>时间（秒）</th>'
                    '<th>通道</th><th>触发指标</th><th>数值 μV</th><th>稳健分数</th></tr>')
    for row in details[:100]:
        trigger = max(row["triggers"], key=lambda item: item["score"])
        label = "峰峰值" if trigger["metric"] == "peak_to_peak_uv" else "RMS"
        values = [str(row["window_index"]), f'{row["start_seconds"]:.1f}–{row["end_seconds"]:.1f}',
                  trigger["channel"], label, f'{trigger["value_uv"]:.2f}', f'{trigger["score"]:.2f}']
        sections.append('<tr>' + ''.join(f'<td>{html.escape(v)}</td>' for v in values) + '</tr>')
    sections.append('</table><p>表格按异常分数排序，最多显示 100 个窗口及各窗最强触发项；完整指标和触发原因见 JSON。</p>')
    if not details:
        sections.append('<p>当前规则未标记可疑窗口；这不保证没有伪迹。</p>')
    names = list(result["channel_names"])
    review = diagnostic.get("review_candidates", details[:3])
    if review:
        sections.append('<h3>分数最高的窗口及上下文</h3><p>以下候选可能未超过告警阈值，仍展示波形供复查。</p>')
    for row in review:
        trigger = max(row["triggers"], key=lambda item: item["score"])
        channel = names.index(trigger["channel"])
        start, end = row["start_seconds"] - 5, row["end_seconds"] + 5
        streams = [(result["eeg"], result["sample_rate"], "处理后")]
        if raw_eeg_uv is not None:
            streams.insert(0, (np.asarray(raw_eeg_uv), raw_rate, "原始"))
        for signal, rate, label in streams:
            x, y = _trace(signal, channel, rate, start, end)
            status = "可疑" if row.get("suspect", True) else "复查候选"
            sections.append(_plot(x, y, f'{label}{status}窗口 {row["window_index"]}：'
                                  f'{trigger["channel"]}，横轴秒／纵轴 μV（黄色为目标窗口）',
                                  highlight=(row["start_seconds"], row["end_seconds"])))
    sections.append('<p>分数最高的 3 个有效窗口显示前后各 5 秒上下文；绘图降采样保留分块极值。'
                    '处理后无效区间的零值是占位，不能解释为真实脑电。仅复查已有 NPZ 时不提供原始波形。</p>')
    return sections


def _write_report(sections, path):
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>EEG 质量报告</title>'
                '<style>body{max-width:1000px;margin:32px auto;padding:0 24px;font:16px system-ui;'
                'color:#23313c}figure{margin:24px 0}svg{width:100%;height:auto}svg text{font-size:12px}'
                'table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;'
                'border-bottom:1px solid #ddd}p{line-height:1.6}</style>' + ''.join(sections) + '</html>')
    Path(path).write_text(document, encoding="utf-8")


def render_window_report(result, path):
    name = result["qc"].get("record_id", Path(path).stem)
    _write_report([f'<h1>{html.escape(name)}</h1>'] + window_qc_sections(result), path)


def render_qc_report(raw_eeg_uv, sample_rate, result, path):
    qc = result["qc"]
    name = qc.get("record_id", Path(path).stem)
    sections = [f'<h1>{html.escape(name)}</h1>',
                '<p>质量指标用于定位干扰，不代表真实神经信噪比。零值占位及滤波保护区不参与有效窗口。</p>',
                f'<p>状态：{html.escape(qc["quality_status"])}；有效窗口 '
                f'{qc["valid_window_count"]}/{qc["total_window_count"]}；'
                f'有效窗口时长 {qc["valid_window_seconds"]:.1f} 秒；'
                f'边界保护 {qc["guard_seconds"]:.2f} 秒。</p>']
    raw = np.asarray(raw_eeg_uv)
    channel = 0
    names = list(result["channel_names"])
    if "C3" in names:
        channel = names.index("C3")
    for signal, rate, title in [(raw, sample_rate, "原始"),
                                (result["eeg"], result["sample_rate"], "处理后")]:
        stop = min(signal.shape[1], round(rate * 15))
        step = max(1, stop // 1500)
        indices = np.arange(0, stop, step)
        sections.append(_plot(indices / rate, signal[channel, indices],
                              f'{title}波形：{names[channel]}，横轴秒／纵轴 μV（前 15 秒）'))
    for key, title in [("psd_before", "原始"), ("psd_after", "处理后")]:
        spectrum = qc.get(key)
        if spectrum and spectrum.get("frequency_hz"):
            power = np.asarray(spectrum["median_power_uv2_per_hz"])
            sections.append(_plot(spectrum["frequency_hz"], 10 * np.log10(np.maximum(power, 1e-20)),
                                  f'{title}频谱：横轴 Hz／纵轴 dB(μV²/Hz)'))
    metrics = qc["channel_metrics"]
    sections.append('<h2>通道质量</h2><table><tr><th>通道</th><th>缺失比例</th><th>峰峰值 μV</th>'
                    '<th>平线样本</th><th>饱和样本</th><th>强跳变</th></tr>')
    for row in metrics:
        values = [row.get(k) for k in ("channel", "nonfinite_fraction", "peak_to_peak_uv",
                                       "flatline_samples", "clipped_samples", "jump_count")]
        sections.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in values) + '</tr>')
    sections.append('</table><p>完整配置、无效时间区间、事件映射和文件校验值见同名 QC JSON。</p>')
    sections.extend(window_qc_sections(result, raw, sample_rate))
    _write_report(sections, path)
