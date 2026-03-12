import json

def analyze():
    # Load Benchmark Data
    with open('results/benchmark/benchmark_results.json', 'r') as f:
        bench_data = json.load(f)

    bench_stats = {}
    baselines = bench_data['metadata']['baselines_run']
    for bl in baselines:
        stats = {'reductions': [], 'latencies': [], 'effs': [], 'confs': [], 'costs': []}
        for vid, v_data in bench_data['per_video'].items():
            if bl in v_data['baselines']:
                b_data = v_data['baselines'][bl]
                
                # Selection stats
                sel = b_data.get('selection')
                if sel and isinstance(sel, dict):
                    sel_count = sel.get('selected_count', 0)
                    total_frames = v_data.get('video_metadata', {}).get('total_frames', 0)
                    if total_frames > 0:
                        stats['reductions'].append(1.0 - (sel_count / total_frames))
                    stats['latencies'].append(sel.get('latency_s', 0))
                    stats['costs'].append(sel.get('vlm_cost_usd', 0))
                
                # Full Extraction stats
                full = b_data.get('full_extraction')
                if full and isinstance(full, dict):
                    meta = full.get('_metadata', {})
                    if meta:
                        stats['confs'].append(meta.get('confidence', 0))
                    metrics = full.get('engagement_metrics', {})
                    if metrics:
                        eff = metrics.get('effectiveness_score')
                        if eff is not None:
                            stats['effs'].append(eff)

        calc_avg = lambda lst: sum(lst) / len(lst) if lst else 0
        bench_stats[bl] = {
            'reduction': calc_avg(stats['reductions']),
            'latency': calc_avg(stats['latencies']),
            'eff': calc_avg(stats['effs']),
            'conf': calc_avg(stats['confs']),
            'cost': calc_avg(stats['costs'])
        }

    # Load New Data
    with open('results/new/first10_results.json', 'r') as f:
        new_data = json.load(f)

    new_stats_raw = {'reductions': [], 'latencies': [], 'effs': [], 'confs': []}
    for res in new_data['results']:
        if res.get('status') == 'success':
            p_stats = res.get('pipeline_stats', {})
            new_stats_raw['reductions'].append(p_stats.get('reduction_rate', 0))
            new_stats_raw['latencies'].append(p_stats.get('processing_time_s', 0))
            
            ext = res.get('extraction', {})
            meta = ext.get('_metadata', {})
            if meta:
                new_stats_raw['confs'].append(meta.get('confidence', 0))
            
            metrics = ext.get('engagement_metrics', {})
            if metrics:
                eff = metrics.get('effectiveness_score')
                if eff is not None:
                    new_stats_raw['effs'].append(eff)

    new_stats = {
        'reduction': calc_avg(new_stats_raw['reductions']),
        'latency': calc_avg(new_stats_raw['latencies']),
        'eff': calc_avg(new_stats_raw['effs']),
        'conf': calc_avg(new_stats_raw['confs'])
    }

    print("===== Benchmark Results =====")
    for bl, s in bench_stats.items():
        if s['latency'] > 0 or s['eff'] > 0:
            print(f"--- {bl} ---")
            print(f"Reduction Rate : {s['reduction']:.2%}")
            print(f"Avg Latency    : {s['latency']:.2f}s")
            if s['cost'] > 0:
                print(f"Avg VLM Cost   : ${s['cost']:.4f}")
            print(f"Avg Effectivns : {s['eff']:.2f}/5")
            print(f"Avg Confidence : {s['conf']:.2%}")

    print("\n===== New Pipeline (first10_results.json) =====")
    print(f"Reduction Rate : {new_stats['reduction']:.2%}")
    print(f"Avg Pipeline Time: {new_stats['latency']:.2f}s")
    print(f"Avg Effectivns : {new_stats['eff']:.2f}/5")
    print(f"Avg Confidence : {new_stats['conf']:.2%}")

if __name__ == '__main__':
    analyze()
