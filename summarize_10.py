import json
from collections import defaultdict

with open('results/new/first10_results.json', 'r') as f:
    data = json.load(f)

reductions = []
times = []
frames_in = []
frames_out = []
effs = []
types = defaultdict(int)

for result in data['results']:
    if result['status'] != 'success': continue
    stats = result['pipeline_stats']
    reductions.append(stats['reduction_rate'])
    times.append(stats['processing_time_s'])
    frames_in.append(stats['total_frames_sampled'])
    frames_out.append(stats['final_frame_count'])
    ext = result.get('extraction', {})
    eff = ext.get('engagement_metrics', {}).get('effectiveness_score')
    if eff: effs.append(eff)
    types[ext.get('ad_type', 'unknown')] += 1

print(f"Videos: {len(reductions)}")
print(f"Avg Redux: {sum(reductions)/len(reductions):.1%}")
print(f"Avg Time: {sum(times)/len(times):.1f}s")
print(f"Avg Frames In: {sum(frames_in)/len(frames_in):.1f}")
print(f"Avg Frames Out: {sum(frames_out)/len(frames_out):.1f}")
print(f"Avg Effectiveness: {sum(effs)/len(effs):.2f}")
print("Types:", dict(types))
