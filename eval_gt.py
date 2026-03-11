import json
import os
import math

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load ground truth
gt_dir = r"data\annotations_videos\video\cleaned_result"
gt_topics = load_json(os.path.join(gt_dir, "video_Topics_clean.json"))
gt_sentiments = load_json(os.path.join(gt_dir, "video_Sentiments_clean.json"))
gt_funny = load_json(os.path.join(gt_dir, "video_Funny_clean.json"))
gt_exciting = load_json(os.path.join(gt_dir, "video_Exciting_clean.json"))
gt_effective = load_json(os.path.join(gt_dir, "video_Effective_clean.json"))

# Load pipeline results
results_file = r"results\new\first10_results.json"
pipeline_data = load_json(results_file)

metrics = {
    "topic_matches": 0,
    "sentiment_matches": 0,
    "total_valid_topics": 0,
    "total_valid_sentiments": 0,
    "funny_mae": 0.0,
    "exciting_mae": 0.0,
    "effective_mae": 0.0,
    "scoring_count": 0
}

import re
print(f"{'Video ID':<15} | {'Topic (GT/Pred)':<30} | {'Sentiment (GT/Pred)':<30} | {'Funny':<10} | {'Exciting':<10} | {'Effective':<10}")
print("-" * 120)

for result in pipeline_data['results']:
    if result['status'] != 'success': continue
    
    # Extract vid (e.g., _6rj5jisB7g from __6rj5jisB7g.mp4 if prefixed, or just standard slice)
    vid_name = result['video_name'].replace('.mp4', '')
    
    # Some video names might have a leading underscore added by yt-dlp if it started with a dash
    # The GT keys are exact 11-char youtube IDs
    vid = vid_name if len(vid_name) == 11 else vid_name.replace('_', '-', 1)
    if vid not in gt_topics and len(vid_name) > 11:
        vid = vid_name[-11:] # fallback

    if vid not in gt_topics:
        # Try one more fallback for youtube-dl prefixing
        vid = vid_name.replace('_', '', 1)

    ext = result.get('extraction', {})
    if not ext: continue

    # Topic
    pred_topic_id = str(ext.get('topic', {}).get('topic_id', -1))
    gt_t = str(gt_topics.get(vid, "N/A"))
    topic_match = pred_topic_id == gt_t if gt_t != "N/A" else False
    
    if gt_t != "N/A":
        metrics["total_valid_topics"] += 1
        if topic_match: metrics["topic_matches"] += 1
        
    # Sentiment
    pred_sent_id = str(ext.get('sentiment', {}).get('primary_sentiment_id', -1))
    gt_s = str(gt_sentiments.get(vid, "N/A"))
    sent_match = pred_sent_id == gt_s if gt_s != "N/A" else False
    
    if gt_s != "N/A":
        metrics["total_valid_sentiments"] += 1
        if sent_match: metrics["sentiment_matches"] += 1

    # Scores
    eng = ext.get('engagement_metrics', {})
    pred_funny = float(eng.get('is_funny', 0))
    pred_exciting = float(eng.get('is_exciting', 0))
    pred_eff = float(eng.get('effectiveness_score', 0))

    gt_f_str = gt_funny.get(vid)
    gt_e_str = gt_exciting.get(vid)
    gt_eff_str = gt_effective.get(vid)
    
    gt_f = float(gt_f_str) if gt_f_str is not None else None
    gt_e = float(gt_e_str) if gt_e_str is not None else None
    gt_eff = float(gt_eff_str) if gt_eff_str is not None else None
    
    if gt_f is not None and gt_e is not None and gt_eff is not None:
        metrics["funny_mae"] += abs(pred_funny - gt_f)
        metrics["exciting_mae"] += abs(pred_exciting - gt_e)
        metrics["effective_mae"] += abs(pred_eff - gt_eff)
        metrics["scoring_count"] += 1
        
    print(f"{vid:<15} | {gt_t:>13}/{pred_topic_id:<16} | {gt_s:>13}/{pred_sent_id:<16} | {gt_f if gt_f is not None else 'N/A'}/{pred_funny} | {gt_e if gt_e is not None else 'N/A'}/{pred_exciting} | {gt_eff if gt_eff is not None else 'N/A'}/{pred_eff}")

print("\n" + "=" * 50)
print("AGGREGATE ACCURACY METRICS")
print("=" * 50)
if metrics["total_valid_topics"] > 0:
    print(f"Topic Accuracy:     {metrics['topic_matches'] / metrics['total_valid_topics']:.1%} ({metrics['topic_matches']}/{metrics['total_valid_topics']})")
if metrics["total_valid_sentiments"] > 0:
    print(f"Sentiment Accuracy: {metrics['sentiment_matches'] / metrics['total_valid_sentiments']:.1%} ({metrics['sentiment_matches']}/{metrics['total_valid_sentiments']})")
    
if metrics["scoring_count"] > 0:
    n = metrics["scoring_count"]
    print(f"Funny MAE:          {metrics['funny_mae'] / n:.3f} (Scale 0-1)")
    print(f"Exciting MAE:       {metrics['exciting_mae'] / n:.3f} (Scale 0-1)")
    print(f"Effective MAE:      {metrics['effective_mae'] / n:.3f} (Scale 1-5)")
