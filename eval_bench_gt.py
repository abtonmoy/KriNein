import json
import os

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
results_file = r"results\benchmark\benchmark_results.json"
pipeline_data = load_json(results_file)
baselines = pipeline_data['metadata']['baselines_run']

print(f"{'Pipeline':<20} | {'Topic Acc':<10} | {'Sent Acc':<10} | {'Funny MAE':<10} | {'Excit MAE':<10} | {'Effec MAE':<10}")
print("-" * 85)

for bl in baselines:
    metrics = {
        "topic_matches": 0, "sentiment_matches": 0,
        "total_valid_topics": 0, "total_valid_sentiments": 0,
        "funny_mae": 0.0, "exciting_mae": 0.0, "effective_mae": 0.0,
        "scoring_count": 0
    }
    
    for vid_name, v_data in pipeline_data['per_video'].items():
        vid = vid_name.replace('.mp4', '')
        if len(vid) != 11:
            vid = vid.replace('_', '-', 1)
        if vid not in gt_topics and len(vid_name) > 11:
            vid = vid_name[-11:] # fallback
        if vid not in gt_topics:
             vid = vid_name.replace('_', '', 1)

        b_data = v_data.get('baselines', {}).get(bl, {})
        ext = b_data.get('full_extraction', {})
        if not ext: continue

        # Topic
        pred_topic_id = str(ext.get('topic', {}).get('topic_id', -1))
        gt_t = str(gt_topics.get(vid, "N/A"))
        if gt_t != "N/A":
            metrics["total_valid_topics"] += 1
            if pred_topic_id == gt_t: metrics["topic_matches"] += 1
            
        # Sentiment
        pred_sent_id = str(ext.get('sentiment', {}).get('primary_sentiment_id', -1))
        gt_s = str(gt_sentiments.get(vid, "N/A"))
        if gt_s != "N/A":
            metrics["total_valid_sentiments"] += 1
            if pred_sent_id == gt_s: metrics["sentiment_matches"] += 1

        # Scores
        eng = ext.get('engagement_metrics', {})
        pred_funny = float(eng.get('is_funny', 0)) if eng.get('is_funny') is not None else 0
        pred_exciting = float(eng.get('is_exciting', 0)) if eng.get('is_exciting') is not None else 0
        pred_eff = float(eng.get('effectiveness_score', 0)) if eng.get('effectiveness_score') is not None else 0

        gt_f_str = gt_funny.get(vid)
        gt_e_str = gt_exciting.get(vid)
        gt_eff_str = gt_effective.get(vid)
        
        if gt_f_str is not None and gt_e_str is not None and gt_eff_str is not None:
            metrics["funny_mae"] += abs(pred_funny - float(gt_f_str))
            metrics["exciting_mae"] += abs(pred_exciting - float(gt_e_str))
            metrics["effective_mae"] += abs(pred_eff - float(gt_eff_str))
            metrics["scoring_count"] += 1
            
    t_acc = metrics['topic_matches'] / metrics['total_valid_topics'] if metrics['total_valid_topics'] else 0
    s_acc = metrics['sentiment_matches'] / metrics['total_valid_sentiments'] if metrics['total_valid_sentiments'] else 0
    f_mae = metrics['funny_mae'] / metrics['scoring_count'] if metrics['scoring_count'] else 0
    e_mae = metrics['exciting_mae'] / metrics['scoring_count'] if metrics['scoring_count'] else 0
    eff_mae = metrics['effective_mae'] / metrics['scoring_count'] if metrics['scoring_count'] else 0
    
    print(f"{bl:<20} | {t_acc:.1%}     | {s_acc:.1%}     | {f_mae:.3f}      | {e_mae:.3f}      | {eff_mae:.3f}")
