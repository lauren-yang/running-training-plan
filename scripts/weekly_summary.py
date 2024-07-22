import pandas as pd

def classify_activity(row, difficulty_dict, distance_dict, pace_dict):
    difficulty = ''
    distance = ''
    
    if not pd.isnull(row['Average Heart Rate']):
        for key, value in difficulty_dict.items():
            if value['min'] <= row['Average Heart Rate'] < value['max']:
                difficulty = key
                break
    else:
        for time_range, paces in pace_dict.items():
            lower, upper = map(int, time_range.split('-'))
            if lower <= row['Predicted 5K Time'] < upper:
                for pace_key, pace_value in paces.items():
                    if pace_value[0] <= row['Average Speed'] < pace_value[1]:
                        difficulty = pace_key
                        break
                break

    for key, value in distance_dict.items():
        if value['min'] <= row['Distance'] < value['max']:
            distance = key
            break
            
    return difficulty, distance

def summarize_week(group):
    summary = {
        'easy_long_runs': len(group[(group['Difficulty'] == 'easy') & (group['Distance Type'] == 'long')]),
        'threshold_long_runs': len(group[(group['Difficulty'] == 'threshold') & (group['Distance Type'] == 'long')]),
        'easy_medium_runs': len(group[(group['Difficulty'] == 'easy') & (group['Distance Type'] == 'medium')]),
        'threshold_medium_runs': len(group[(group['Difficulty'] == 'threshold') & (group['Distance Type'] == 'medium')]),
        'hard_long_runs': len(group[(group['Difficulty'] == 'hard') & (group['Distance Type'] == 'long')]),
        'hard_medium_runs': len(group[(group['Difficulty'] == 'hard') & (group['Distance Type'] == 'medium')]),
        'hard_short_runs': len(group[(group['Difficulty'] == 'hard') & (group['Distance Type'] == 'short')]),
        'threshold_short_runs': len(group[(group['Difficulty'] == 'threshold') & (group['Distance Type'] == 'short')]),
        'easy_short_runs': len(group[(group['Difficulty'] == 'easy') & (group['Distance Type'] == 'short')]),
    }
    for key in distance_dict.keys():
        for diff in difficulty_dict.keys():
            run_type = f"{diff}_{key}"
            summary[f"cumulative_distance_{run_type}"] = group[(group['Difficulty'] == diff) & (group['Distance Type'] == key)]['Distance'].sum()
            summary[f"cumulative_time_{run_type}"] = group[(group['Difficulty'] == diff) & (group['Distance Type'] == key)]['Moving Time'].sum()
    return pd.Series(summary)
