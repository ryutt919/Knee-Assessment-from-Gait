import pandas as pd
import json

target_file = r"c:\Users\kimt9\OneDrive\Desktop\Gal4_PM\ryutt\Walking\data\processed\Master_Gait_Dataset_lower.parquet"

try:
    df = pd.read_parquet(target_file, columns=['group', 'participant'])
    cols = pd.read_parquet(target_file).columns.tolist()
    
    output = {
        "Total columns": len(cols),
        "Unique Groups": df['group'].unique().tolist(),
        "Shape": [len(df), len(cols)],
        "Joint Angle sample": [c for c in cols if 'jointAngle' in c][:30]
    }
    
    with open('agent_temp/parquet_summary.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
        
except Exception as e:
    with open('agent_temp/parquet_summary.json', 'w', encoding='utf-8') as f:
        json.dump({"error": str(e)}, f, indent=4)
