import xml.etree.ElementTree as ET
import pandas as pd

def parse_mvnx_to_df(file_path: repr) -> pd.DataFrame:
    """
    Reads an MVNX XML file efficiently using iterparse.
    Returns a flattened Pandas DataFrame where each row is a frame in time.
    """
    rows = []
    
    try:
        # Use both start and end to track when we are inside a normal frame
        context = ET.iterparse(file_path, events=("start", "end"))
        
        in_target_frame = False
        current_frame = {}
        
        for event, elem in context:
            tag = elem.tag.split('}')[-1]
            
            if event == "start":
                if tag == "frame" and elem.get("type") == "normal":
                    in_target_frame = True
                    current_frame = {'time_ms': elem.get('ms')}
                    
            elif event == "end":
                if in_target_frame and tag != "frame":
                    # Extract text values from children nodes inside the frame securely
                    if elem.text and elem.text.strip():
                        values = elem.text.strip().split()
                        for i, val in enumerate(values):
                            # Ensure fast numeric operations
                            current_frame[f"{tag}_{i}"] = float(val)
                            
                elif tag == "frame":
                    if in_target_frame:
                        rows.append(current_frame)
                        in_target_frame = False
                        
                # Clear memory specifically on end elements to prevent memory leak
                elem.clear()

    except Exception as e:
        print(f"Error parsing MVNX file -> {file_path}: {e}")
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    return df
