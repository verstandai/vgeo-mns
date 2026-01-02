import pandas as pd

def apply_feedback_overrides(df, feedback_df):
    """
    Applies user feedback overrides to the main dataframe.
    
    Args:
        df (pd.DataFrame): The main news dataframe.
        feedback_df (pd.DataFrame): The dataframe containing user feedback.
        
    Returns:
        tuple: (Modified pd.DataFrame, recap_tree dictionary)
    """
    # Store recap relationships: {original_id: [list_of_recap_ids]}
    recap_tree = {} 
    # Store user overrides: {news_id: {'recap_status': 'RECAP', 'sentiment': 0.5}}
    user_overrides = {}

    if not feedback_df.empty and 'news_id' in feedback_df.columns:
        # Get latest feedback per story
        latest_fb = feedback_df.sort_values('timestamp').groupby('news_id').tail(1)
        
        for _, fb in latest_fb.iterrows():
            nid = fb['news_id']
            user_overrides[nid] = {
                'is_recap': fb.get('is_recap', False),
                'linked_original_id': fb.get('linked_original_id'),
                'is_duplicate': fb.get('is_duplicate', False)
            }
            
            # If this is linked to an original, add it to the tree
            orig_id = fb.get('linked_original_id')
            if pd.notna(orig_id) and orig_id != "None":
                try:
                    orig_id = int(float(orig_id))
                    if orig_id not in recap_tree: recap_tree[orig_id] = []
                    recap_tree[orig_id].append(nid)
                except (ValueError, TypeError):
                    pass

    # Apply overrides to the dataframe
    def apply_overrides(row):
        oid = row.name
        if oid in user_overrides:
            info = user_overrides[oid]
            if info['is_recap']:
                row['breaking_recap'] = 'Recap'
            row['linked_original_id'] = info['linked_original_id']
            row['is_duplicate_user'] = info.get('is_duplicate', False)
        else:
            row['linked_original_id'] = None
            row['is_duplicate_user'] = False
        
        # --- Auto-Boost Significance ---
        # If this story has children recaps, it's definitely significant!
        if oid in recap_tree and len(recap_tree[oid]) > 0:
            row['classification'] = 'SIGNIFICANT'
            
        return row

    df = df.apply(apply_overrides, axis=1)
    
    return df, recap_tree
