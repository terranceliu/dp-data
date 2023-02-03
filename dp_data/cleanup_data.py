import pdb

# Create single target variable for whether person tests positive with any four of the methods
def cleanup_cervical(df, attrs_dict):        
    target_cols = ['Hinselmann', 'Schiller', 'Citology', 'Biopsy']

    df['Has_cancer'] = (df[target_cols].sum(axis=1) > 0).astype(int)
    df.drop(target_cols, axis=1)
    
    attrs_dict['categorical'] = [x for x in attrs_dict['categorical'] if x not in target_cols]
    attrs_dict['categorical'].append('Has_cancer')
    attrs_dict['target'] = 'Has_cancer'

    return df, attrs_dict

def cleanup(dataset, df, attrs_dict):
    # Fills NA with mode/mean. Removes categorical attributes that on a single column
    for col in attrs_dict['categorical']:
        mode = df[col].mode()[0]
        df.loc[:, col] = df[col].fillna(mode).values
    for col in attrs_dict['numerical']:
        mean = df[col].mean()
        df.loc[:, col] = df[col].fillna(mean).values

    # Dataset specific cleanup
    if dataset == 'cervical':
        df, attrs_dict = cleanup_cervical(df, attrs_dict)
    
    # Remove categorical columns that only take on a single value
    for col in attrs_dict['categorical']:
        if len(df[col].unique()) == 1:
            del df[col]
    attrs_dict['categorical'] = [x for x in attrs_dict['categorical'] if x in df.columns]

    return df, attrs_dict