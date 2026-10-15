"""Export model-ready sparse matrices, fitting only on the frozen training split."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from train_only_preprocessing import fit_austin_preprocessor

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    source, out = Path(a.input), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    features = pd.read_csv(source/'intake_features.csv')
    labels = pd.read_csv(source/'labels_and_splits.csv')
    assert set(features.episode_id) == set(labels.episode_id)
    processor, matrices = fit_austin_preprocessor(features, labels)
    joined = features[['episode_id']].merge(labels, on='episode_id', validate='one_to_one')
    report = {'target': 'observed_all_exit_LOS_over_30_days', 'fit_scope': 'train only', 'splits': {}}
    for split, values in matrices.items():
        matrix = sparse.csr_matrix(values)
        if not np.isfinite(matrix.data).all(): raise ValueError('Non-finite model input')
        y = joined.loc[joined.split.eq(split), ['episode_id','animal_id','long_stay_30d','duration_days','event_cause']]
        assert len(y) == matrix.shape[0] and y.long_stay_30d.notna().all()
        y['matrix_row'] = np.arange(len(y))
        sparse.save_npz(out/f'X_{split}.npz', matrix)
        y.to_csv(out/f'labels_{split}.csv', index=False)
        report['splits'][split] = {'matrix_shape': list(matrix.shape), 'positive_labels': int(y.long_stay_30d.sum())}
    joblib.dump(processor, out/'train_fitted_preprocessor.joblib')
    (out/'feature_names.json').write_text(json.dumps(processor.get_feature_names_out().tolist(), indent=2))
    (out/'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
