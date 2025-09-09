# SOC Predictive Distribution Pipeline

Goal: Estimate per-pixel soil organic carbon (SOC) stocks in mangroves with calibrated predictive distributions using NGBoost.

## Modules
- features: building satellite & ancillary covariates
- models: NGBoost distributional models (LogNormal, Gamma)
- evaluation: spatial CV, calibration diagnostics (PIT, CRPS, coverage)
- io: loading in-situ data & raster stacks, tiling inference
- utils: shared helpers

## High-level Steps
1. Load and harmonize in-situ SOC observations (depth normalization, units)
2. Build / load feature raster stacks (Sentinel-2 indices, Sentinel-1 stats, DEM, distances, climate)
3. Extract feature vectors at point locations (buffer median option)
4. Train NGBoost (LogNormal first) with spatial block CV
5. Evaluate: NLL, CRPS, coverage of central prediction intervals, PIT histogram, sharpness
6. (Optional) Compare Gamma vs LogNormal
7. Predict distribution parameters to raster tiles; write mean, std, selected quantiles (e.g. 0.05,0.5,0.95)

## Commands (planned)
```
python -m soc_pipeline.run build-features --config config.yml
python -m soc_pipeline.run train --config config.yml --dist lognormal
python -m soc_pipeline.run evaluate --config config.yml --dist lognormal
python -m soc_pipeline.run predict --config config.yml --dist lognormal --out out_dir
```

## To Do
- Implement io loaders
- Implement feature computations (Sentinel-2 temporal composites, S1 stats)
- Implement spatial block splitter
- Add NGBoost wrappers
- Add evaluation metrics
- Add raster tiling prediction writer

## Pseudo-code (Steps 12–14)

```python
# 1. Load rasters -> feature stack (ensure alignment)
features_cfg = 'features.csv'  # columns: feature,path

# 2. Load in-situ
df = load_insitu_csv('soc_points.csv')

# 3. Transform target (log if lognormal)
eps = 1e-6
df['y_log'] = np.log(df.soc + eps)

# 4. Spatial split by site
for train_idx, test_idx in leave_group_out_indices(df, 'site_id'):
	X_train, y_train = df.loc[train_idx, feature_cols], df.loc[train_idx, 'y_log']
	X_val, y_val = df.loc[test_idx, feature_cols], df.loc[test_idx, 'y_log']
	model = NGBRegressor(Dist=Normal, n_estimators=1500, learning_rate=0.03,
						 col_sample=0.8, minibatch_frac=0.7, Base=DecisionTreeRegressor(max_depth=4, min_samples_leaf=50))
	model.fit(X_train.values, y_train.values)
	# early stopping placeholder: track val NLL per staged model (custom loop needed)
	nll = negative_log_likelihood(model, X_val.values, y_val.values)

# 5. Refit on all training points (excluding final blind test region)
model.fit(df[feature_cols].values, df['y_log'].values)

# 6. Raster inference
raster_predict(bundle={ 'model': model, 'features': feature_cols, 'transform': LogTransform(eps) },
			   raster_config=features_cfg, out_dir='out_maps', transform=LogTransform(eps))
```

### Reporting
- Provide maps: median (band1), mean (band2), std (band3), Q05 (band4), Q95 (band5), width (band6).
- Diagnostics JSON: NLL, CRPS, PIT mean/std, coverage 50/80/95.
- Save PIT histogram & coverage plots (future extension).
- List OOD pixels masked (percentage of area).

### Extensions
- Add Student-t (log-space) distribution.
- Conformal prediction for guaranteed marginal coverage of Q95 band.
- KDE-based feature density for nuanced OOD.
```
