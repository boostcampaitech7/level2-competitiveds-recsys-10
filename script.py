from sklearn.metrics import mean_absolute_error
import pandas as pd
import rentpredictor

manager = rentpredictor.Manager()

manager.select_model('geo_model.py')
true, pred = manager.validate_model(
    pd.Timestamp('2019-01-01'), pd.Timestamp('2023-06-30'), pd.Timestamp('2023-07-01'), pd.Timestamp('2023-12-31')
)
mask = pred.notna()
true, pred = true[mask], pred[mask]
mae = mean_absolute_error(true, pred)
print('MAE:', mae)