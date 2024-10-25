from sklearn.metrics import mean_absolute_error
import pandas as pd
import rentpredictor

manager = rentpredictor.Manager()
manager.select_model('ensemble_model.py')

'''train_start_date = pd.Timestamp('2019-01-01')
val_start_date = pd.Timestamp('2020-01-01')
val_end_date = pd.Timestamp('2023-12-31')
val_span = pd.Timedelta(days=180)
val_delta = pd.Timedelta(days=360)

avg_mae = 0
date_range = pd.date_range(val_start_date, val_end_date - val_span, freq=val_delta)
for cur_date in date_range:
    true, pred = manager.validate_model(train_start_date, cur_date, cur_date, cur_date + val_span)
    true, pred = true[pred.notna()], pred[pred.notna()]
    mae = mean_absolute_error(true, pred)
    avg_mae += mae
    print('MAE:', mae)
    
avg_mae /= len(date_range)
print('Average MAE:', avg_mae)
'''

true, pred = manager.validate_model(
    pd.Timestamp('2019-01-01'), pd.Timestamp('2023-06-30'), pd.Timestamp('2023-07-01'), pd.Timestamp('2023-12-31')
)
true, pred = true[pred.notna()], pred[pred.notna()]
mae = mean_absolute_error(true, pred)
print('MAE:', mae)

'''pred = manager.test_model()
pred.name = 'deposit'
pred.to_csv('submission.csv', index_label='index')'''