from sklearn.metrics import mean_absolute_error
import rentpredictor

manager = rentpredictor.Manager()
manager.load_dataframes()
manager.load_model('geo_model.py')
true, pred = manager.validate_model(199901, 202306, 202307, 202312)
print(mean_absolute_error(true, pred))

