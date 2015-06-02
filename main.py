from predict import *
from analysis import *

# Read data:                                                                                                                                                                     
data = pandas.read_csv('turnstile_data_master_with_weather.csv')

# Rain distribution plot:                                                                                                                                                        
plt = entries_histogram(data)                                                                                                                                                   
plt.show()                                                                                                                                                                      

# Weekend plot:                                                                                                                                                                  
plt2 = plot_weekend_data(data)
plt2.show()

# Get mannwhitney                                                                                                                                                                
print 'Mann-Whitney:'
print '(mean[rain], mean[norain], U, p)'
print mann_whitney_plus_means(data)

# Predict subway entries                                                                                                                                                         
predictions, res = predict.predictions_normal_eq(data)
print res.summary()

# Plot residuals:                                                                                                                                                                
plt = plot_residuals(data, predictions)
plt.show()

# Compute own R-squared (for comparison and for laughs):                                                                                                                         
print 'Computed R^2-value:'
print compute_r_squared(data, predictions)
