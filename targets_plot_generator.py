from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt 
import seaborn as sns
import joblib
import datetime
# from tests_folder.tests import resolve_TargetPlotsUploader
from scipy.stats import linregress
# from tests_folder import settings
plt.style.use('ggplot')

# def sir_parameters(x,y): #sir stands for slope, intercept, rvalue (actually there's also the average trend line distance or avg_tld, but it came later)
#   x=np.array(x)
#   y=np.array(y)
#   analytical_params = linregress(x, y)

#   slope = analytical_params.slope
#   intercept = analytical_params.intercept
#   rvalue = analytical_params.rvalue #pay attention that here we have the correlaton coefficient (so not r2 that is the coefficient of determination)

#   y_trend_line = slope*x + intercept #this is computed just for the avg_tld
#   avg_trend_line_distance = np.mean(np.abs(y_trend_line - y)/y_trend_line)

#   return slope, intercept, rvalue**2, avg_trend_line_distance

def sir_parameters(x, y):
    x = np.array(x).flatten()  # Ensure 1D array
    y = np.array(y).flatten()  # Ensure 1D array
    analytical_params = linregress(x, y)

    slope = analytical_params.slope
    intercept = analytical_params.intercept
    rvalue = analytical_params.rvalue
    y_trend_line = slope * x + intercept
    avg_trend_line_distance = np.mean(np.abs(y_trend_line - y) / y_trend_line)

    return slope, intercept, rvalue**2, avg_trend_line_distance


def comparison_plot(predicted_targets,actual_targets,dates,target):
  predicted_targets = np.array(predicted_targets).flatten()
  actual_targets = np.array(actual_targets).flatten()
  trend_slope,trend_intercept,trend_r2,dispersion=sir_parameters(actual_targets,predicted_targets)

  fig,ax = plt.subplots(1,figsize=(20,16))
  date_format = "%Y-%m-%d"
  dates = [datetime.datetime.strptime(date, date_format) for date in dates]
  # print(len(dates))
  dates = (np.array(dates)).squeeze()
  actual_targets = np.array(actual_targets)
  # print(dates.shape)
  # sys.exit()
  sns.lineplot(x=dates,y=actual_targets,ax=ax, label='Actual')
  sns.lineplot(x=dates,y=predicted_targets,ax=ax, label='Predicted')

  ax.set_xlabel('DATE', fontsize=20)
  ax.set_ylabel(f'{target}', fontsize=20)
  # Convert list to pandas Series
  dates = pd.Series(pd.to_datetime(dates))
  start_date = str(dates.min())
  end_date = str(dates.max())
  plt.title(f'Comparison graph - Predicted and Actual Targets versus Dates for {target} from {start_date} to {end_date}', fontweight='bold', fontsize=20)
  plt.legend(fontsize=20)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)

  text_size = 'x-large'

  plt.figtext(.92,.85,['trend_slope:',round(trend_slope,4)], fontsize = text_size)
  plt.figtext(.92,.80,['trend_intercept:',round(trend_intercept,4)], fontsize=text_size)
  plt.figtext(.92,.75,['trend_r2:',round(trend_r2,4)], fontsize=text_size)
  plt.figtext(.92,.70,['trend_standard_deviation:',round(np.std(actual_targets),4)], fontsize=text_size)
  plt.figtext(.92,.65,['trend_dispersion:',round(dispersion,4)], fontsize = text_size)
  # plt.tight_layout()  # Adjust subplots parameters
  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
  plt.savefig(temp_file.name,  bbox_inches='tight')
  plt.close()
  # plt.savefig(plot_path + '_comparision',  bbox_inches='tight')
  
  result = {
    'trend_slope' : trend_slope,
    'trend_intercept': trend_intercept,
    'trend_r2': trend_r2,
    'dispersion': dispersion
  }

  return result,temp_file.name


# def ratio_plot(predicted_targets,actual_targets,dates,target):
#   predicted_targets = np.array(predicted_targets)
#   actual_targets = np.array(actual_targets)
#   fig,ax = plt.subplots(1,figsize=(20,16))
#   ratio = predicted_targets/actual_targets
#   # print("ok")
#   a=np.linspace(0,len(actual_targets),len(actual_targets),dtype=np.int32)
#   z=np.polyfit(a,ratio,1)
#   p=np.poly1d(z)

#   plt.plot(ratio,label='Ratio') 
#   plt.plot(a,p(a), alpha=0.75, label='Fitted Line')

#   plt.xlabel('DATE',fontsize=20)
#   plt.ylabel(f'{target}', fontsize=20)
#     # Convert list to pandas Series
#   dates = pd.Series(pd.to_datetime(dates))
#   start_date = str(dates.min())
#   end_date = str(dates.max())
#   plt.title(f' Ratio plot - Predicted / Actual Targets Ratios versus Dates for {target} from {start_date} to {end_date}', fontweight='bold', fontsize=20)
#   plt.legend(fontsize=20)
#   plt.xticks(fontsize=15)
#   plt.yticks(fontsize=15)

#   trend_slope,trend_intercept,trend_r2,dispersion=sir_parameters(actual_targets,ratio)
#   text_size = 'x-large'

#   plt.figtext(.92,.85,['trend_slope:',round(trend_slope,4)], fontsize = text_size)
#   plt.figtext(.92,.80,['trend_intercept:',round(trend_intercept,4)], fontsize=text_size)
#   plt.figtext(.92,.75,['trend_r2:',round(trend_r2,4)], fontsize=text_size)
#   plt.figtext(.92,.70,['trend_standard_deviation:',round(np.std(actual_targets),4)], fontsize=text_size)
#   plt.figtext(.92,.65,['trend_dispersion:',round(dispersion,4)], fontsize = text_size)
#   # plt.tight_layout()  # Adjust subplots parameters
#   temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
#   plt.savefig(temp_file.name,  bbox_inches='tight')
#   plt.close()
#   # plt.savefig(plot_path + '_ratio',  bbox_inches='tight')

#   result = {
#     'trend_slope' : trend_slope,
#     'trend_intercept': trend_intercept,
#     'trend_r2': trend_r2,
#     'dispersion': dispersion
#   }

#   return result,temp_file.name

def ratio_plot(predicted_targets, actual_targets, dates, target):
    predicted_targets = np.array(predicted_targets).flatten()
    actual_targets = np.array(actual_targets).flatten()
    ratio = predicted_targets / actual_targets

    # Ensure 'a' is a 1D array of indices
    a = np.arange(len(ratio))

    # Fit the polynomial
    z = np.polyfit(a, ratio, 1)  # Linear fit (degree=1)
    p = np.poly1d(z)  # Create polynomial object

    # Plotting
    fig, ax = plt.subplots(1, figsize=(20, 16))
    ax.plot(ratio, label='Ratio')
    ax.plot(a, p(a), alpha=0.75, label='Fitted Line')

    ax.set_xlabel('DATE', fontsize=20)
    ax.set_ylabel(f'{target}', fontsize=20)
    dates = pd.Series(pd.to_datetime(dates))
    start_date = str(dates.min())
    end_date = str(dates.max())
    ax.set_title(
        f'Ratio Plot - Predicted / Actual Targets Ratios vs. Dates for {target} from {start_date} to {end_date}',
        fontweight='bold', fontsize=20
    )
    ax.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    trend_slope, trend_intercept, trend_r2, dispersion = sir_parameters(actual_targets, ratio)
    text_size = 'x-large'

    plt.figtext(.92, .85, f'trend_slope: {round(trend_slope, 4)}', fontsize=text_size)
    plt.figtext(.92, .80, f'trend_intercept: {round(trend_intercept, 4)}', fontsize=text_size)
    plt.figtext(.92, .75, f'trend_r2: {round(trend_r2, 4)}', fontsize=text_size)
    plt.figtext(.92, .70, f'trend_standard_deviation: {round(np.std(actual_targets), 4)}', fontsize=text_size)
    plt.figtext(.92, .65, f'trend_dispersion: {round(dispersion, 4)}', fontsize=text_size)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, bbox_inches='tight')
    plt.close()

    result = {
        'trend_slope': trend_slope,
        'trend_intercept': trend_intercept,
        'trend_r2': trend_r2,
        'dispersion': dispersion
    }

    return result, temp_file.name


def scatter_plot(predicted_targets,actual_targets,dates,target):
  trend_slope,trend_intercept,trend_r2,dispersion=sir_parameters(actual_targets,predicted_targets)

  fig,ax = plt.subplots(1,figsize=(20,16))

  sns.regplot(x=actual_targets,y=predicted_targets,ax=ax)

  ax.set_xlabel(f'Actual Target {target}', fontsize=20)
  ax.set_ylabel(f'Predicted Target {target}', fontsize=20)
  dates = pd.Series(pd.to_datetime(dates))

  start_date = str(dates.min())
  end_date = str(dates.max())
  plt.title(f'Scatter graph - Predicted and Actual Targets versus Dates for {target} from {start_date} to {end_date}', fontweight='bold', fontsize=15)
  text_size = 'x-large'

  plt.figtext(.92,.85,['trend_slope:',round(trend_slope,4)], fontsize = text_size)
  plt.figtext(.92,.80,['trend_intercept:',round(trend_intercept,4)], fontsize=text_size)
  plt.figtext(.92,.75,['trend_r2:',round(trend_r2,4)], fontsize=text_size)
  plt.figtext(.92,.70,['trend_standard_deviation:',round(np.std(actual_targets),4)], fontsize=text_size)
  plt.figtext(.92,.65,['trend_dispersion:',round(dispersion,4)], fontsize = text_size)
  # plt.tight_layout()  # Adjust subplots parameters
  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
  plt.savefig(temp_file.name,  bbox_inches='tight')
  plt.close()
  # plt.savefig(plot_path + '_scatter',  bbox_inches='tight')

  result = {
      'trend_slope' : trend_slope,
      'trend_intercept': trend_intercept,
      'trend_r2': trend_r2,
      'dispersion': dispersion
  }

  return result,temp_file.name

def generate_plot(predicted_targets,actual_targets,dates,target):
  print("Started")
  predicted_targets = np.array(predicted_targets)
  actual_targets = np.array(actual_targets)
  # dates = np.array(dates,dtype='datetime64')

  plot_funcs = {
    'Comparision': comparison_plot(predicted_targets,actual_targets,dates,target)[0],
    'Comparisio_plot_address':comparison_plot(predicted_targets,actual_targets,dates,target)[1],
    'Ratio': ratio_plot(predicted_targets,actual_targets,dates,target)[0],
    'Ratio_Plot_address': ratio_plot(predicted_targets,actual_targets,dates,target)[1],
    'Scatter': scatter_plot(predicted_targets,actual_targets,dates,target)[0],
    'Scatter_plot_address':scatter_plot(predicted_targets,actual_targets,dates,target)[1]
  }

  # result = {}
  # plot_type ={}
  # for value in plot_funcs.values():
  #  result[value] = plot_funcs[value](predicted_targets,actual_targets,dates,target)[1]
  # # results_list = [value for value in plot_funcs.values()]
  # # print(results_list)
  # for value in plot_funcs.keys():
  #  plot_type[value] = plot_funcs[value](predicted_targets,actual_targets,dates,target)
  # # plot_type = [value for value in plot_funcs.keys()]

  return plot_funcs

def image_addresses(predicted_targets,actual_targets,dates,target):
  # print("started")
  result,comparison_plot_address = comparison_plot(predicted_targets,actual_targets,dates,target)
  result,ratio_plot_address = ratio_plot(predicted_targets,actual_targets,dates,target)
  result,scatter_plot_address = scatter_plot(predicted_targets,actual_targets,dates,target)
  # print("ok")

  plot_addresses = {
    'comparison_plot': comparison_plot_address,
    'ratio_plot': ratio_plot_address,
    'scatter_plot': scatter_plot_address
  }
  return plot_addresses
