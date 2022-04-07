from email import header
import time
import dask.dataframe as dd

start = time.time()

columns = ['Year', 'Month', 'DayofMonth', 'DayofWeek', 'CRSDepTime', 
           'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'ActualElapsedTime', 
           'Origin', 'Dest', 'Distance', 'Diverted', 'ArrDelay']

df = dd.read_csv('airline_dataset/airline_14col.csv', names=columns, header=None) # CSV File (~5.4 gb) not added to github due to size limitation 
df.to_parquet('airline_dataset/airline_14col.parquet') # Parquet File (~2 gb) not added to github due to size limitation 

end = time.time()

print(f"\n{'='*40} \n\nTotal Time Taken: {round(end-start, 2)} seconds") # Took 40.99 seconds on i7-10750H with 16gb RAM