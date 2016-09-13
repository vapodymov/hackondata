# Databricks notebook source exported at Tue, 13 Sep 2016 01:45:16 UTC
# MAGIC %md
# MAGIC 
# MAGIC # HackOnData.com
# MAGIC 
# MAGIC ## Exercise #3 - Log Analysis
# MAGIC 
# MAGIC 
# MAGIC ### Week 3 Lab 1:
# MAGIC 
# MAGIC Make sure you complete the Week 3 Lab 1:
# MAGIC 
# MAGIC https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2799933550853697/3719583162088724/2202577924924539/latest.html
# MAGIC 
# MAGIC Include the public link to the lab solutions:

# COMMAND ----------

# MAGIC %md
# MAGIC https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8122459673715921/364006105196051/2531719484635850/latest.html
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## TranQuant Log:
# MAGIC 
# MAGIC Starting from
# MAGIC 
# MAGIC http://tranquant.com/subscriptions/569c985d-f1f3-42c5-8b81-11b644e42895 
# MAGIC 
# MAGIC The following fields are included:
# MAGIC 
# MAGIC ```
# MAGIC  log>
# MAGIC  |-- action: string (nullable = true) | Action
# MAGIC  |-- obj_id: string (nullable = true) | Object Id
# MAGIC  |-- obj_type: string (nullable = true) | Object Type
# MAGIC  |-- timestamp: long (nullable = true) | Time Stamp
# MAGIC  |-- type: string (nullable = true) | Type
# MAGIC  |-- ua: string (nullable = true) | User Agent
# MAGIC  |-- uuid: string (nullable = true) | UUID
# MAGIC ```
# MAGIC 
# MAGIC Example:
# MAGIC 
# MAGIC | action | obj_id | obj_type |	timestamp |	type |	ua |	uuid |
# MAGIC | ------------- |:-------------:| -----:|
# MAGIC |hover |	Accommodation and Food Services	| main_category_title |	1467962138212	| Event	| Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36	| 83e28957-3a80-4d3c-b189-339fe0d66c4b|
# MAGIC 
# MAGIC 
# MAGIC The log corresponds to front-end events for TranQuant. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Text parsing
# MAGIC Read each line of the log file as text. Write a regular expression to parse the first 3 fields (action, obj_id, and obj_type), ignore the remaining fields for now. 
# MAGIC 
# MAGIC Hint: Similar to `def parseLogs():` in the Lab

# COMMAND ----------

import json 

fileName = "/mnt/my-hod-data/TQ_web_analytics_de_identified.log"

# Import full dataset as text file
data_raw = sc.textFile(fileName)

# Parse JSON entries in dataset
data = data_raw.map(lambda line: json.loads(line))

# Extract relevant fields in dataset
data_triplet = data.map(lambda line: (line['action'], line['obj_id'], line['obj_type']))

data_triplet.count()


# COMMAND ----------

# MAGIC %md
# MAGIC What are the most popular (top 6) `obj_id`? present the results in a plot.

# COMMAND ----------

obj_id_count = (data.map(lambda x: (x['obj_id'], 1))
                          .reduceByKey(lambda a, b : a + b))

obj_id_count_ordered = obj_id_count.sortBy(lambda x:x[1], ascending=False)

import pandas as pd
obj_counted = pd.DataFrame(obj_id_count_ordered.collect(), columns=["obj_id", "count"])
obj_counted[:6]

# COMMAND ----------

# Create an RDD with Row objects
counts_schema_rdd = sqlContext.createDataFrame(obj_counted)

# Display a plot of the obj_id distribution 
display(counts_schema_rdd)

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see at the plot above, the most poopular obj_id's are about 50% of the whole population. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Read the logs using the dataframes API. Example:
# MAGIC 
# MAGIC `log = sqlContext.read.json("/mnt/HackOnData/challenge3/TQ_web_analytics_de_identified.log")`
# MAGIC 
# MAGIC `log.printSchema()`

# COMMAND ----------

logDF = sqlContext.read.json(fileName)

logDF.printSchema() 


# COMMAND ----------

# MAGIC %md ### UA Fields
# MAGIC 
# MAGIC Parse the ua field  (https://en.wikipedia.org/wiki/User_agent) and extract all the subfields

# COMMAND ----------

dataUA = logDF.select('ua')

dataUA.first()

# COMMAND ----------

import re
from pyspark.sql import Row

# A regular expression pattern to extract UA fields 
USER_AGENT_PATTERN = '^(\S+) \(([^(]*)\) (\S+) \(([^(]*)\) (\S+) (\S+)'

def parseUALogLine(logline):
    """ Parse UA log line 
    Args:
        logline (str): a line of text in the UA Log format
    Returns:
        tuple: either a dictionary containing the parts of the UA Fields and 1,
               or the original invalid log line and 0
    """
    match = re.search(USER_AGENT_PATTERN, logline)
    if match is None:
        return (logline, 0)
    
    return (Row(
        browser_compatibility    = match.group(1),
        os_ver                   = match.group(2),
        platform                 = match.group(3),
        browser_platform_details = match.group(4),
        enhancement_1            = match.group(5),
        enhancement_2            = match.group(6)        
    ))

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the timestamp to hour of the day (0-23). Make a histogram. 

# COMMAND ----------

import datetime

dataTS = logDF.select('timestamp')

hour_of_day = dataTS.map(lambda x: (datetime.datetime.fromtimestamp(int(x[0])/1000).hour,1)).reduceByKey(lambda a,b: a+b)
hour_of_day = hour_of_day.sortByKey()

hd_counted = pd.DataFrame(hour_of_day.collect(), columns=["Hour", "Count"])

counts_schema_hd = sqlContext.createDataFrame(hd_counted)
display(counts_schema_hd)


# COMMAND ----------

# MAGIC %md
# MAGIC What we can see on the histogram:
# MAGIC 
# MAGIC * Morning is a period of the lowest load on the server
# MAGIC * The maximum load is observed during non-business hours

# COMMAND ----------

# MAGIC %md Repeat the same exercise for the the top 3 browsers, do you find any interesting results?

# COMMAND ----------

wb = (dataUA.map(lambda log: (log[0], 1)).reduceByKey(lambda a, b : a + b))

wb_counted = pd.DataFrame(wb.collect(), columns=["UA", "Count"])

wb_count_ordered = wb_counted.sort(['Count'], ascending=[0])

# Top3
wb_count_ordered[:3]


# COMMAND ----------

counts_schema_wb = sqlContext.createDataFrame(wb_count_ordered)
display(counts_schema_wb)

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see from the chart above, Top 3 browser configurations are more than 60% of all the observed. 

# COMMAND ----------

top3wb = wb_count_ordered[:3]
top3wb.iat[0,0], top3wb.iat[1,0], top3wb.iat[2,0]

# COMMAND ----------

# MAGIC %md
# MAGIC And we can notice that all top3 configurations are based on Chrome. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## flight solo....
# MAGIC 
# MAGIC We want to encourange creativity, we want you to tell a story supported by data.
# MAGIC 
# MAGIC What other slice and dice can you perform to extract useful insights?
# MAGIC 
# MAGIC What recommendations do you have for TranQuant based on the log analysis you performed? please show your intermadiate steps, use graphs, be yourself and express your point of view.

# COMMAND ----------

# MAGIC %md
# MAGIC The Day of the Week pattern may be as important as the Hour of the Day. 

# COMMAND ----------

dow = dataTS.map(lambda x: (datetime.date.fromtimestamp(int(x[0])/1000).isoweekday(), 1)).reduceByKey(lambda a,b: a+b)

dow = dow.sortByKey()

dw_counted = pd.DataFrame(dow.collect(), columns=["Day of Week", "Count"])

counts_schema_wd = sqlContext.createDataFrame(dw_counted)
display(counts_schema_wd)

# COMMAND ----------

# MAGIC %md
# MAGIC This chart is the histogram of events depending on the day of the week, from Monday (1) to Sunday (7). The histogram shows that the load on the server in average is higher from Monday to Thursday (when it is maximal), than in the rest of the week. 
# MAGIC 
# MAGIC Combining these results with the Hour of the Day histogram, it is possible to suggest that the load on the server reached extremal values on Thursday nights for the observed period. 
