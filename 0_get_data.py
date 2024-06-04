"""
Load the air quality data...


https://www.kaggle.com/datasets/epa/epa-historical-air-quality/data

https://console.cloud.google.com/bigquery?project=airquality-425418&ws=!1m0

https://github.com/SohierDane/BigQuery_Helper/tree/master?tab=readme-ov-file

https://www.kaggle.com/code/sohier/introduction-to-the-bq-helper-package

https://www.kaggle.com/code/sohier/getting-started-with-big-query
"""

import pandas as pd
from google.cloud import bigquery
client = bigquery.Client()
from bq_helper import BigQueryHelper

client = bigquery.Client()


QUERY = """
    SELECT
        extract(DAYOFYEAR from date_local) as day_of_year,
        aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
      city_name = "Los Angeles"
      AND state_name = "California"
      AND sample_duration = "24 HOUR"
      AND poc = 1
      AND EXTRACT(YEAR FROM date_local) = 2015
    ORDER BY day_of_year
        """

bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")




