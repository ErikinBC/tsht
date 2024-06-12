"""
https://github.com/gradlab/wastewater_equity/tree/23fe996a3aa3987384e737a10a9c6b1837f0f147/scripts

https://asdfree.com/american-housing-survey-ahs.html
"""

library(haven)
library(httr)

tf <- tempfile()

this_url <-
    paste0(
        "https://www2.census.gov/programs-surveys/ahs/" ,
        "2021/AHS%202021%20National%20PUF%20v1.0%20Flat%20SAS.zip"
    )

GET( this_url , write_disk( tf ) , progress() )

ahs_tbl <- read_sas( tf )

ahs_df <- data.frame( ahs_tbl )

names( ahs_df ) <- tolower( names( ahs_df ) )
