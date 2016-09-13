# Databricks notebook source exported at Mon, 12 Sep 2016 01:42:09 UTC
# MAGIC %md
# MAGIC # HackOn(Data) Challenge
# MAGIC ## Parking Tickets Play
# MAGIC 
# MAGIC [Valerii Podymov](http://ca.linkedin.com/in/valeriipodymov)
# MAGIC 
# MAGIC September 10, 2016
# MAGIC 
# MAGIC ### Introduction
# MAGIC 
# MAGIC According to the City of Toronto [1], approximately 2.8 million parking tickets are issued annually across the city. 
# MAGIC 
# MAGIC The aim of this challenge is to find out what factors contribute to the amount of parking tickets for the city of Toronto.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis of Data Patterns and Influencing Factors

# COMMAND ----------

library(lattice)
library(ggplot2)

# Loading Parking Tickets dataset (three CSV files)
file_names <- dir("/dbfs/mnt/my-hod-data/chm", full.names = T)
df <- do.call(rbind, lapply(file_names, read.csv))
str(df)

# COMMAND ----------

# Histogram of the fine amount
hist(df$set_fine_amount, breaks = 20, col = "peachpuff", xlab = "Fine amount", main = "Histogram of Fine Amount")
abline(v = mean(df$set_fine_amount), col = "red", lwd = 2)
abline(v = median(df$set_fine_amount), col = "blue", lwd = 2)
legend(x = "topright", c("Mean", "Median"), col = c("red", "blue"), lwd = c(2, 2, 2))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Determine the annual pattern

# COMMAND ----------

# Transform date_of_infraction to Date format
df$date_of_infraction <- as.Date(as.character(df$date_of_infraction), format = '%Y%m%d')


# COMMAND ----------

# Find a total daily fine amount over a year
amount_per_day <- tapply(df$set_fine_amount, df$date_of_infraction, sum, na.rm = TRUE)

par(cex.main = 0.8)
plot(amount_per_day, type = 'l', col = "blue",
     xlab = "Day of a year", ylab = "Total fine amount",
     main = "Daily fine amount in $$$ over a year")

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see, the chart above has a distinct weekly pattern with cyclic tops and bottoms. Obviuosly bottoms fall on weekends, when the total ticket amount is significantly lower.
# MAGIC Let's quantify how days of week influences the fine amount. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Determine the day of week pattern

# COMMAND ----------

# Week days
w_days <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

df$daytype <- NULL
df$daytype <- factor((weekdays(as.POSIXlt(df$date_of_infraction)) 
    %in% w_days), levels = c(FALSE, TRUE), labels = c("weekend", "weekday"))

# Aggregate data by days of week
avg_week_pattern <- aggregate(set_fine_amount ~ daytype, data = df, mean)


# COMMAND ----------

barchart(set_fine_amount ~ daytype, data = avg_week_pattern, col = "blue",
         xlab = "Day of week", ylab = "Avg fine amount",
         main = "Average fine amount in $$$ by days of week")

# COMMAND ----------

# MAGIC %md
# MAGIC Thus, the average amount is ~ $6 higher on weekdays. It may mean that more expensive tickets were issued mostly on weekdays than on weekends. 
# MAGIC 
# MAGIC To check if this difference is significant, we run two-sided T-test assuming that the fine amount has normal distribution.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Hypotesis testing
# MAGIC 
# MAGIC Let *H0* be a null hypotesis
# MAGIC 
# MAGIC *H0: weekday and wekend tickets belong to the same population*

# COMMAND ----------

test1 <- t.test(set_fine_amount ~ daytype, data = df, paired = FALSE, var.equal = FALSE)
test1$p.value

# COMMAND ----------

# MAGIC %md
# MAGIC "Zero" result for P-value does not necessarily mean that the probability of the hull hypotesis is actually equal to zero. The point is that the probavility is too small and this is why it was rounded to zero. It means we should reject *H0* and agree with the alternative hypothesis that the noted difference between tickets issued on weekdays and weekends is statistically significant. Accordingly, chances to write a more expensive ticket are higher on weekdays than on weekends. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determine the average daily pattern
# MAGIC 
# MAGIC Next step is to discover how the fine amount changes over the time of the day. 

# COMMAND ----------

## Extract avg daily pattern

# Transformation of time from int to one-minute intervals of a day
df$time_of_infraction <- sprintf("%04d", df$time_of_infraction)
df$time_of_infraction <-  format(strptime(df$time_of_infraction, format="%H%M"), format = "%H:%M")
df$time_of_infraction <- factor(df$time_of_infraction)

# Aggregating data and averaging by each one-minute interval
average_pattern <- aggregate(x = list(amount = df$set_fine_amount), by = list(day_time = df$time_of_infraction), FUN = mean, na.rm = T)


# COMMAND ----------

plot(average_pattern$day_time, average_pattern$amount, type="l", 
     main = "Average daily pattern", xlab = "One-minute intervals", 
     ylab = "Total amount in $$$")


# COMMAND ----------

# MAGIC %md
# MAGIC According to the chart above, the total fine amount has two peaks, the first one before business hours and the second one in the end of the work day. 
# MAGIC The total fine amouny is minimal from 3 to 6 a.m. and fluctuating close to the mean average value for the rest of time periods. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determine the monthly pattern
# MAGIC 
# MAGIC Our next step is to discover if there are best or worst months to collect fine payments.

# COMMAND ----------

# Define month as factor variable
df$month <- as.Date(cut(df$date_of_infraction, breaks = "month"))

# COMMAND ----------

# Aggregating and charting
ggplot(data = df, aes(month, set_fine_amount)) +
    stat_summary(fun.y = sum, geom = "bar") +     # adds up all observations for the month
    labs(title = "Total monthly amounts") + labs(x = "Months", y = "Total amount in $$$")


# COMMAND ----------

# MAGIC %md 
# MAGIC Only two moths are really noticeable. December is corresponding to the lowest total amount since this is a holiday season. Second to the worst month is February. It is known as snowy and the coldest month of the year. Most probably car drivers prefer switching to the public transportation during this period. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determine the most profitable kinds of infractions

# COMMAND ----------

# MAGIC %md
# MAGIC It is interesting to determine if there are infractions which bring more tickets revenue. 

# COMMAND ----------

# The number of unique infraction descriptions
num_infr_descr <- length(unique(df$infraction_description))
num_infr_descr


# COMMAND ----------

## Aggregate data by infractions description
infr_descr <- aggregate(set_fine_amount ~ infraction_description, data = df, FUN = sum)

## Top-10 infractions by total $$$ amount
top10_infractions <- infr_descr[order(-infr_descr$set_fine_amount), ][1:10, ]

# COMMAND ----------

par(mar = c(10, 4, 3, 2), mgp = c(3, 1, 0), cex.main = 0.8)

barplot(top10_infractions$set_fine_amount, names.arg = top10_infractions$infraction_description, 
        main = "Top10 Infractions", col = "blue",
        cex.names = 0.5, las = 3, ylab = "Total amount in $$$")

# COMMAND ----------

# MAGIC %md
# MAGIC Also it would be interesting to know if the Top10 list is defferent fo weekdays and weekends. 

# COMMAND ----------

# Subsetting df to weekdays and weekends
weekday_df <- df[ which(df$daytype == "weekday"), ]
weekend_df <- df[ which(df$daytype == "weekend"), ]

# COMMAND ----------

# Aggregate data for weekdays
infr_weekdays <- aggregate(set_fine_amount ~ infraction_description, data = weekday_df, FUN = sum)

# Aggregate data for weekdays
infr_weekends <- aggregate(set_fine_amount ~ infraction_description, data = weekend_df, FUN = sum)

# COMMAND ----------

# Top-10 infractions on weekdays
top10_infr_weekdays <- infr_weekdays[order(-infr_weekdays$set_fine_amount), ][1:10, ]

# Top-10 infractions on weekends
top10_infr_weekends <- infr_weekends[order(-infr_weekends$set_fine_amount), ][1:10, ]

# COMMAND ----------

par(mar = c(10, 4, 3, 2), mgp = c(3, 1, 0), cex.main = 0.8)

barplot(top10_infr_weekdays$set_fine_amount, names.arg = top10_infr_weekdays$infraction_description, 
        main = "Top10 Infractions on Weekdays", col = "skyblue",
        cex.names = 0.5, las = 3, ylab = "Total amount in $$$")

# COMMAND ----------

par(mar = c(10, 4, 3, 2), mgp = c(3, 1, 0), cex.main = 0.8)

barplot(top10_infr_weekends$set_fine_amount, names.arg = top10_infr_weekends$infraction_description, 
        main = "Top10 Infractions on Weekends", col = "purple",
        cex.names = 0.5, las = 3, ylab = "Total amount in $$$")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determine the influence of geolocation 
# MAGIC 
# MAGIC It is reasonable to suggest that a proximity to attractions, cultural places and other points of interest can be a factor influencing the tickets revenue. 
# MAGIC To check this, we will build a heat map of locations where tickets were issued and compare it with points of interest locations. 

# COMMAND ----------

## Aggregate data by address
infr_address <- aggregate(set_fine_amount ~ location2, data = df, FUN = sum)

# COMMAND ----------

## Convert address from factor to char
infr_address$address <- NULL
infr_address$address <- as.character(infr_address$location2)

# COMMAND ----------

## Extract only non-empty addresses
infr_address <- subset(infr_address, location2 != "")


# COMMAND ----------

str(infr_address)

# COMMAND ----------

# MAGIC %md
# MAGIC To get longitude and latitude for street addresses (`location2` variable in our `infr_address` dataframe) we will use the Address Points Repository [2] which contains geocodes for over 500,000 addresses within the City of Toronto. 

# COMMAND ----------

library(foreign)

# Read Address Points database
# http://www1.toronto.ca/wps/portal/contentonly?vgnextoid=91415f9cd70bb210VgnVCM1000003dd60f89RCRD
ap_df <- read.dbf('/dbfs/mnt/my-hod-data/ADDRESS_POINT_WGS84.dbf', as.is = T)
str(ap_df)

# COMMAND ----------

# Select variables we are interested in
my_vars <- c("ADDRESS", "LFNAME", "LONGITUDE", "LATITUDE")
my_ap_df <- ap_df[my_vars]

# Bind number with street and convert to uppercase
my_ap_df$address = paste(my_ap_df$ADDRESS, my_ap_df$LFNAME, sep=" ")
my_ap_df$address <- toupper(my_ap_df$address)

# Merge datasets by key
geo_df <- merge(infr_address, my_ap_df, by = "address")
str(geo_df)

# COMMAND ----------

devtools::install_github("dkahle/ggmap")


# COMMAND ----------

## Mapping tickets locations 
library(ggmap)

centre = c(mean(geo_df$LONGITUDE, na.rm=TRUE), mean(geo_df$LATITUDE, na.rm=TRUE))
map = get_map(location = centre, zoom=11, scale=2, source = "google", maptype="roadmap")

map.plot = ggmap(map)
map.plot = map.plot + geom_point(data = geo_df, aes(x = LONGITUDE, y = LATITUDE, colour = log(set_fine_amount)))

# to use color brewer gradient scale:
library(RColorBrewer)
map.plot = map.plot + scale_colour_continuous(low = "blue", high = "red", space = "Lab", na.value = "grey50", guide = "colourbar")

print(map.plot)

# COMMAND ----------

# MAGIC %md 
# MAGIC Then we use the Places of Interest database [3] to build the related map.

# COMMAND ----------

## Loading Places of Interest (attractions etc.) data PLACES_OF_INTEREST_WGS84.dbf

poi_df <- read.dbf('/dbfs/mnt/my-hod-data/PLACES_OF_INTEREST_WGS84.dbf', as.is = T)
str(poi_df)

# COMMAND ----------

## Mapping POI
map.plot2 = ggmap(map)
map.plot2 = map.plot2 + geom_point(data = poi_df, aes(x = LONGITUDE, y = LATITUDE, colour = "red"))
map.plot2 = map.plot2 + theme(legend.position="none")
print(map.plot2)

# COMMAND ----------

# MAGIC %md
# MAGIC It is possible to notice that on the first map there are several points which have hotter color. And they have corresponding points on the second map. 
# MAGIC Thus, we can qualify that the proximity to some points of interests may be significant for tickets revenue. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC 
# MAGIC To sum up, we discovered the following influencing factors and dependencies:
# MAGIC 
# MAGIC * __Time of day__. The total dollar amount is maximal before business hours and in the end of business day and minimal in early morning.
# MAGIC 
# MAGIC * __Day of week__. The total dollar amount is much higher on weekdays that on weekend. The chances to write an expensive ticket are also higher on weekdays. 
# MAGIC 
# MAGIC * __Month__. The least profitable month is December. The second to worst is February. 
# MAGIC 
# MAGIC * __Top kinds of infractions__ are related to parking in prohibited zones, prohibited time and on private property.
# MAGIC 
# MAGIC * __Proximity to places of interest__ contributes to higher tickets revenue.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Further Research
# MAGIC 
# MAGIC In this project we managed to qualify some factors which have an influence on the total tickets revenue and quantify only one of them (days of week).
# MAGIC It would be worth it to answer a questions "how to quantify the influence of the proximity to places of interest" and "what kind of data we have/need for that?" In this case we would obtain a measurable criteria necessary for the formulation and hopefully succesful solution of the optimization problem on distribution/planning of limited parking enforcement resources. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### References
# MAGIC 
# MAGIC 1. [Parking Tickets Open Data](http://www1.toronto.ca/wps/portal/contentonly?vgnextoid=ca20256c54ea4310VgnVCM1000003dd60f89RCRD)
# MAGIC 2. [Address Points (Municipal) - Toronto One Address Repository](http://www1.toronto.ca/wps/portal/contentonly?vgnextoid=91415f9cd70bb210VgnVCM1000003dd60f89RCRD)
# MAGIC 3. [Places of Interest and Toronto Attractions Open Data](http://www1.toronto.ca/wps/portal/contentonly?vgnextoid=d90ac71db136c310VgnVCM10000071d60f89RCRD&vgnextchannel=8896e03bb8d1e310VgnVCM10000071d60f89RCRD)
