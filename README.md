# Analysis of YouTube 2023 Statistics

![Logo YouTube](/assets/yt_logo_rgb_dark.png "YouTube")

## Analysis objectives
The main objective is to analyze the relationships between the different categories with potential earnings. To detect which categories are the ones that have a higher return with respect to the number of subscribers needed.

I perform a previous exploration of the dataset to evaluate the quality of the data and to know in detail its content. As far as possible, I try to validate the data and preprocess the data for proper analysis.

## Dataset
### Source
This dataset is original from the **Kaggle** platform, <a href = 'https://www.kaggle.com/datasets/nelgiriyewithana/global-youtube-statistics-2023' target="_blank"> here </a> is the link to download the latest version if you are interested. 

Kaggle is a popular online platform and community for data science and machine learning enthusiasts. It was founded in 2010 and has since become one of the largest and most well-known platforms for data science competitions, datasets, and collaboration among data scientists, machine learning engineers, and researchers.

### Variables description
* **rank**: Position of the YouTube channel based on the number of subscribers
* **Youtuber**: Name of the YouTube channel
* **subscribers**: Number of subscribers to the channel
* **video views**: Total views across all videos on the channel
* **category**: Category or niche of the channel
* **Title**: Title of the YouTube channel
* **uploads**: Total number of videos uploaded on the channel
* **Country**: Country where the YouTube channel originates
* **Abbreviation**: Abbreviation of the country
* **channel_type**: Type of the YouTube channel (e.g., individual, brand)
* **video_views_rank**: Ranking of the channel based on total video views
* **country_rank**: Ranking of the channel based on the number of subscribers within its country
* **channel_type_rank**: Ranking of the channel based on its type (individual or brand)
* **video_views_for_the_last_30_days**: Total video views in the last 30 days
* **lowest_monthly_earnings**: Lowest estimated monthly earnings from the channel
* **highest_monthly_earnings**: Highest estimated monthly earnings from the channel
* **lowest_yearly_earnings**: Lowest estimated yearly earnings from the channel
* **highest_yearly_earnings**: Highest estimated yearly earnings from the channel
* **subscribers_for_last_30_days**: Number of new subscribers gained in the last 30 days
* **created_year**: Year when the YouTube channel was created
* **created_month**: Month when the YouTube channel was created
* **created_date**: Exact date of the YouTube channel's creation
* **Gross tertiary education enrollment (%)**: Percentage of the population enrolled in tertiary education in the country
* **Population**: Total population of the country
* **Unemployment rate**: Unemployment rate in the country
* **Urban_population**: Percentage of the population living in urban areas
* **Latitude**: Latitude coordinate of the country's location
* **Longitude**: Longitude coordinate of the country's location

## Exploration data
### Top 5 Youtubers
| Ranking | Youtuber                  | Subscribers |
|:-------:|---------------------------|-------------|
| 1       | T-Series                  | 245 million |
| 2       | YouTube Movies            | 170 million |
| 3       | MrBeast                   | 166 million |
| 4       | Cocomelon - Nursery Rhymes| 162 million |
| 5       | SET India                 | 159 million |

### Qualitative variables
* Top 5 Category:
    * Entertainment 24.2 %
    * Music 20.3 % 
    * People & Blogs 13.3 %
    * Gaming 9.4 %
    * Comedy 6.9 %

* Top 5 Country:
    * United States 31.5 %
    * India 16.9 %
    * Brazil 6.2 %
    * United Kingdom 4.3 %
    * Mexico 3.3 %

* Top 5 Channel Type:
    * Entertainment 30.6 %
    * Music 21.7 %
    * People 10.2 %
    * Games 9.8 %
    * Comedy 5.1 %

### Count of nulls by variables
* subscribers_for_last_30_days               337
* Longitude                                  123
* Latitude                                   123
* Urban_population                           123
* Unemployment rate                          123
* Population                                 123
* Gross tertiary education enrollment (%)    123
* Country                                    122
* Abbreviation                               122
* country_rank                               116
* video_views_for_the_last_30_days            56
* category                                    46
* channel_type_rank                           33
* channel_type                                30
* created_date                                 5
* created_month                                5
* created_year                                 5
* video_views_rank                             1

### Statistics nulls
* There are 1616/27860 with nulls, a **5.8 % about total**. 
* There are 441/995 of rows have any nulls, a **44.32 % about total of rows**.

### Outliers per variable
![Data Exploration: Outliers](/assets/exploration_outliers.png "Outliers")

### Handling nulls and outliers
In the notebook more outliers and null records are treated. Here I show the most relevant for the development of the objectives of the analysis.

1. Pearson correlation method on subscribers and video views for last 30 days

#### 
**Code:**
```
mask = data.loc[:, ['subscribers', 'subscribers_for_last_30_days']].isnull().any(axis = 1)
df = data[~mask]
corr = df['subscribers'].corr(df['subscribers_for_last_30_days'])
print(f'The correlation of subscribers vs subscribers for last 30 days is: { "{:.2f}".format(corr) }')

mask = data.loc[:, ['video_views_for_the_last_30_days', 'subscribers_for_last_30_days']].isnull().any(axis = 1)
df = data[~mask]
corr = df['video_views_for_the_last_30_days'].corr(df['subscribers_for_last_30_days'])
print(f'The correlation of video views for the last 30 days vs subscribers for last 30 days is: { "{:.2f}".format(corr) }')
```

**Output:**
```
    The correlation of subscribers vs subscribers for last 30 days is: 0.31
    The correlation of video views for the last 30 days vs subscribers for last 30 days is: 0.45
```

After to explore the data, and finding no correlation of logical variables to estimate the null values of the variable subscribers for last 30 days. For this example, I decided to remove this from dataset.

2. Category nulls are filled with the value "unkown".

The rest of the nulls are caused by ignorance of the country. They will be deleted when the data will be processed. For the rest of the analysis the data are perfectly valid. So they will not be removed from the dataset.

For this example, it makes no sense to estimate such data and reach possible erroneous conclusions.

## Data Analisys
### Subscribers by Category
The figure shows how a small number of youtubers have many subscribers, while the rest are concentrated in many of them.

![Data Analisys: Subscribers by Category](/assets/subscribers_by_category.png "Subscribers by Category")

### Earings by Category
As might be expected, the more specific categories of high-dollar sectors have the highest revenue potential. 

The most generalist ones, are very well known youtubers and this makes us think that they can be the ones that generate more income. But these earn very little per subscriber and the business model focuses on the need to have a large number of subscribers.

Personally, I am surprised by the travel and events position. I thought that, being a mature sector and focused on adult audiences with purchasing power, it would have a better position in the graph.

![Data Analisys: Earings by Category](/assets/earings_by_subscriber.png "Earings by Category")

### Range Earings by Subscriber
The range between the highest and lowest estimates is shown here. Increasing the estimate also increases the range of uncertainty between the highest or lowest value of the prediction.

![Data Analisys: Range Earings by Category](/assets/range_earings_by_subscriber.png "Range Earings by Category")

### Range Earings by Subscriber vs Earings per year
The following figure shows the relationship between the earing per subscriber and the estimated highest earings in millions by category. Earlier we saw that the more niche categories have better earnings estimates. And how the most generalist and most popular categories have the highest revenue per year and average or even low revenue per subscriber.

![Data Analisys: Range Earings per Subs vs Earings per year](/assets/earings_per_subs_vs_earings_per_year.png "Range Earings per Subscriber")

## Conclusions
If we want to make money through a YouTube channel, we must not only create good content and related to our target customer. We must also be attentive to what sector we are going to focus on.

We have seen what, through this small data exploration, depending on whether it is a more generalist or more niche sector. It is essential to know whether we should focus our efforts on creating a large community or if the effort is to optimize the quality of the community as much as possible.

There are sectors that, in order to have a good ratio of earnings per subscriber, it is more important to generate a community as close as possible to our target customer. To be able to generate a viable project with a small critical mass.

However, there are others where we need a huge critical mass. And that, based on generating viability in adding very little money through thousands or millions of subscribers.

#### **Disclaimer**
In addition to being an initial data exploration, there are also many other factors to take into account when creating the optimal content and marketing strategy for each case.




