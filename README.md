## Covid-19 Sentiment Analysis Project 


## Introduction
COVID-19 had a severe impact, affecting public health, policy responses, and the global economy. Yet public sentiment towards COVID-19 and its vaccine varies strongly between countries. Understanding the public sentiment towards the COVID-19 vaccines is crucial for policymakers because it can inform outreach methods and strategies that need to be used to counter vaccine hesitancy, misinformation, and public concerns. The COVID-19 pandemic has brought significant attention to vaccine distribution and acceptance, making it crucial to examine how different regions perceive and react to vaccination efforts. Gaining insight into how people feel about the vaccine can shape our approach to future pandemics, helping to mitigate critics' voices and economic losses by ensuring that policies and communications are better aligned with public sentiment. This can lead to better policy implementation and greater public compliance, safeguarding public health and economic stability.

Research Question: How does public sentiment regarding COVID-19 vaccines differ between the United Arab Emirates and the United States during the delta variant surge in 2021? The hypothesis for this research question is that public sentiment towards COVID-19 vaccines in the United Arab Emirates will be more positive than in the United States, where a greater degree of skepticism is expected. The reason is it’s a tale of two different societies; each country has different societal norms and regimes in place. The United States is known for its diverse demographics, which creates a unique context to analyze sentiments as there are many different cultural backgrounds, political affiliations, and regional differences that can influence public sentiment. On the other hand, the United Arab Emirates represents a different regime type and different societal norms than the United States. Whether you’re an expatriate or not in the United Arab Emirates, when you see that most of the locals are trusting of the government and is getting the vaccine, it will somehow influence your decision. Whereas in the United States, it’s the opposite effect; because you have a large portion of Americans questioning the vaccine, as an expatriate in the United States, many begin to question it too, and we will see how that evolves into the public sentiment in the two YouTube videos that are analyzed. The first video analyzed involves Dr. Anthony Fauci, who was the former chief medical advisor to the president of the United States, taking the Covid-19 Vaccine. The second video analyzed is a video of the United Arab Emirates approving the Sinopharm vaccine, and the video is encouraging citizens and residents to get vaccinated.

## Comparative Policy Analysis: UAE and USA
The United Arab Emirates initially adopted a different approach in early 2020; they implemented a full-blown lockdown. But by mid-2020, they relaxed their lockdown rules and opened up their country to tourists. Tourists from all over the world started coming to live in Dubai as a permanent hub until their countries relaxed their laws. Once the vaccines were approved, the UAE encouraged their population to get vaccinated and required it for entry into the country for tourists. Residents and citizens with the vaccine could avoid quarantine, and those vaccinated who were in contact with a positive case no longer needed to quarantine. Immediately, there was a spike in the number of people taking the vaccine. According to Our World in Data vaccination rates in the UAE exceeded 100% of the population due to the vaccination of non-residents. Meanwhile, in the United States, many people were required to take the vaccine to work. Some states even mandated it for entry into restaurants, and the lockdown was more intense. It became more of a requirement than an option, whereas in the UAE, it was perceived as an option. I believe this difference will play a huge role in skepticism and misinformation, and I expect this to be proven in the sentiment analysis conducted in this project.


<img width="468" alt="image" src="https://github.com/mosalah2126/Finalproject.md/assets/144922510/a8962fe6-551b-44b3-8179-0f39290a1c03">



Understanding these sentiments is crucial as it shapes our approach to future pandemics. Questions arise: Will intense lockdowns and the requirement of COVID passes for entry into restaurants and businesses deter people from vaccination? Does mandating vaccines in workplaces lead to misinformation and increased skepticism? Alternatively, could a strategy involving relaxed rules without full-blown lockdowns encourage vaccination? The contrasting methods employed by these two countries make this sentiment analysis vital. It enables us to discern, from the tone of the comments, which country garnered more support than the other.

## Methodology: Data Collection and Analysis
This research will employ the APIs of YouTube, capitalizing on the platform's unique advantages for data collection. The primary focus will be on user comments from YouTube videos concerning Covid-19 from December 2020. This period provides a comprehensive snapshot of public sentiment during the spike of the delta variant.To supplement the sentiment analysis and gain deeper insights I will also utilize word cloud visualization to analyze the most frequent words, enabling me to identify dominant themes and sentiments that resonate in the public discourse.
### Section A: Rationale for Selecting YouTube as the Data Source

1. **Broad User Interaction**: YouTube's substantial and diverse user base is ideal for assessing public sentiment. Users from a wide range of demographics engage with the platform's content, offering a panorama of perspectives that are representative of numerous cultural, social, and political backgrounds.

2. **Topic-Centric Commentary**: YouTube comments are often directly connected to the content of the videos, yielding targeted and pertinent textual data for analysis. This direct relationship is particularly advantageous for sentiment analysis when examining responses to the COVID-19 vaccine and related policies.

3. **Accessibility of Data**: The YouTube Data API enhances the systematic collection of comment data, enabling researchers to efficiently compile extensive datasets necessary for a solid analysis of public sentiment. This accessibility is a vital component in gathering the volume of data required for a substantial sentiment analysis.

### Section B: Data Preprocessing Steps for Sentiment Analysis
1. **Exclusion of Long Comments**: In sentiment analysis, brevity is often associated with the clarity of sentiment. Therefore, comments exceeding 100 words have been filtered out, as the analysis aims to focus on concise and more pointed feedback, which can be assessed more reliably. For both YouTube videos, there were long comments unrelated to the video content. For instance, in the UAE video, there were instances of Quranic verses, song lyrics, or lengthy anecdotes about life. Similarly, in the U.S. video, comments ranged from mentions of animals to unrelated song lyrics. A thorough review of the comments on both videos revealed that more than 95% of the comments over 100 words did not pertain to the video content, leading to the decision to exclude them from the research entirely.
2. **Link Removal**: To ensure the integrity of the dataset, comments containing links have been excluded. This precaution ensures that the sentiments analyzed are purely reflective of the comment content and not influenced by any external content that the commenters may link to. Many links led to advertisements or solicitations for payments, while others were links to additional YouTube videos, which could potentially distract from the sentiment analysis.
3. **Specific Word Removal**: During preprocessing, common stopwords such as "and", "it", and "the" have been removed. These words typically do not convey sentiment and can skew the analysis if not addressed. Eliminating these allows the algorithm to concentrate on terms that are more indicative of sentiment.

Initially, the plan included the removal of emojis; however, upon further consideration, emojis are recognized as expressive tools that can significantly contribute to understanding sentiment. 

### Section C: Sentiment Analysis Code and Explanation


`````
import nltk
nltk.download('vader_lexicon')
import os
import re
import googleapiclient.discovery
from nltk.sentiment import SentimentIntensityAnalyzer
API_KEY = 'AIzaSyAhLQnohaqHgLHxG2yfyIDxoCkGnroNtb4'
def extract_comment_threads(video_id, max_results=1000):
    """
    Extract comment threads from a YouTube video using the YouTube Data API.

    :param video_id: The ID of the YouTube video.
    :param max_results: The maximum number of comments to retrieve (default is 1000).
    :return: A list of comment threads.
    """
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    nextPageToken = None
    sia = SentimentIntensityAnalyzer()
    total_score = 0
    positive_count, negative_count, neutral_count = 0, 0, 0

    while len(comments) < max_results:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            textFormat='plainText',
            pageToken=nextPageToken
        ).execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            if len(comment.split()) <= 100 and not re.search(r'(youtu\.be/|youtube\.com/watch\?v=)', comment):
                filtered_comment = re.sub(r'\b(and|it|the)\b', '', comment, flags=re.IGNORECASE)
                sentiment_score = sia.polarity_scores(filtered_comment)['compound']
                total_score += sentiment_score


                if sentiment_score > 0:
                    positive_count += 1
                elif sentiment_score < 0:
                    negative_count += 1
                else:
                    neutral_count += 1

                comments.append((filtered_comment, sentiment_score))

            if len(comments) >= max_results:
                break

        if 'nextPageToken' in response:
            nextPageToken = response['nextPageToken']
        else:
            break

    return comments, total_score, positive_count, negative_count, neutral_count

  if __name__ == "__main__":
    video_id = 'cQPoa9cFUyc'
    max_results = 1000

    comments, total_score, positive_count, negative_count, neutral_count = extract_comment_threads(video_id, max_results)

    print(f"Total Sentiment Score: {total_score}")
    print(f"Positive Comments: {positive_count}")
    print(f"Negative Comments: {negative_count}")
    print(f"Neutral Comments: {neutral_count}")

    for idx, (comment, score) in enumerate(comments, start=1):
        sentiment = 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'
        print(f"{idx}. {comment} - Sentiment: {sentiment} (Score: {score})") 
`````

Here is an explanation of the code: 
1. **Script Setup and Imports**: The sentiment analysis project was underpinned by a Python script accurately designed to interact with the YouTube Data API and perform  text analysis using the Natural Language Toolkit (NLTK). The script imports essential libraries: os and re for operating system interactions and regular expression operations, googleapiclient.discovery for accessing YouTube's API, and nltk.sentiment for sentiment analysis functionalities.

2. **API Initialization and Data Retrieval**: The script initiates by setting up the YouTube Data API, using the googleapiclient.discovery.build method to create a youtube service object. This object facilitates communication with YouTube's servers. The API key is hard-coded into the script. The extract_comment_threads function plays a pivotal role, taking a video_id and an optional max_results parameter (defaulting to 1000) to define the scope of data retrieval. Within this function, a while loop iterates through the pages of comments. Each page fetches up to 100 comments, controlled by the maxResults parameter in the youtube.commentThreads().list() method, until the desired number of comments is reached or no more pages are available (indicated by the absence of a nextPageToken).

3. **Comment Filtering and Processing**: As the script iterates through the comments, it applies several filters:

   #### Length Check: 
   It uses Python's string split() method to ensure each comment's word count does not exceed 100, filtering out    lengthy and potentially off-topic responses.
   #### Link Detection:
   Regular expressions (re.search()) identify comments containing YouTube links, which are then excluded to maintain    focus on the primary content.
   #### Stop Word Removal:
   (re.sub()) removes specified stopwords ("and", "it", "the") from the comments.

4. **Sentiment Analysis**: In my sentiment analysis, after filtering out the content from the comments, I leveraged NLTK's SentimentIntensityAnalyzer to dive into the emotional depth of each response. My primary focus was on the compound score—it's like the heartbeat of the comment, summing up its sentiment atmosphere. I added up these scores to get a total sentiment score and kept a tally of the positive, negative, and neutral counts. Once I wrapped up the analysis, my script  packaged each comment with its sentiment score, giving me a full picture of the sentiment mood. Moreover, I dialed in the video ID and orchestrated the extract_comment_threads function to unveil the total sentiment score and the breakdown of reactions. This offered a clear overview of the public sentiment, as if I was listening to the audience be positive or negative to the video content.

### Section D: Word Cloud Code and Explanation

`````
!pip install wordcloud
import os
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import googleapiclient.discovery
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

API_KEY = 'AIzaSyAhLQnohaqHgLHxG2yfyIDxoCkGnroNtb4'  


def extract_comment_threads(oGNHCpmKlVw, max_results=1000):
    """
    Extract comment threads from a YouTube video using the YouTube Data API.

    :param video_id: The ID of the YouTube video.
    :param max_results: The maximum number of comments to retrieve (default is 1000).
    :return: A list of comment threads.
    """
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    nextPageToken = None

   
    sia = SentimentIntensityAnalyzer()

    while len(comments) < max_results:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            textFormat='plainText',
            pageToken=nextPageToken
        ).execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            
            if len(comment.split()) <= 100 and not re.search(r'(youtu\.be/|youtube\.com/watch\?v=)', comment):
                filtered_comment = re.sub(r'\b(and|it|the)\b', '', comment, flags=re.IGNORECASE)
                
                comments.append(filtered_comment)

            if len(comments) >= max_results:
                break

        if 'nextPageToken' in response:
            nextPageToken = response['nextPageToken']
        else:
            break

    return comments

if __name__ == "__main__":
    video_id = 'oGNHCpmKlVw'  
    max_results = 1000  

    comments = extract_comment_threads(video_id, max_results)

    all_comments = ' '.join(comments)

    wordcloud = WordCloud(width = 800, height = 400, background_color='white').generate(all_comments)

    plt.figure(figsize=(15,7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
`````

Here is an explanation of the code:
1. **Setting Up for Visualization**: In this part of the script, I start by importing the necessary libraries. I use os for operating system interactions, re for regular expression operations, matplotlib.pyplot for creating visual plots, googleapiclient.discovery for accessing YouTube's API, and nltk for natural language processing. This setup is crucial as it prepares my script for both visual representation through a word cloud and in-depth sentiment analysis using NLTK's VADER. 
2. **YouTube API and Comment Extraction with Sentiment Metrics**: Similar to the previous script, the YouTube Data API is set up with an API key, allowing the script to fetch comments from a specified YouTube video. NLTK is also used.
3. **Word Cloud Generation from Filtered Comments**: After processing the comments and analyzing their sentiments, the script concatenates them into a single string. The WordCloud class then takes this string and generates the data. 
4. **Visualization**: Using matplotlib, the script creates a plot to display the word cloud. This visualization is not just a collection of words but a reflection of the sentiments offering the prevailing words in the comments. 


## Methodology: Data Collection and Analysis


