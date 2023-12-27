## Covid-19 Sentiment Analysis Project 


## Introduction
COVID-19 impacted public health, policy responses, and the economy. Public sentiment towards COVID-19 vaccine varies between countries, influencing policymakers' strategies to counter vaccine hesitancy. Understanding these sentiments is crucial in shaping approaches to future pandemics, aligning policies with public sentiment for better implementation.

Research Question: How does public sentiment regarding COVID-19 vaccines differ between the United Arab Emirates and the United States during the delta variant surge in 2021? This study hypothesizes more positive sentiments in the UAE compared to skepticism in the U.S. This divergence is attributed to differing societal norms and regimes. The U.S.'s diverse demographics create a unique context, influencing public sentiment across cultural and political lines. In contrast, the UAE, with its different regime type and norms, shows a higher trust in government-led vaccination efforts. This study analyzes public sentiment through YouTube videos: one featuring Dr. Fauci receiving the vaccine, and another endorsing the Sinopharm vaccine in the UAE. The analysis focuses on how societal differences between these nations and public policy on Covid-19 shape public perceptions and responses to the COVID-19 vaccine.

According to Our World in Data vaccination rates in the UAE exceeded 100% of the population due to the vaccination of non-residents. In the United States 70% are considered fully vaccinated. However, many people in the U.S. were required to take the vaccine to work. Some states mandated it for entry into restaurants, it became more of a requirement than an option, whereas in the UAE, it was viewed as an option. This difference will play a role in skepticism and misinformation, and I expect this to be proven in the sentiment analysis conducted in this project. Understanding these sentiments is crucial for the approach of future pandemics. An important question is: did lockdowns and the requirement of COVID passes for entry into restaurants and businesses deter people from vaccination?

**Exhibit A: U.S. V.S. U.A.E Vaccination Rates**



<img width="468" alt="image" src="https://github.com/mosalah2126/Finalproject.md/assets/144922510/a8962fe6-551b-44b3-8179-0f39290a1c03">



## Methodology: Data Collection and Code Analysis
This research will employ the APIs of YouTube, capitalizing on the platform's advantages for data collection. The primary focus will be on user comments from YouTube videos concerning Covid-19 from December 2020. This period provides a comprehensive snapshot of public sentiment during the spike of the delta variant. To supplement the sentiment analysis and gain deeper insights I will also utilize word cloud visualization to analyze the most frequent words, enabling me to identify dominant themes and sentiments that resonate in the public discourse.

### Section A: Rationale for Selecting YouTube as the Data Source

1. **Broad User Interaction**: YouTube's substantial and diverse user-base is ideal for assessing public sentiment. Users from a wide range of demographics engage with the platform's content, offering different perspectives that are representative of numerous cultural and social backgrounds.

2. **Topic-Centric Commentary**: YouTube comments are often directly connected to the content of the videos, yielding targeted and pertinent textual data for analysis unlike Reddit. This direct relationship is particularly advantageous for sentiment analysis when examining responses to the vaccine.

3. **Accessibility of Data**: The YouTube Data API enhances the systematic collection of comment data, enabling researchers to efficiently compile extensive datasets necessary for a solid analysis of public sentiment.

### Section B: Data Preprocessing Steps for Sentiment Analysis
1. **Exclusion of Long Comments**: In sentiment analysis, brevity is often associated with the clarity of sentiment. Therefore, comments exceeding 100 words have been filtered out, as the analysis aims to focus on concise feedback. For both videos, there were long comments unrelated to the video content. For instance, in the UAE video, there were Quranic verses, song lyrics, or lengthy anecdotes about life. In the U.S. video, comments ranged from mentions of animals to unrelated song lyrics. A thorough review of the comments on both videos revealed that more than 95% of the comments over 100 words did not pertain to the video content, leading to the decision to exclude them from the research.
2. **Link Removal**: To ensure the integrity of the dataset, comments containing links have been excluded. This precaution ensures that the sentiments analyzed are purely reflective of the comment content and not influenced by any external content. Many links led to advertisements or solicitations for payments, while others were links to additional YouTube videos.
3. **Specific Word Removal**: During preprocessing, common stopwords such as "and", "it", and "the" have been removed. These words typically do not convey sentiment and can skew the analysis if not addressed. 

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
1. **Script Setup and Imports**: The sentiment analysis project was underpinned by a Python script accurately designed to interact with the YouTube Data API and perform text analysis using the Natural Language Toolkit (NLTK). The script imports essential libraries: os and re for operating system interactions and regular expression operations, googleapiclient.discovery for accessing YouTube's API, and nltk.sentiment for sentiment analysis functionalities.

2. **API Initialization and Data Retrieval**: The script initiates by setting up the YouTube Data API, using the googleapiclient.discovery.build method to create a youtube service object. This object facilitates communication with YouTube's servers. The API key is hard-coded into the script. The extract_comment_threads function plays a pivotal role, taking a video_id and an optional max_results parameter (defaulting to 1000) to define the scope of data retrieval. Within this function, a while loop iterates through the pages of comments. Each page fetches up to 100 comments, controlled by the maxResults parameter in the youtube.commentThreads().list() method, until the desired number of comments is reached or no more pages are available (indicated by the absence of a nextPageToken).

3. **Comment Filtering and Processing**: As the script iterates through the comments, it applies several filters:

   #### Length Check: 
   It uses Python's string split() method to ensure each comment's word count does not exceed 100, filtering out lengthy and potentially off-topic responses.
   #### Link Detection:
   Regular expressions (re.search()) identify comments containing YouTube links, which are then excluded to maintain focus on the primary content.
   #### Stop Word Removal:
   (re.sub()) removes specified stopwords ("and", "it", "the") from the comments.

4. **Sentiment Analysis**: In my sentiment analysis, after filtering out the content from the comments, I leveraged NLTK's SentimentIntensityAnalyzer to dive into the emotional depth of each response. My primary focus was on the compound score; it's like the heartbeat of the comment, summing up its sentiment atmosphere. I added up these scores to get a total sentiment score and kept a tally of the positive, negative, and neutral counts. Once I wrapped up the analysis, my script  packaged each comment with its sentiment score, giving me a full picture of the sentiment mood. Moreover, I dialed in the video ID and orchestrated the extract_comment_threads function to unveil the total sentiment score and the breakdown of reactions. This offered a clear overview of the public sentiment, as if I was listening to the audience be positive or negative to the video content.

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


## Results, Discussion & Conclusion

The sentiment analysis conducted offers a comparative glimpse into the public sentiment regarding vaccines during a critical period of the pandemic. The UAE video depicted a general approval and trust in vaccination as evidenced by the positive total sentiment score of 14.4935. In contrast, the U.S. video featuring Dr. Fauci had a total sentiment score of -15.9412, reflecting negativity and skepticism towards the vaccine. Misclassifications was the biggest limitation in analyzing the U.S. video, where tools failed to accurately interpret sarcasm and nuanced language, leading to positive sentiments being incorrectly categorized as negative. The video's 133 comments initially perceived as positive were negative. An example of this can be seen in comments 119 and 128 in Exhibit B below. 

**Exhibit B: Sentiment Analysis Scores on Comments**
 

<img width="1195" alt="image" src="https://github.com/mosalah2126/Finalproject.md/assets/144922510/6b2e2b5b-2711-490e-a9a8-d91ad7d1cffc">






In the UAE context, the sentiment was more straightforward. The video reveals a positive outlook towards the vaccine, with comments commending the actions of the UAE government, expressing trust in the vaccine. Out of 104 unfiltered comments, only 13 are negative, and notably, 10 of these do not criticize the vaccine directly. Instead, they discuss preferences for other vaccines. This distinction suggests that the sentiment analysis may not fully capture the underlying positive attitude towards vaccination in general.

A key aspect of this research was the comparative study of sentiments between two diverse societies. The U.S. population appears more outspoken and negatively inclined, likely influenced by prior pandemic handling. In contrast, the UAE, with different societal norms and a distinct regime type, shows a more positive public opinion. This positivity may stem from the Emirati government's provision of free healthcare and education, which could be perceived as indicative of a government that cares well for its citizens. Consequently, this may lead to greater public trust and less negativity towards vaccination, a trend observable in the comments.

From a public health policy perspective, sentiment analysis serves as a crucial tool for assessing public sentiment. Positive sentiments can signify policy alignment, while negative sentiments may indicate a lack of trust. This analysis is vital for policy leaders to evaluate their public health strategies. This could point to the possibility that the stringent measures undertaken by the U.S. led many to view the U.S. as untrustworthy, fostering skepticism about their decisions. Meanwhile, the UAE, which had a less stringent COVID policy and vaccine process, ended up having most of their population vaccinated. 

The utilization of data visualization, particularly the creation of word clouds, further enhanced the understanding of the sentiment analysis results. Visuals provided a clear and concise way to compare sentiment scores, emphasizing the contrast between the largely positive sentiments surrounding the UAE video and the mixed sentiments in the U.S. video. In the Fauci video analysis, it's evident that the most common words are 'vaccine', 'Fauci', 'shot', and 'fake'. Terms like 'syringe' and 'saline' appear frequently and align with the overall sentiment analysis, indicating a predominance of negative comments. Other words such as 'water', 'vitamin', 'placebo', and 'fraud' also emerge in the word cloud. This sentiment reflects a hesitancy towards vaccination among this audience. The UAE video presents a different picture. Words like 'congratulation', 'vaccine', 'Sinopharm', and 'Alhamdulillah' (meaning 'thank God'), along with 'safe', 'accept', and 'love' dominate the word cloud. The word cloud reveals minimal skepticism and a generally positive sentiment, highlighting the differing attitudes towards government and vaccine trust in these two populations. See Exhibit C and D below.



**Exhibit C: Word Cloud UAE** 



![image](https://github.com/mosalah2126/Finalproject.md/assets/144922510/724908ee-6b7d-4af8-b6d6-80c9c5fe7dd3)





**Exhibit D: Word Cloud U.S.** 




![image](https://github.com/mosalah2126/Finalproject.md/assets/144922510/65d1c9e7-e1aa-4eb9-b922-efec7f7d1a4e)






In conclusion, despite some errors in classifying negative and neutral comments, the overall sentiment analysis scores aligned with the word cloud results, offering a somewhat accurate depiction of the sentiment in the videos. However, as shown in Exhibit E below, while the overall result was correct, the percentages were significantly off. Therefore, relying solely on sentiment analysis may not always yield a clear picture, given its inherent limitations in detecting sarcasm and inaccuracies in scoring comments.


**Exhibit E: Sentiment Analysis Chart** 








<img width="641" alt="image" src="https://github.com/mosalah2126/Finalproject.md/assets/144922510/a23bff1f-f9cd-4daf-b0eb-f0c8aebabd33">


