---
layout: post
title: "Creating a no-fluff Telegram-based newsletter that you can customize – Part 1"
date: 2025-04-02
post_score_hist: "assets/images/dd_post_score_hist.png"
post_size_hist: "assets/images/dd_post_size_hist.png"
wordcloud: "assets/images/dd_wordcloud.png"
conceptual_model: "assets/images/dd_conceptual_model.png"
confusion_matrix: "assets/images/dd_conf_matrix_bert_test.png"
---

### The Why

Telegram turned my media consumption habits and preferences upside down with the rollout and popularization of 'channels'. Channels are essentially just curated message boards, where only one party can post on the main board, while others might be able to comment/interact with it in a different way. Channels reporting news are extremely popular on Telegram, getting millions of subscribers. Consuming news through snippets that someone carefully tailored to your political bubble's status quo is just much more fun than reading long articles. Also it drives engagement growth. 

Despite now reading news snippets, you still find yourself spending just as much time as you would were you to read the original article. That is because news happens all the time, and with a shorter pipeline from 'spawned' to 'reported', more and more things are now being posted that are not so newsworthy. All the news channels I am subscribed to produce more than 300 snippets per day combined, which is almost as much as Kanye posts on Twitter. Some of the problems it creates:

- You don't have time to critically assess what you read
- You clutter your mind with ~~bullshit~~ unimportant things
- You waste time
- ...

### Conceptual Model

Heavily inspired by [newsminimalist](https://www.newsminimalist.com/) and my own laziness, I created a Telegram bot that takes news posts from channels you select, analyzes their importance and spits out only the highest-performing ones. A rough conceptual model: 

![]({{ page.conceptual_model | relative_url }}){: .small-image}

In words: through Telegram's frontend, the user selects the channels they want to subscribe to. These channels are stored in a single table in a local SQLite database. Every morning, a Cron job activates, pulling all the posts you missed the previous day. Those posts are cleaned and turned into arrays of numbers, which are then classified as either worthy or unworthy of your attention. Finally, the posts are sent back to the user. 

The project turned out to be a bit larger than I expected, so I am dedicating two posts to it. In this one, I will talk about data collection, preprocessing and the machine learning behind it. 

### The Data

I needed a lot of training data, so I chose six news channels that had consistently delivered news in a formal or semi-formal format. The channels are 'Раньше всех. Ну почти.', 'РИА Новости', 'Varlamov News', 'Медуза — LIVE', 'BRIEFLY', and 'Ньюсач/Двач'. From each channel, I extracted 10,000 most recent posts with the following functions:

```python
from telethon import TelegramClient
import asyncio

async def fetch_messages(channel, limit):
    """ Fetch messages from single channel."""
    try:
        message_list = []
        async for message in client.iter_messages(channel, limit):
            message_list.append(message.text)
        print(f'Finished fetching messages from {channel}')
        return channel, message_list
    except Exception as e:
        print(f'Error fetching messages from {channel}: {e}')
        return channel, []

async def main():
    """Fetching messages from all channels."""
    try: 
        post_history = {}
        for channel in channels: 
            dialogs = await client.get_dialogs()
            channel, messages = await fetch_messages(channel, limit)
            if messages: 
                post_history[channel] = messages
        with open('./data/post_history.json', 'w') as f:
            json.dump(post_history, f, ensure_ascii=False)
    except Exception as e: 
        print(f'Error: {e}')


```

Afterwards I cleaned them, removing links, deleting markdown symbols,etc., and was left with around 50,000 posts. 

#### Scoring 50,000 News Articles on Importance

I left data labeling to LLMs, once again drawing heavy inspiration from [newsminimalist](https://www.newsminimalist.com/). I used the following prompt to generate multiple scores for each snippet: 

> ##### Instruction:
Analyze the following Russian news snippet and provide scores for each of the following dimensions. Please respond in the format: "Dimension: Score" (e.g., "Scale: 8"). Ensure your response is concise and only includes the dimension name followed by a score.
##### Dimensions:
1. **Scale**: How broadly the event affects humanity.  
2. **Impact**: How strong the immediate effect is.  
3. **Novelty**: How unique and unexpected is the event.  
4. **Potential**: How likely it is to shape the future.  
5. **Legacy**: How likely it is to be considered a turning point in history or a major milestone.  
6. **Positivity**: How positive is the event.  
##### Input:
[Insert Russian news snippet here]
##### Response Format:
Scale: [Score]  
Impact: [Score]  
Novelty: [Score]  
Potential: [Score]  
Legacy: [Score]  
Positivity: [Score]

Final score weighs the factors and is calculated with the following weights and function:

```python
coefs = {
    "Scale": 0.25,
    "Impact": 0.25,
    "Novelty": 0.05,
    "Potential": 0.25,
    "Legacy": 0.15,
    "Positivity": 0.05
}

def calculate_scores(responses):
    final_scores = []
    for score in responses: 
        all_scores = re.findall(r'\d', score)
        all_scores= [int(s) for s in all_scores]
        final_score = round(sum([coef * all_scores[i] for i, coef in enumerate(coefs.values())]),1)
        final_scores.append(final_score)
    return final_scores

```
These weights reflect what I personally expect from my newsletter, so you might find them somewhat arbitrary. That is, however, not the main problem with my approach. To radically decrease investments into this inherently net-negative project, I opted for a smaller, cheaper LLM: ChatGPT-4o mini. This later comes back to bit me in the ass, as I will mention. That said, I did conduct multiple sanity checks to ensure the scores made any sense - and they did! Here are a couple of examples (translated from Russian to English):

| TG Post                                                                                                           | Importance by 4o-mini|
|-----------------------------------------------------------------------------------------------------------------------|---------------------------|
| Reuters: Trump says he will discuss power plants, territories with Putin in talks to end military conflict in Ukraine   | 8.3                       |
| VPN without traffic and device restrictions from 152₽ per month                                                       | 3.7                       |
| Slovakia will not participate in any military missions in Ukraine and will not provide "a single cent" in military aid to Kyiv, Prime Minister Fico said. | 5.2                       |
| Escort girl entered the Betting League chat and left with a jackpot                                                    | 3.7                       |

ChatGPT-4o successfully ranked both commercials low, while boosting the score of the important snippet. Seeing these scores, one might think we are close to AGI, since even a low-cost model understands that Slovakia's politics is unimportant. Despite many good hits, the model, just like an average Internet user, lacked critical thinking and ranked especially arrogant commercials, lies, and random things highly: 

| TG Post            | Importance by 4o-mini   |
|--------------------|-----------|
| ChatGPT and Midjourney to be added to Telegram!      | 6.2|
| Prada and Axiom Space unveil spacesuit for first moon landing in 50 years    | 8.2|
| Trump: 'We can end the war in Ukraine within weeks'     | 8.1|

Below you can find the distribution of the scores. Something like a normal distribution with a mean of around 6.5 can be observed. I believe the average is higher than what [newsminimalist](https://www.newsminimalist.com/) reports because a cheaper model is much more impressionable and, as I already mentioned, can fall for simple tricks. 

![]({{ page.post_score_hist | relative_url }}){: .small-image}

Bad scoring practices create the _shit in, shit out_ problem: if the scores lack intrinsic insight, no model will be able to work with them effectively. This does not have much impact on my project much since I am just happy to be here [and showcase my skills].

### Quick Dive Into the Data

Below is a wordcloud I created from the entirety of the data that I gathered. I lemmatized the text before to get a more general view of the corpus. For the English speakers: most popular words were "year", "say", "also", "Russia", "Trump", which makes sense.

> To lemmatize means to find the 'lemma' or root form of a word. 

![]({{ page.wordcloud | relative_url }}){: .small-image}

Now let's take a look into the sizes of posts. The distribution is heavily skewed to the right, with most posts having around 20 words in them. A small bump at around 300 is caused by the channel 'BRIEFLY', since they group snippets together and post larger messages. There is a moderate correlation between post size and post score, with a Pearson correlation coefficient of 0.37. I believe it is caused by posts from 'BRIEFLY' getting ranked highly since they contain more information in a single post.

![]({{ page.post_size_hist | relative_url }}){: .small-image}

<blockquote class="orange">
  Pearson correlation is a measure of a linear relationship between two continuous variables.
</blockquote>

### Translating Human Language Into Numbers

Collecting scores only gives us the response variable, and we still need to use something as a predictor. We cannot, unfortunately, use documents as they are for model training - we need to translate them from human language into a format that machines understand, i.e., numbers. I applied two methods of calculating document embeddings: training own Doc2Vec model and using a pre-trained BERT model. I trained my own Doc2Vec model and used the pre-trained BERT model rubert-tiny-turbo to generate embeddings. I chose it because of its outstanding size/performance ratio. Benchmark results can be found [here](https://huggingface.co/sergeyzh/rubert-tiny-turbo).

<blockquote class="red">
  Doc2Vec is an extension of Word2Vec, which embeds words in a vector space using a shallow neural network. Doc2Vec we act as if all documents have another word-like vector which contributes to all training predictions and it a doc-vector.
</blockquote>

<blockquote class="green">
  BERT is a language model that understands text by considering the context of each word in both directions. It uses the transformer architecture, which applies attention mechanisms to capture relationships between words regardless of their position in a sentence.
</blockquote>

### Training a Model to Predict Importance

Essentially, the problem I need to tackle is classifying news snippets as important or unimportant. Therefore, to keep things simple, I decided to choose a threshold value of **6** and fit a logistic regression model to the data. Doc2Vec requires two important hyperparameters: vector size and epochs (i.e., how many times the model goes through the corpus). To find the optimal values, I fitted a logistic regression for multiple combinations, with the best one achieving a balanced accuracy of 71% and an F1 score of 0.65. Embeddings generated by a pre-trained BERT model showed much better performance, with accuracy of almost 80% and F1 score of 0.75. F1 score is calculated as follows:

$$F_1 = \frac{2 \cdot (\text{Precision} \cdot \text{Recall})}{\text{Precision} + \text{Recall}}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

**Doc2Vec (test):**

| Vector Size | Epochs | Balanced Accuracy | F1 Score |
|-------------|--------|-------------------|----------|
| 10          | 40     | 0.692             | 0.630    |
| 10          | 60     | 0.699             | 0.639    |
| 30          | 40     | 0.684             | 0.627    |
| **30**          | **60**     | **0.705**             | **0.645**    |
| 50          | 40     | 0.682             | 0.623    |
| 50          | 60     | 0.687             | 0.633    |
| 100         | 40     | 0.684             | 0.629    |
| 100         | 60     | 0.689             | 0.632    |
| 150         | 40     | 0.681             | 0.620    |
| 150         | 60     | 0.692             | 0.630    |


**rubert-tiny-turbo:**

| Dataset | Balanced Accuracy | Accuracy | F1 Score |
|---------|-------------------|----------|----------|
| Train   | 0.788             | 0.788    | 0.748    |
| Test    | 0.790             | 0.790    | 0.750    |


### Closing thoughts

In the future, I would like to re-do this project with proper data labeling and model training. I believe that a 'great content filter' would be valuable and could be implemented in various ways across different platforms. If you feel like participating, either through labor, funding or in any other way, feel free to reach out.