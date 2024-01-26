# Customer Review Summarization
This project utilizes the latest extractive and abstractive summarization algorithms to summarize a corpus of customer reviews. The catch is to create better summaries, as measured by  BLEU (Bilingual Evaluation Understudy) scores, than applying either one of these techniques. 

**BLEU** Scores - which indicates how similar the candidate text is to the reference texts (think of it like comparing computer-generating summaries with human-generated summaries)

**α** - length of desired summary/length of the input corpus

## Mechanics - 

1. **Extractive summarization Algorithms** - Algorithms like text rank/ TF-IDF are pretty good at finding unique keywords and phrases. This ability can be utilized to generate the first-level summary.
2. **Abstractive Summarization** - These algorithms are really good in alternative representations. This allows to condense the same information into smaller sequences of words. Here, we would be using a sequence-to-sequence model with LSTMs as base units.
3. **Attention Layer** - An attention layer is added to the model to further increase the model accuracy and perform better outputs for larger inputs.
4. **Combining them all** - I utilized Python's scripting ability to automate the whole process.
     * Results from the first level of summarization are stored in a text file.
     * The Python automation script then calls the abstractive summarizer function, which reads from the text input and then applies LSTM to it.
     * Within the LSTM layer an attention layer is added to improve the correlation between distant words and find better summaries. 


## Dataset

Amazon customer Reviews dataset from Kaggle - https://www.kaggle.com/code/currie32/summarizing-text-with-amazon-reviews/input?select=Reviews.csv

## Algorithms

### TF-IDF
       
![image](https://github.com/Apyarn95/Customer-Reviews-Summarization/assets/68912820/e21f17c0-e944-4037-ad88-3212e57549c0)

### Text Rank
Steps for Text Rank Algorithm

![image](https://github.com/Apyarn95/Customer-Reviews-Summarization/assets/68912820/8ba5b80c-5136-47c4-90ff-5e69df2b4de6)

### LSTM models

![image](https://github.com/Apyarn95/Customer-Reviews-Summarization/assets/68912820/f2615c7d-4c9b-449c-8284-f54761eb21b5)


### Attention Layer

Please refer to this wiki article about how to apply attention mechanisms to LSTM - https://primo.ai/index.php?title=Bidirectional_Long_Short-Term_Memory_%28BI-LSTM%29_with_Attention_Mechanism

![image](https://github.com/Apyarn95/Customer-Reviews-Summarization/assets/68912820/5027a726-dacf-475f-be13-b2a29405fb66)



