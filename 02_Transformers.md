# Transformer Architecture

![alt text](<resources/Screenshot 2026-04-25 194422.png>)
![alt text](<resources/Screenshot 2026-04-25 195055.png>)
![alt text](<resources/Screenshot 2026-04-25 195238.png>)

Encoder decorder limitation was when sentence length increases the context passed to decoder was not enough. Bleau Score decreases.

This problem can be solved by Attention mechanism.

![alt text](<resources/Screenshot 2026-04-25 195635.png>)

With this we are able to provide additional context to the decorder.
In the above we are using Bi directional LSTM, RNN.

## Problem with Attention Mechanism and how Transformers fix it
### Parallel input processing
- We send each word based on timestamp. So we cannot parallaly send all the words in a sentence. So the model is not scalable wrt to training
- Transformers never use LSTM RNN encoders decoders, rather they use **Self Attention Module** (All the words will be parallely sent to the encoder)
- We Will learn here positional encoding since we are sending all inputs together
- With Transformers : As we keep increasing data set we will get some state of the art (SOTA) models.
- Transformers also work in multi model task : NLP + Image
- Transformers have changed the AI space
- From GPT BERT -> Transfer Learning -> SOTA models -> like DALLE
- Various LLM models are used in Generative AI
- ![alt text](resources/www.youtube.com_watch_v=3bPhDUSAUYI&t=20s.png)
### Contextual Embedding
- When we pass our input through an Embedding layer, it converts all the words in the sentence to a vector we call embedding vector using word2vec for example.
- But these does not have Contextual vector.
  - for ex: **I am Vishnu and I play Badminton.**
    - In above line **I** is realted to **Vishnu** also **Badminton**  is associated with **Vishnu**
    - So we need some contextual Embedding layer that can produce vectors that are not just plaijn vectors in embedding vector but should also have some context about the relation between words in the sentence.
    - THIS PROBLEM IS SOLVE BY **SELF ATTENTION MODEL**
    - Becuase of this our models will be very accurate

## Transformer Architecture

### Basic Transformer Architecture
- We are doing a seq2seq task for ex: Language Transaltion lets suppose from Eng -> French
![alt text](<resources/Screenshot 2026-04-25 203011.png>)
- So input is English and output is French
- Now we go into the transformer block above
  
![alt text](<resources/Screenshot 2026-04-25 203203.png>)

- Transformer also follows an Encoder Decoder architecture
- Inside the encode we can have multiple encoders step by step, similarly for decoder as well
- In the research paper. **"ATTENTION IS ALL YOU NEED"**, they have used 6 encoders and decoders
  ![alt text](<resources/Screenshot 2026-04-25 210248.png>)
- Inside single encoder we can see that there are 2 layers
  - Feed Forward Neural Network
  - Self Attention layer
- Inside a Single decoder we can see that we have 3 layers
  - Feed Forward Neural Network
  - Encoder-Decoder Attention
  - Self Attention layer

- Now Lets Dig into the Encoder further
![alt text](<resources/Screenshot 2026-04-25 210909.png>)
-  Once the inputs are converted into vector using some embedding layer like word2vec
-  These vectors are passed to the self attention layer.
-  Self attention layer converts these vectors inot Contextual vectors.
-  Which are then passed to the Feed Forward layer
-  Output from the Feed Forward neural network is passed to the next Encoder
-  In which the same process repeats
-  

#### Self Attention at a higher level
Self-attention, also known as scaled dot-product attention, is a crucial mechanism in the transformer architecture that allows the model to weigh the importance of different tokens in the input sequence relative to each other

![alt text](<resources/www.youtube.com_watch_v=3bPhDUSAUYI&t=20s (1).png>)

- We need to derive 3 important vectors
  - **Queries Vector (Q)**
    - Query vector represent the token for which we are calculating the attention. They help determine the importance of other tokesn in the context of the current token
    - Importance:
      - **Focus Determination**:
        - Queries help the model decide which parts of the sequence to focus on for each specific token. By calculating the dot product between a query vector and all key vectors, the model assesses how much attention to give to each token relative to the current token
      - **Contextual Understanding**:
        - Queries contribute to understanding the relationship between the current token and the rest of the sequence, which is essential for capturing dependencies and context
        - 
  - **Keys Vectors (K)**
    - Role: Key Vectors represent all the tokens in the sequence and are used to compare with query vectors to calculate attention scores.
    - Importance:
      - **Relevance Measurement**:
        - Keys are compared with queries to measure the relevance or compatibility of each token with current token. This comparison helps in determining how much attention each token should receive.
      - **Information Retrieval**
        - Keys play a critical role in retrieving the most relevant information from the sequence by providing a basis for the attention mechanism to compute similarity scores.
  - **Values Vectors (V)**
    - Role: Value vectors hold the actual information that will be aggregated to form the output of the attention mechanism.
    - Importance:
      - **Information Aggregation**:
        - Values contain data that will be weighted by the attention scores. The weighted sum of values forms the output of the self attention mechanism, which is then passed on to the next layers in the network.
      - **Context Preservation**:
        - By weighting the values according to the attention scores, the model preserves and aggregates relevant context from the entire sequence, which is crucial for tasks like translation, summarization, and more.