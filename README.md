# Document Summarization NLP

```Prerequistes
* torch
* pickle
* gzip
```

```Data
Input and validation data can be downloaded from https://github.com/harvardnlp/sent-summary 
```

```Steps
* Change the file paths in the params class
* Run the summarizer.py to train the model 
```
```js
wrap(str, {cut: true});
```

![network](https://user-images.githubusercontent.com/19767662/76513516-640f9600-6491-11ea-873b-ebd5ba2dc47a.png)
```Pointer Generator Network explained
Here is my attempt at understanding the network:
* The network has mostly copying capability with some generation capacity
* A history of encoder attention weights is maintained , to discourage new attention calculation to <br/>
give much weight to already covered tokens, thus avoiding repetition
* Decoder attention history is maintained to calculate decoder context(weghts to already generated words) <br/>
* It decided at each token generation, how much weightage should be given to copying and how much to generation from <br/> vocabulary.
* This is done by learning pointer probability, which takes into account encoder-decoder context, decoder context and <br/> decoder hidden state at a particular generation step.
* The network also uses a coverage loss, which is minimum of the coverage vector and encoder-decoder attention, to <br/> punish the network for paying any attention to tokens where attention is already paid in previous steps.
* The network also uses an extra vocabulary, which is existing vocab + words that appear in the training\test set. <br/> These new words are not added to existing vocan, and are only used for copying, for input document. Thus it avoids having a huge vocabulary.
```

## Any questions 👨‍💻
<p> If you have any questions, feel free to ask me: </p>
<p>📧: nirmalendu@outlook.com<p>
