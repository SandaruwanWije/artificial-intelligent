# artificial-intelligent

Companion code for the AI articles on [byteblueprints.com](https://byteblueprints.com).
Each folder is one project, built in order, with the article's own numbers.
See [CONVENTIONS.md](CONVENTIONS.md) for how every project is built.

```bash
uv sync
uv run python -m ai_01_perceptron   # run one project
uv run pytest                       # check every project against its article
```

| Project | Article |
|---------|---------|
| [ai_01_perceptron](ai_01_perceptron) | [The perceptron](https://byteblueprints.com/artificial-intelligence/ann/04_perceptron/) |
| [ai_02_activations](ai_02_activations) | [Activation functions](https://byteblueprints.com/artificial-intelligence/ann/05_activations/) |
| [ai_03_layers](ai_03_layers) | [Layers and depth](https://byteblueprints.com/artificial-intelligence/ann/06_layers_depth/) |
| [ai_04_forward_pass](ai_04_forward_pass) | [Forward pass](https://byteblueprints.com/artificial-intelligence/ann/07_forward_pass/) |
| [ai_05_loss](ai_05_loss) | [Loss functions](https://byteblueprints.com/artificial-intelligence/ann/08_loss/) |
| [ai_06_gradient_descent](ai_06_gradient_descent) | [Gradient descent](https://byteblueprints.com/artificial-intelligence/ann/09_gradient_descent/) |
| [ai_07_backpropagation](ai_07_backpropagation) | Backpropagation, [part 1](https://byteblueprints.com/artificial-intelligence/ann/10_backprop/) and [part 2](https://byteblueprints.com/artificial-intelligence/ann/10_backprop_2/) |

## Roadmap



LEVEL 1 – From Scratch ML (No Libraries)



Goal: Understand how ML actually works



Linear Regression From Scratch



Gradient descent



MSE loss



Visualization of learning



Logistic Regression From Scratch



Sigmoid



Binary classification



Decision boundary



k-Nearest Neighbors From Scratch



Distance metrics



Classification logic



LEVEL 2 – First “Real” ML Using Libraries



Goal: Practical ML workflow



Linear Regression Using sklearn



Classification Pipeline



Train / validation



Confusion matrix



Feature Engineering Project



Scaling



Encoding



Feature selection



End-to-End ML Project



Dataset → Model → Evaluation



LEVEL 3 – Neural Networks (From Zero to Framework)



Goal: Demystify deep learning



Single Neuron From Scratch



Multi-Layer Neural Network From Scratch



Backpropagation Visualizer



Neural Network Using PyTorch



Custom Dataset



Training loop



Overfitting \& Regularization Project



LEVEL 4 – Computer Vision Path



Goal: Visual intelligence



Image Classifier (CNN)



Digit Recognition (MNIST)



Image Augmentation Pipeline



Transfer Learning Project



Real-Time Webcam Object Detection



LEVEL 5 – NLP Before LLMs



Goal: Language understanding basics



Text Cleaning Pipeline



Bag-of-Words Classifier



TF-IDF Search Engine



Sentiment Analysis Model



Sequence Model (RNN / LSTM)



LEVEL 6 – Transformers (Pre-LLM)



Goal: Understand modern NLP



Attention Mechanism From Scratch



Transformer Encoder From Scratch



Mini-Transformer Language Model



LEVEL 7 – First LLM Interaction



Goal: Using LLMs correctly



LLM API Client (OpenAI / Local)



Prompt Engineering Playground



Tokenizer Explorer



Chatbot With Memory



Cost-Aware LLM Calls



LEVEL 8 – LLM Systems Engineering



Goal: Real-world AI systems



Retrieval-Augmented Generation (RAG)



Vector Database From Scratch



Semantic Search Engine



PDF / Knowledge-Base Chatbot



Multi-Document QA System



LEVEL 9 – AI Agents



Goal: Autonomous reasoning systems



Tool-Calling Agent



Planner-Executor Agent



Multi-Agent Collaboration



Code-Writing Agent



Debugging Agent



LEVEL 10 – Production-Grade AI



Goal: Industry-level skills



LLM API Gateway



Rate Limiting \& Caching



Model Evaluation Harness



Long-Term Memory System



Fine-Tuning Pipeline



LEVEL 11 – Advanced / Research-Grade



Goal: Stand out



Train a Small LLM From Scratch



Quantization \& Inference Optimization



Self-Reflecting Agent



AI Dev Team Simulation



Autonomous Trading / Decision Agent





LEVEL 1 – Core Machine Learning (From Scratch)



Linear Regression From Scratch



Logistic Regression From Scratch



k-Nearest Neighbors From Scratch



Naive Bayes Classifier From Scratch



Decision Tree From Scratch



Gradient Descent Visualizer



ML Model Evaluation Toolkit



LEVEL 2 – Applied ML With Libraries



Linear Regression Using sklearn



Classification Pipeline Project



Feature Engineering Pipeline



Regression vs Classification Comparison



End-to-End ML Project



ML Experiment Tracking System



LEVEL 3 – Neural Networks



Single Neuron Implementation



Multi-Layer Neural Network From Scratch



Backpropagation Engine



Neural Network Using PyTorch



Custom Training Loop Project



Overfitting Detection \& Regularization



Hyperparameter Tuning System



LEVEL 4 – Computer Vision



Image Classifier (CNN)



Handwritten Digit Recognition



Image Augmentation Pipeline



Transfer Learning Image Classifier



Real-Time Webcam Object Detection



Face Detection System



Image Similarity Search



LEVEL 5 – NLP Fundamentals (Pre-LLM)



Text Preprocessing Pipeline



Bag-of-Words Text Classifier



TF-IDF Search Engine



Sentiment Analysis System



Spam Detection Model



N-gram Language Model



RNN / LSTM Text Generator



LEVEL 6 – Transformers (Before LLMs)



Attention Mechanism From Scratch



Transformer Encoder Implementation



Mini Transformer Language Model



Text Classification With Transformers



Sequence-to-Sequence Model



LEVEL 7 – LLM Basics



LLM API Client



Prompt Engineering Playground



Tokenizer \& Context Window Explorer



Chatbot With Short-Term Memory



Streaming Chat Interface



Cost-Aware LLM Usage Monitor



LEVEL 8 – LLM Systems (RAG \& Search)



Vector Database From Scratch



Embedding-Based Semantic Search



Retrieval-Augmented Generation System



PDF / Knowledge-Base Chatbot



Multi-Document Question Answering



Hybrid Search Engine



LEVEL 9 – Agents \& Tool Use



Tool-Calling LLM Agent



Planner–Executor Agent



Multi-Agent Collaboration System



Autonomous Research Agent



Code Generation Agent



Debugging \& Refactoring Agent



LEVEL 10 – Production AI Systems



LLM API Gateway



Authentication \& Rate Limiting



Caching \& Cost Optimization



LLM Evaluation \& Benchmarking



Long-Term Memory System



Fine-Tuning Pipeline



LEVEL 11 – Advanced / Capstone



Train a Small Language Model



Quantized Inference Engine



Self-Reflecting LLM Agent



Autonomous Decision-Making System



AI Developer Team Simulation



End-to-End AI Product



artificial\_intelligent/

│

├─ 01\_linear\_regression/

├─ 02\_logistic\_regression/

├─ 03\_knn\_classifier/

├─ 04\_naive\_bayes/

├─ 05\_decision\_tree/

├─ 06\_gradient\_descent\_visualizer/

├─ 07\_ml\_model\_evaluation/

│

├─ 08\_stock\_price\_prediction/

├─ 09\_customer\_churn\_prediction/

├─ 10\_sales\_forecasting/

├─ 11\_credit\_risk\_scoring/

├─ 12\_product\_demand\_prediction/

├─ 13\_ml\_pipeline\_auto\_retrain/

│

├─ 14\_handwritten\_digit\_recognition/

├─ 15\_handwritten\_letter\_recognition/

├─ 16\_fashion\_image\_classifier/

├─ 17\_sentiment\_analysis\_reviews/

├─ 18\_time\_series\_trend\_prediction/

├─ 19\_neural\_network\_vs\_ml\_comparison/

├─ 20\_hyperparameter\_tuning\_dashboard/

│

├─ 21\_fruit\_image\_classifier/

├─ 22\_face\_mask\_detection/

├─ 23\_vehicle\_detection\_cctv/

├─ 24\_face\_recognition\_system/

├─ 25\_ocr\_scanned\_documents/

├─ 26\_image\_similarity\_search/

├─ 27\_webcam\_object\_detection/

│

├─ 28\_news\_article\_classifier/

├─ 29\_resume\_keyword\_extractor/

├─ 30\_blog\_post\_search\_engine/

├─ 31\_product\_review\_sentiment/

├─ 32\_chat\_auto\_reply\_suggestion/

├─ 33\_language\_detection/

├─ 34\_text\_auto\_completion/

│

├─ 35\_text\_summarization/

├─ 36\_question\_answering\_system/

├─ 37\_faq\_chatbot/

├─ 38\_translation\_system/

├─ 39\_document\_classification\_engine/

│

├─ 40\_chatgpt\_terminal\_chatbot/

├─ 41\_prompt\_testing\_tool/

├─ 42\_token\_cost\_calculator/

├─ 43\_chatbot\_short\_term\_memory/

├─ 44\_chatbot\_streaming\_responses/

├─ 45\_ai\_assistant\_system\_rules/

│

├─ 46\_pdf\_chatbot/

├─ 47\_internal\_knowledge\_chatbot/

├─ 48\_semantic\_search\_research\_papers/

├─ 49\_multi\_document\_qa/

├─ 50\_legal\_policy\_chatbot/

├─ 51\_website\_content\_chat\_assistant/

│

├─ 52\_web\_research\_summary\_agent/

├─ 53\_planning\_task\_agent/

├─ 54\_ai\_research\_paper\_reader/

├─ 55\_code\_writing\_agent/

├─ 56\_bug\_fixing\_agent/

├─ 57\_multi\_agent\_system/

│

├─ 58\_ai\_api\_service/

├─ 59\_rate\_limiter/

├─ 60\_response\_caching\_system/

├─ 61\_output\_quality\_evaluator/

├─ 62\_long\_term\_memory\_assistant/

├─ 63\_finetuned\_domain\_chatbot/

│

├─ 64\_train\_small\_llm/

├─ 65\_llm\_low\_hardware\_inference/

├─ 66\_self\_correcting\_agent/

├─ 67\_autonomous\_trading\_agent/

├─ 68\_ai\_dev\_team\_simulation/

├─ 69\_full\_ai\_saas\_product/



Inside each project folder, can have:



data/           # datasets used

notebooks/      # Jupyter notebooks

src/            # source code

tests/          # unit / integration tests

README.md       # instructions, explanation, and results

