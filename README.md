# Abstractive Text Summarization using ProphetNet with LoRA Fine-Tuning

## Introduction
This project fine-tunes the ProphetNet model using the XSum dataset for the task of abstractive text summarization. The goal is to generate one-sentence summaries that capture the core idea of a news article.

Key features of this project include:
- **Dataset**: BBC XSum - a large-scale, abstractive summarization dataset with diverse and information-dense summaries.
- **Model**: ProphetNet-Large (pretrained on CNN/DailyMail), adapted via LoRA for parameter-efficient fine-tuning.
- **Fine-Tuning**: Applied LoRA adapters (rank = 16) to both attention and feed-forward sublayers of ProphetNet-Large, then fine-tuned on the XSum training split leveraging mixed-precision (FP16), gradient accumulation to simulate large batches, and periodic checkpointing on Google Drive.
- **Evaluation**: Assessed using ROUGE scores to measure summary quality (ROUGE-1, ROUGE-2, ROUGE-L).

Final model performance:
- **Test Loss**: 4.5372 
- **ROUGE-1**: 0.3748 
- **ROUGE-2**: 0.1453 
- **ROUGE-L**: 0.2996

This approach is ideal for researchers and developers interested in experimenting with efficient fine-tuning strategies for sequence-to-sequence models under limited compute budgets.

## Features
- **Optimized Summarization Model**: Fine-tuned ProphetNet with LoRA achieving stable convergence (4.5 loss) after only 2000 steps, demonstrating excellent efficiency for text summarization tasks.
- **Memory-Efficient Implementation**: Employs gradient checkpointing and optimized hyperparameters, allowing for larger batch sizes and faster training while maintaining model quality.
- **Robust Learning Dynamics**: Training and validation curves show ideal parallel learning pattern with no overfitting, indicating a well-regularized model suitable for production deployment.
- **Extreme Summarization**: Focused on the XSum dataset, where each article is distilled into a single-sentence summary.
- **ProphetNet Fine-Tuning**: Leveraged Microsoft's prophetnet-large-uncased-cnndm model, originally trained on CNN/DailyMail, and adapted it to the XSum dataset.
- **Parameter-Efficient Training with LoRA**: Applied Low-Rank Adaptation (LoRA) to enable efficient fine-tuning with significantly fewer trainable parameters, saving memory and compute.
- **Efficient and Resumable Training**: Enabled mixed precision (FP16) for faster training and lower GPU memory usage, along with checkpointing to allow training to resume smoothly after interruptions.
- **Evaluation Metrics**: Evaluated model performance using ROUGE scores (ROUGE-1, ROUGE-2, and ROUGE-L).
- **Test Set Prediction**: Final evaluation on the held-out test set to measure generalization.

## Background to ProphetNet & LoRA

### ProphetNet: Sequence-to-Sequence Pretrained Model
[ProphetNet]([https://arxiv.org/abs/2001.04063](https://arxiv.org/abs/2001.04063)) is a pre-trained encoder-decoder model developed by Microsoft, specifically designed for abstractive text generation tasks such as summarization and translation.

Key Characteristics:
- **N-gram Prediction Objective**: Unlike traditional language models that predict only the next word, ProphetNet predicts multiple future tokens (n-gram prediction) during training. This encourages the model to learn better planning and coherence in its generation.
- **Transformer-based Architecture**: ProphetNet follows a typical Transformer layout but modifies the decoder to handle multi-token future prediction.
- **Strong Summarization Performance**: Achieved state-of-the-art ROUGE scores on multiple summarization benchmarks at the time of release.

### LoRA: Low-Rank Adaptation for Efficient Fine-Tuning
[LoRA (Low-Rank Adaptation of Large Language Models)]([https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)) is a parameter-efficient fine-tuning method designed to adapt large pre-trained models with fewer trainable parameters.

Core Concepts:
- **Freezes Pretrained Weights**: Rather than updating all model parameters, LoRA injects trainable low-rank matrices into attention and feedforward layers.
- **Low-Rank Decomposition**: By approximating weight updates with low-rank matrices, LoRA significantly reduces memory and compute requirements.
- **Task-Specific Adaptation**: LoRA enables quick and effective fine-tuning for specific tasks (like summarization) without the overhead of full model updates.

### Why LoRA in This Project?
- **Reduced training costs and memory usage** on Colab GPUs.
- **Enabled fine-tuning a large model** like ProphetNet efficiently.
- **Allowed training with fewer trainable parameters** while retaining strong performance.

## Fine-Tuning Pipeline

1. **Data Preparation**
   - Load the XSum dataset via Hugging Face Datasets.
   - Apply preprocess_function to tokenize articles (max length = 512) and summaries (max length = 64), producing input_ids, attention_mask, and labels.

2. **Model & Adapter Configuration**
   - Load microsoft/prophetnet-large-uncased-cnndm.
   - Configure LoRA (rank = 16) on attention projections and feed-forward sublayers.
   - Wrap with get_peft_model() to inject LoRA adapters.

3. **TrainingArguments Setup**
   - Persist output_dir on Google Drive.
   - Use batch size = 8, gradient accumulation = 8 (effective 64), FP16 mixed precision.
   - Evaluate & save every 1,000 steps (eval_strategy="steps", save_strategy="steps", save_steps=1000, save_total_limit=2).
   - Learning rate = 4e-5, epochs = 3, warmup = 500, weight decay = 0.01.
   - load_best_model_at_end=True, metric_for_best_model="loss", greater_is_better=False.

4. **Trainer Initialization**
   - Instantiate Trainer with model, args, tokenized data, DataCollatorForSeq2Seq, and tokenizer.

5. **Resumable Training & Checkpointing**
   - Checkpoints are automatically saved every 1,000 steps to Drive (keeping only the two most recent).
   - Use trainer.train(resume_from_checkpoint=True) to start fresh or resume from the latest checkpoint.
   - At training end, the best checkpoint (lowest validation loss) is reloaded (load_best_model_at_end=True).

## Evaluation & Results

After completing fine-tuning, the model was evaluated on the XSum test split to assess its summarization performance. The evaluation was performed using the standard ROUGE metrics, a common choice for comparing generated summaries against reference summaries, along with the final test loss.

### Final Evaluation Metrics:
- **Test Loss**: 4.5372
  - Indicates the average negative log-likelihood of the model on the test set. A lower value suggests the model has a better fit to the data, and this value is consistent with the observed convergence during training.
- **ROUGE-1 (unigram overlap)**: 0.3748
  - Measures how many unigrams (individual words) in the generated summary match those in the reference summary. A score of ~0.37 indicates the model captures a fair amount of the key content.
- **ROUGE-2 (bigram overlap)**: 0.1453
  - Measures the overlap of consecutive word pairs. This is a more stringent metric, and the score reflects moderate success in capturing coherent and contextually relevant phrases.
- **ROUGE-L (longest common subsequence)**: 0.2996
  - Captures sentence-level structure by identifying the longest sequences of matching words. A ~0.30 score indicates that the model-generated summaries maintain decent structural alignment with the reference.

### Interpretation:
These results suggest that the fine-tuned model has learned to generate relevant, concise summaries with reasonable lexical and structural similarity to the human-written references. While not state-of-the-art, the performance is strong for a LoRA-adapted model trained with limited resources and time and represents a solid foundation for further refinement.

### Training & Validation Loss Analysis

![Training and Validation Loss](https://github.com/Neel-Raibole/Extreme-Text-Summarization-using-Prophetnet-and-LoRA/blob/main/prophetnet_tb_logs/Training%20Loss%20and%20Validation%20Loss.png)

The training and validation loss curves reveal key characteristics of the model's learning process:
- **Stabilization**: By around step 2000, both curves flatten, indicating convergence.
- **Validation Loss Consistently Lower**: The validation loss remains slightly below the training loss throughout, pointing to effective regularization.
- **No Signs of Overfitting**: The losses remain closely aligned with no divergence, suggesting the model is not memorizing the training data.
- **Performance Plateau**: After step 2000, both losses remain stable, showing limited further improvement.

These trends indicate that the model has undergone a healthy and efficient fine-tuning process.

## Sample Summaries

Below are 3 random examples from the test set. Each entry shows the original article, my model's generated one-sentence summary, and the reference summary.

### ARTICLE #409
Donald Trump, Jeb Bush and Scott Walker will take the stage in Cleveland on Thursday night with seven rivals.
Fox News selected the 10 most popular Republicans based on five national polls, excluding Mr Perry and South Carolina Senator Lindsey Graham.
Those two and five other candidates will take part in an earlier debate.
Former Pennsylvania Senator Rick Santorum reacted angrily to his omission.
"The idea that they have left out the runner-up for the 2012 nomination [Santorum], the former four-term governor of Texas [Perry], the governor of Louisiana [Bobby Jindal], the first female Fortune 50 CEO [Carly Fiorina], and the 3-term Senator from South Carolina [Graham] due to polling seven months before a single vote is cast is preposterous," his spokesman said.
In contrast, Mr Perry tweeted that he was looking forward to being on Fox at 5pm for "a serious exchange of ideas and positive solutions to get America back on track".
The main debate takes place four hours later at 9pm local time (01:00 GMT).
All eyes will be on hotel tycoon Mr Trump, who leads the polls and has made headlines with outspoken remarks about many of his rivals.
A fun guide to the 10 Republican debaters
One of the Republican frontrunners, Jeb Bush, became embroiled in a row with leading Democratic candidate Hillary Clinton over women's health funding on Tuesday.
The former Florida governor was attacked by Mrs Clinton after he told a conservative Christian audience he wasn't sure "we need half a billion dollars for women's health issues".
But he later said he "misspoke" after criticism of his remarks.
Meet all of the 2016 hopefuls

#### GENERATED SUMMARY
fox news selected the 10 most popular republicans based on five national polls. [X_SEP] rick santorum, bobby jindal, carly fiorina, rick perry and lindsey graham will take part in an earlier debate. [X_SEP] all eyes will be on hotel tycoon donald trump, who leads the polls.

#### REFERENCE SUMMARY
Rick Perry, the former governor of Texas, is not among the 10 Republicans running for president who will take part in the first primetime TV debate.

### ARTICLE #3657
Lance Naik (Corporal) Hanamanthappa Koppad was tapped under 8m of snow at a height of nearly 6,000m along with nine other soldiers who all died. Their bodies have now been recovered.
The critically ill soldier has been airlifted to a hospital in Delhi.
"We hope the miracle continues. Pray with us," an army statement said.
The army added that "he has been placed on a ventilator to protect his airway and lungs in view of his comatose state".
"He is expected to have a stormy course in the next 24 to 48 hours due to the complications caused by re-warming and establishment of blood flow to the cold parts of the body," the army said.
The avalanche hit a military post on the northern side of the glacier.
Senior military officials said at the time there was little chance of finding any of the soldiers alive after the incident last Wednesday.
Siachen is patrolled by troops from both India and Pakistan, who dispute the region's sovereignty.
It is known as the world's highest battlefield. Four Indian soldiers were killed by an avalanche in the same area last month.
The soldiers were on duty at an army post on the glacier at an altitude of 5,900m (19,350ft) when the avalanche struck.
Specialist army and air force teams immediately began searching for the missing soldiers close to the Line of Control that divides Indian and Pakistani-administered Kashmir.
The chances of any soldiers being found alive were so slim that Prime Minister Narendra Modi even offered condolences in a message on Twitter last week.
Avalanches and landslides are commonplace in the area during winter where temperatures can drop to -60C.
More soldiers have died from harsh weather on the glacier than in combat since India seized control of it in 1984. Soldiers have been deployed at heights of up to 6,700m (22,000ft) above sea level.
The neighbours have failed to demilitarise the Siachen glacier despite several rounds of peace talks.

#### GENERATED SUMMARY
lance naik ( corporal ) hanamanthappa koppad was tapped under 8m of snow at a height of nearly 6, 000m along with nine other soldiers who all died. [X_SEP] he has been placed on a ventilator to protect his airway and lungs in view of his comatose state.

#### REFERENCE SUMMARY
An Indian soldier who was buried in an avalanche that struck the Siachen glacier in Indian-administered Kashmir six days ago has been found alive.

### ARTICLE #6912
ZTE Corp obtained and illegally shipped US-made equipment to Iran in violation of US sanctions, the Justice Department said.
It also sent goods to North Korea without the correct export licences.
The US said ZTE lied to authorities and its own lawyer about the violations.
It must now pay a $892m (£740m) penalty as well as $300m which will be suspended for seven years depending on the firm meeting certain conditions.
ZTE says it acknowledges it has made mistakes, and is working towards improving its procedures.
The US said that the highest levels of management at ZTE approved the scheme which involved the shipment of $32m worth of US-made goods to Iran between 2010 and 2016.
The equipment included routers, microprocessors and servers controlled under export regulations for "security, encryption... and/or anti-terrorism reasons".
ZTE also made 283 shipments of mobile phones to North Korea despite knowing this contravened rules around exports to the country.
According to Reuters, ZTE buys around a third of its its components from US businesses such as Qualcomm, Microsoft and Intel. It also sells phone handsets to major carriers such as T-Mobile and AT&T.
US Attorney General Jeff Sessions said: "ZTE Corporation not only violated export controls that keep sensitive American technology out of the hands of hostile regimes like Iran's - they lied to federal investigators and even deceived their own counsel and internal investigators about their illegal acts."
He added: "This plea agreement holds them accountable, and makes clear that our government will use every tool we have to punish companies who would violate our laws, obstruct justice and jeopardise our national security."
The company reached the agreement with the US Justice, Commerce and Treasury departments.
As part of the deal, it must submit to a three-year period of probation, during which time it will be independently monitored to ensure it remains compliant.
Chairman and chief executive of ZTE, Zhao Xianming, said: "ZTE acknowledges the mistakes it made, takes responsibility for them and remains committed to positive change in the company."
"Instituting new compliance-focused procedures and making significant personnel changes has been a top priority for the company."

#### GENERATED SUMMARY
zte corp illegally shipped us - made equipment to iran in violation of us sanctions. [X_SEP] also sent goods to north korea without the correct export licences. [X_SEP] must now pay a $ 892m penalty as well as $ 300m which will be suspended for seven years depending on the firm meeting certain conditions.

#### REFERENCE SUMMARY
Chinese telecom giant ZTE has been fined $1.1bn and will plead guilty to charges that it violated US rules by shipping US-made equipment to Iran and North Korea.
