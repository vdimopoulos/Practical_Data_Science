# A3 LLM Text Detection
## Vasileios Dimopoulos

## A. Data augmentation
### Prompt an LLM to generate essays, so that you balance the data (use both prompts provided by the challenge).
For the first part of the assignment we used the prompts given by the competition, to generate new training essays for our data. The student essays of the given training set were around 1400 and made by 2 different prompts, using a collection of texts as sources for the students. Firstly, for the essay generation, I used the OpenAI API and GPT-3.5-turbo to generate 1400 essays, 700 for each given prompt (cars and election essays). We didn't use the exact prompts that were given to the students, but changed them using 2 approaches. Firstly, we made ChatGPT to summarize all the source text and we used the summary as input, together with the instructions. The prompts were:

Prompt0: The provided excerpts discuss various aspects of the changing relationship between people and cars in different parts of the world. In Vauban, Germany, an experimental suburb promotes a car-free lifestyle, with limited parking spaces and an emphasis on public transportation and walkability. Similar initiatives are emerging globally as part of the "smart planning" movement to reduce suburban dependence on automobiles and combat greenhouse gas emissions.Paris faced a driving ban in response to severe smog, highlighting the environmental consequences of heavy car usage. Meanwhile, Bogota, Colombia, implemented a successful annual car-free day to promote alternative transportation, reduce smog, and improve urban living conditions.The excerpts also touch upon shifting trends in the United States, where there is a potential decline in car ownership and driving. Efforts are being made to create "car-reduced" communities, and President Obama's goals aim to curb greenhouse gas emissions with support from changing American behaviors related to car usage. The idea of "peak driving" is explored, with studies suggesting that Americans are driving fewer miles, buying fewer cars, and obtaining fewer licenses. This shift is seen as a cultural change, influenced by factors such as telecommuting, the rise of shared transportation services, and changing urban preferences. While this trend could benefit the environment, it poses challenges for the traditional car industry.Overall, these excerpts highlight a global movement towards reducing car dependency, with various approaches and initiatives aimed at creating more sustainable and environmentally friendly urban living environments.Write an explanatory essay to inform fellow citizens about the advantages of limiting car usage. Your essay must be based on ideas and information that can be found in the passage set. Manage your time carefully so that you can read the passages; plan your response; write your response; and revise and edit your response. Be sure to use evidence from multiple sources; and avoid overly relying on one source. Your response should be in the form of a multiparagraph essay. Write up to 600 words. 

Prompt1: The Electoral College is a constitutional process established by the founding fathers as a compromise between congressional and popular vote methods. It involves the selection of electors, their voting for the President and Vice President, and the subsequent counting of electoral votes by Congress. With 538 electors, a majority of 270 electoral votes is required to elect the President. Each state's allotment of electors equals its Congressional delegation's size.Candidates running for President have their own group of electors, generally chosen by their political party. The election occurs every four years on the Tuesday after the first Monday in November. Most states use a "winner-take-all" system, where all electors go to the winning candidate, but Maine and Nebraska employ proportional representation.After the election, each state's governor prepares a "Certificate of Ascertainment" listing candidates and their electors. This certificate is sent to Congress and the National Archives as part of the official records of the presidential election.Despite its existence, the Electoral College faces criticism. Critics argue that it is unfair, outdated, and irrational. They highlight flaws in the system, such as the potential for electors to defy the popular vote, state legislatures influencing elector selection, and the possibility of an electoral vote tie leading to House intervention.On the other hand, defenders of the Electoral College, such as Judge Richard A. Posner, emphasize practical reasons for its retention. These include the certainty of outcome, the requirement for a presidential candidate to have trans-regional appeal, focus on swing states, balancing the influence of big states, and avoiding runoff elections. In essence, the debate over the Electoral College revolves around its perceived flaws and practical benefits, with critics advocating for its abolition and defenders emphasizing its role in maintaining stability and regional balance in presidential elections.Write a letter to your state senator in which you argue in favor of keeping the Electoral College or changing to election by popular vote for the president of the United States. Use the information from the texts in your essay. Manage your time carefully so that you can read the passages; plan your response; write your response; and revise and edit your response. Be sure to include a claim; address counterclaims; use evidence from multiple sources; and avoid overly relying on one source. Your response should be in the form of a multiparagraph essay. Write up to 600 words.

The resulting essays from this prompt (train_df) had 1 significant problem. The LLM plagiarised the summary text and created almost identical essays, copying the source summary. The texts generated from this approach weren't used, even though they exist on augmentation.csv.
The next approach was to only give the instructions of the prompts, so:

Prompt0 : "Write an explanatory essay to inform fellow citizens about the advantages of limiting car usage. Manage your time carefully so that you can read the passages; plan your response; write your response; and revise and edit your response. Be sure to use evidence from multiple sources; and avoid overly relying on one source. Your response should be in the form of a multiparagraph essay.Write up to 600 words."

Prompt1: "Write a letter to your state senator in which you argue in favor of keeping the Electoral College or changing to election by popular vote for the president of the United States. Manage your time carefully so that you can read the passages; plan your response; write your response; and revise and edit your response. Be sure to include a claim; address counterclaims; use evidence from multiple sources; and avoid overly relying on one source. Your response should be in the form of a multiparagraph essay. Write your response in the space provided.Write up to 600 words."

These essays were used in the following analysis. Both approaches' essays were saved in augmentation.csv with labels 0 and 1 according to their topic.

### Build text classifiers on the augmented data, using cross validation with appropriate classification evaluation metrics to assess them, and suggest the best performing one.

We used 5 different classifiers on our data: Naive Bayes, Logistic Regression, Linear SVM, Non-Linear SVM (rbf) and k-NN. We performed 10-fold cross validation after spliting our data on train and test sets. Inside the cross validation the training set was splitted again to test and train and these texts were vectorized using Tfidf Vectorizer. The mean F1, accuracy scores and the std of the accuracy for the folds were printed for comparisson. All the classifiers performed extremely well (over 99%). We can attribute that to the big difference our generated texts had in comparisson with the student texts, something that made the classes easily separable and the accuracy very high. The best performing classifier was the Linear-SVM classifier (maybe implying that our data were linearly separable).

### Compute two scores per generated text, one reflecting the maximum and the other the average similarity of that text with student essays.
### Study the correlation between the similarity scores and the prediction probability of your best classifier for the generated texts; compute the prediction probability per text, by training the selected classifier on all except from that text, which is used a test instance (a.k.a. the leave-one-out cross validation setting). 

To compute the similarity per text we calculated the dot product of the Tfidf representation of the texts, for each generated text, in comparisson with all the student texts. After getting all the scores for each text, on a column on our training set we kept the average similarity score and the max similarity score of each generated text. What we observed is that on the average the similarities, as we said before, were low, and the generated texts were quite different to the student texts.
To get the predictability for each generated text, we trained the Linear SVM on all our data and got the prediction probability for each generated text, keeping it on a column named "Predictability" on our training set. What we observed, was that that the predictability was almost 1 for all the essays, as we expected from the high accuracy of the model and the factors we explained before. 

### Study the correlation between the similarity scores and the prediction probability of your best classifier for the generated texts; compute the prediction probability per text, by training the selected classifier on all except from that text, which is used a test instance (a.k.a. the leave-one-out cross validation setting). Based on your study so far, decide which generated texts should be discarded in order to improve the benchmark and yield a more robust classifier.

Because of our unsimilar generated and student texts, there was no correlation between similarity and predictability (pearson r = 0.01, spearman = -0.12, kendall = -0.09).
Results were saved to train_df_after_A.csv.

## B. Learning curves
### Keep a test set apart and split the train data to portions (10%, …, 90%, 100%). Train your best performing algorithm on each portion. Assess each trained instance on the test (the same across portions) and on the training data.
### Visualise the two curves (train, test), based on an appropriate evaluation measure, diagnosing weak and strong points of your classifier (a.k.a. the learning curves).
### Add a regressor to the plot, to estimate how many more texts should you generate to reach the "best" performance.

The above tasks were performed and the lines were plotted. Due to the unsimilar texts and the very high performance of the SVM Linear model, the training and validation Mean Squared Error (evaluation metric we used for the curves) was close to 0 and slightly increased when the split between train and validation became 40-60.

## C. Clustering-based augmentation
### Use K-Means, based on an approprate text representation and the (estimated) optimum K, to cluster the generated essays, and then the student essays.
### Compare the cluster balance (number of instances per cluster) between the two clusterings.
### Yield a title per cluster, reflecting the topic of the texts included.
We used Word2Vec embedding to represent our data and then we used the k means algoritm to cluster the data. For the hyperparameter k, we used silhouette scores to find the best number of clusters. For both our sets, generated and student essays, the value of k was 2. For the student essays, cluster 0 and 1 had the bellow top 20 most frequent words and title:

Cluster 0: "Presidential Campaigns and Strategies"
Top 20: stated, fewer, idea, presidency, walk, says, example, americans, presidential, alternative, means, 12, different, smaller, benefit, happen, chance, posner, healthier, said

Cluster 1: "Diverse Opinions on Presidential Driving Factors"
Top 20: presidential, united, duffer, government, smaller, driving, 4, colombia, popular, away, free, does, reasons, congress, reason, didnt, little, party, walter, bogota

The clusters were balanced with cluster 1 having 706 observations and cluster 0 666.

For the generated essays, cluster 0 and 1 had the bellow top 20 most frequent words and title:

Cluster 0: "Critical Infrastructure Protection"
Top 20: heavily, numerous, popular, safeguard, spaces, infrastructure, thank, matter, imperative, critics, livable, intro, voices, ensure, ultimately, inclusive, activity, representative, outcomes, transit

Cluster 1: "Air Quality and Environmental Perspectives"
Top 20: dominance, electing, including, promoting, air, large, perspectives, component, counterclaim, better, additionally, citizens, emissions, stability, pollution, positive, reducing, carbon, furthermore, implications

The clusters were slightly not balanced with cluster 1 having 716 observations and cluster 0 546.

### Generate more texts (as in A) in order to better balance your clusters.

Again we used OpenAI API and GPRT-3.5-turbo to fetch essays. to balance the student essays cluster, we got 40 essays with the prompt:  
'Write an essay, up to 600 words with topic: "Diverse Opinions on Presidential Driving Factors". Similar essays had as top 20 words: smaller, presidential, duffer, united, driving, does, away, popular, count, 4, reasons, congress, free, walter, government, didnt, bogota, little, process, number'
To balance the generated essays cluster, we got 170 essays with the prompt:  
'Write an essay, up to 600 words with topic:"Critical Infrastructure Protection". Similar essays had as top 20 words:safeguard, popular, numerous, infrastructure, voices, heavily, matter, spaces, critics, thank, outcomes, intro, activity, imperative, economic, transit, promote, representative, inclusive, prevents'
We appended this data to the training set, with the prompt ids 3 and 4.

### Re-train your best-performant classifier on the new data (or a careful selection of them) and analyze the benefits of using clustering to improve the classifier.
We used the whole new set to train the SVM Linear Classifier. As we mentioned before the classifier was already performing to almost perfection, so the accuracies and f1 scores remained the same, even with the addition of new texts.
Generally, integrating clustering into a classifier provides notable advantages. Clustering enhances feature representation by identifying patterns and relationships, aiding in noise reduction. It simplifies data preprocessing, focusing on homogeneous clusters to improve interpretability. Addressing class imbalances, clustering identifies minority clusters for targeted training. The approach also adapts to evolving data patterns, ensuring the classifier's flexibility over time. Additionally, clustering aids in dimensionality reduction, accelerating training and promoting computational efficiency for more effective models.
Results were saved to train_df_after_C.csv.
