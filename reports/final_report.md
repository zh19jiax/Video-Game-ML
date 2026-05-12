![](media/image3.jpg){width="5.598958880139983in"
height="3.3620220909886265in"}

By Jiaxu Zhao, Humphrey Wang, Jiarui Yu, Xinni Cai

**1. Introduction**

**a. Dataset selection**

For this project, our team decided to build a machine learning model
that provides strategic recommendations to game developers for
establishing the market price of their new game based on the specific
characteristics that the game acquires such as the overall game length,
genre, minimum players, etc. Based on the first few conversations in the
group, we do believe that gaming is a high trend right now that bonds
everyone together and it would be interesting to help the publishers
analyze the current trends on the market, determining the root cause of
obtaining a high selling price and positive user response. At the start
of the project, we were really focusing on the time-series perspective
by trying to discover the trend of the game price over decades. However,
due to a large amount of missing values inside our original dataset, we
shifted towards the dataset with more current game details observations,
but no missing values.

**b. The Video Game Reviews Dataset**

The Video Game Reviews Dataset is a randomly generated collection of
dataset with a lot of flexibility and diversity for analysis. The
dataset provides information on 18 different video game-related
categories over 47,774 games across multiple gaming platforms, with
important features like *User Ratings*, *Price, Platform, Genre,* and
*Game Length*. The dataset includes 13 categorical variables and 5
numerical variables, making it a perfect blend-in for combinations of
classifiers and regressors models. This way, we can acquire optimized
results by exploring the dataset from multiple angles. However, several
columns exhibit a pattern of synthetically generated rather than sourced
from verified records. For example, some numerical columns display
uniform distribution such as the *User Rating, Game Length, and Price,*
indicating the factor of randomized data generation. One possible
explanation is that those data have been collected from different
platforms and normalized together into a single scale for better
modeling purposes. Besides, the columns about game quality such as
Graphic, Soundtrack and Story Quality have vague criteria to
differentiate the quality level. Therefore, those columns might provide
us with an ideal dataset for the later modeling process.

**2. Preliminary Analysis**

**a. Data Exploration and Correlation Analysis of Numeric Variables**

To understand the structure and characteristics of our dataset clearly,
we conducted an exploratory analysis before the dataset preprocessing.
How the quantitative variables relate to others is the first
consideration for our research since our objective is to develop a
recommendation system to help game developers optimize profitability
while keeping high customer satisfaction. Based on this thinking, we
created a correlation matrix to see the relationship between numerical
variables, specifically *Price, User Rating, Game length, Release Year,*
and *Minimum Number of Players.*

There are some key insights we can extract from this correlation matrix.
First, there is a strong positive correlation between *Price* and *User
Rating* (r = 0.76), implying that high priced video games tend to have
higher player satisfaction rate. One assumption we can make based on
this observation is that videos with high prices tend to have better
game quality or experience to attract players. Second notable
correlation found between *Game Length* and *User Rating* (r = 0.63),
suggesting that relatively longer game time is one of the factors for
players to choose the video game. Usually, the Game Length represents
rich content that players can fully engage themselves in the game
setting. Based on those observations, game developers can prioritize the
game quality and length with high price when creating video games. In
contrast, *Release Year* and *Minimum Number of Players* have minimal
correlations with other variables, indicating the slight impact on
player's satisfaction rate.

Overall, the correlation analysis indicates *Price* and *Game Length*
have the greatest impact on user rating. These findings enhance our
understanding of the quantitative factors that may shape player
satisfaction and provide a direction for building later models.

![](media/image1.png){width="5.348958880139983in"
height="3.744270559930009in"}

Figure 1: Correlation matrix of numeric variables

**3. Data Preprocessing**

**a. Data Reduction**

As we finalize our dataset and verify that no variables have mostly
"null" values like the Critic Scores in our initial dataset, we proceed
to the preprocessing and data cleaning stage. After completing the ETL
process for extracting the raw data, transforming it into the python
environment and loading it in, we began preprocessing by dropping out
columns that were not relevant in our actual analysis and final delivery
of the model.

Based on the understanding of our goal for building an optimized
platform for predicting game prices, we decided to initially drop five
columns: *Game Title*, *Requires Special Device*, *User Review Text*,
*Developer*, and *Publisher*. The *Game Title* column functions as
textual identifier rather than analytical variables that give us
predictive information useful for modeling. The similarities between two
games' names may even cause noise that affect our model's performance.
The *Requires Special Device* column contains limited variation without
the information on additional hardware required for games playing, so we
choose to drop this column for simplicity and clarity. The *User Review
Text* consists of textual reviews that are highly varied for detailed
analysis based on its free-form of contents and lack of repetition. For
columns *Developer* and *Publisher*, prices and game quality really
fluctuates widely depending on the quality of each individual game, not
the company that beyond designing the game, and the columns often
contain many unique values with few repetition. As a result, those two
columns may lead to sparse dummy variables, risking potential
overfitting without proportional insight, ending up being dropped for
clarity and better accuracy.

After further analysis and careful considerations, we also decided to
remove the *release year* column away from the project for improving
analytical clarity, as we found out that the characteristics of a
pooled-cross sectional data didn't consistently appear across different
time periods. Based on the time series chart that we made in tableau,
the average price and user rating per game stays relatively constant
throughout years, which is clearly controversial as compared to the
initial idea that we think time plays a huge component in the actual
modeling and analysis. As a result, time is not being considered as a
determining factor in our model.

![](media/image4.png){width="6.057292213473316in"
height="2.8330785214348206in"}

Figure 2: Trends of average game price and user rating from 2010 to 2023

**b. Dummy encoding**

After removing certain columns from the raw data loaded into the ide
environment, we proceeded to scale some of the data and encode some
variables to make them available for next step modeling procedure. We
began by selecting all columns with boolean information, which includes
*Multiplayer* and *Game Mode*. After confirming their datatype, we used
the aggregate astype function to convert the True/False results into the
numerical representations of 0 and 1s, which enables regression
algorithms to interpret variables more effectively.

Next, we selected the columns that contain multiple unique values, such
as *Game Mode*, *Multiplayer*, *Platform*, and *Genre*, and converted
them into the dummy variables using the get_dummies build-in function in
pandas to ensure there are no ordinal relationships. Similarly, we also
wanted to address the quality columns, specifically *Graphics Quality*,
*Soundtrack Quality*, *Story Quality,* by converting them from actually
text scaling into numerical representation. We first tried expanding the
usage of the sklearn package by applying the label encoder and fit
transform to convert the text into the integer, but realized that it
will also sort based on the alphabetical order which violates the
overall meaning as ranking "average" will be encoded in numbers before
"poor". Eventually, we decided to manually create the dictionary that
stores the actual corresponded value based on the scaling input. And by
mapping the quality with the actual values, we should be able to exhibit
a better analysis after the transformation.

**4. Model Exploration and Selection**

**a. Preliminary approach**

-   **Baseline Model**

> We wanted to try out different approaches that we learned so far on
> our dataset, and find the best model for our research purposes. To
> create a benchmark for our project, we built a baseline model using
> the average value of our target variable Price. We first calculated
> the average price from the training set, and then created a list of
> predictions where every prediction is just that average. We also made
> it the same length as y_test so that we can evaluate its $R^{2}$ and
> RMSE value. The result came out as $R^{2}$ = -2.308e-05, which is
> approximately 0; and RMSE = 11.502.

-   **K-NN**

> K-Nearest neighbor is the first actual algorithm that we wanted to try
> out. After evaluating K-Nearest Neighbors model with 80/20 train-test
> split, we found out that game length, user rating, multiplayer
> features and story quality are the key factors for identifying high
> value games. According to our model, the RMSE decreases as the value
> of k increases and it reaches the lowest point when k equals 49.
> However, the relatively low R squared is not competitive compared to
> the other models we tested, with $R^{2}$ = 9.452 and RMSE =0.329 .
> These outcomes suggest that K-NN did not perform well with our
> dataset, as the model's mechanism limited its ability to capture the
> full variability of game prices. Although the K-NN outperformed the
> baseline model and revealed some meaningful relationships between
> feature and price, its performance metrics were not strong enough for
> the analytical needs, so we ultimately decided not to use K-NN as our
> final model.

-   **Naive Bayes**

> Naive Bayes is a classification algorithm with a prioritized objective
> of estimating the conditional probability and determining the
> likelihood of a given new observation belonging to a specific class
> based on the current existing data. The Gaussian NB might be a better
> candidate in this scenario since we are predicting the specific price
> for a given game. However, based on the preliminary parameters in the
> Gaussian NB that we have to assume that the dataset is having a normal
> distribution, the assumption can be too strict and not well supported
> by the data, so eventually we will be using the Multicolinal NB. Since
> the Multicolinal NB requires all of the columns to be considered as
> categorical, we converted the existing numerical columns: User Rating,
> Game Length, Price and Min \# of Players into categorical variables by
> binning them. Although after the dummy encodings and data
> transformations, the results for the accuracy was 59.8%, which is
> still not high enough for the ultimate decision. Additionally, since
> Multicolinal NB is a classification algorithm, this style doesn't
> really match our target prediction variable, which is better fitted
> for regressor models.

-   **Neural Network**

> We also built a multilayer perceptron(MLP) neural network model which
> is designed to capture some non-linear relationships and complex
> interactions among variables, since we believe that Price would be
> influenced by multiple factors such as Age Group Targeted, Genre, Game
> Length and so on. To increase the power of prediction, we firstly
> normalized all the numerical variables and dummy encoded the
> multi-category variables. After constructing the model with 5 hidden
> layers, our MLP neural network model is able to predict continuous
> price values and fit the data exceptionally well. According to our
> results, the model had $R^{2}$ = 0.962 and a RMSE score of 2.248.
> Besides that, the predicted and actual price value scale are tightly
> aligned, showing that the NN model achieved a satisfactory result in
> terms of predictive power, compared to KNN and naive bayes model.
> However, there are some drawbacks of neural networks that we cannot
> ignore as well. As a "black box" model, a neural network is unlikely
> to provide a transparent conclusion about the insights that ultimately
> leads to the price changes. Additionally, the computational
> requirement of the Neural Network model is much higher than other
> models due to its complex way of processing data. Compared with other
> models, NN is less interpretable, less computationally efficient, and
> slower in running time. Therefore, even with a high accuracy of
> prediction, we decided that the Neural Network model might not be the
> best option when it comes to the actual deployment in the real world
> market.

**b. Decision Trees and Rationale for Selection**

After evaluating the drawbacks from all of our preliminary models, we
eventually identified Decision Trees as the optimal approach for our
problem statement. Unlike the classification model that categorizes data
into discrete classes, our objective required predicting and capturing
the continuous numerical outcome of the market Price for a given video
game. Given this requirement, we selected the DecisionTreeRegressor from
the sklearn.tree library, rather than the standard
DecisionTreeClassifier that we learned in class. Because we are
predicting the specific price rather than an actual range for the
categorical representation, this regression-based approach allows the
model to minimize the root mean squared error (RMSE) in its predictions.

On the other hand, we ultimately chose the Decision Tree Regressor over
other high-performing models, such as Neural Networks above, mainly due
to the high interpretability and computational efficiency. While Neural
Network captured the relatively same $R^{2}$ compared to the Decision
Tree, it really operates more as a \"black box\", which makes it
difficult to explain the derivation of a specific price to stakeholders
interested in the drivers behind the scene. In contrast, the Decision
Tree offers a more transparent solution where every split can be
visualized through the tags and easily understood through accessible
interpretations by the same high-management level group. Additionally,
the Decision Tree costs way less time to get the outputs and is
significantly more computationally efficient during the training phase
compared to the Neural Network, which required extensive normalization
and complex optimization functions, resulting in a long-processing and
running time before the outputs.

**5. Model Performance Enhancement**

To improve our model\'s predictive power and generalizability, we wanted
to explore deeper into hyperparameters, and go a step further with
Decision Forest.

**a. Parameter Exploration for Decision Trees**

We used Grid Search, Randomized Search, along with Cross Validation to
try out different combinations of hyperparameters to find the best ones
to use. This way, we can balance the simplicity and complexity of our
tree by using the most optimal parameters.

We focused on the following parameters:

-   **Max Depth:** We tried for different values of tree depth to limit
    tree growth.

-   **Min Samples Leaf:** We tested minimum samples in a leaf to make
    sure there is a sufficient number of samples for splitting into leaf
    nodes.

-   **Min Impurity Decrease:** We controlled the minimum impurity
    decrease to ensure there is enough information gain for each split
    of a leaf.

We found the best parameters are: Max Depth = 20, Min Samples Leaf = 50,
Min Impurity Decrease = 0.

The $R^{2}$ improved from 0.922 to 0.958, and RMSE improved from 3.201
to 2.348 with the optimal parameter.

**b. Overfitting Considerations and Decision Forest**

We also tried out Random Forest Regressor to improve the robustness of
our model. Random Forest combines the result from multiple uncorrelated
decision trees, to achieve better results, less variation, and handle
overfitting issues.

The result comparison:

-   **Optimized Decision Tree:** $R^{2}$ = 0.958, RMSE = 2.348.

-   **Random Forest:** $R^{2}$ = 0.959, RMSE = 2.30.

While the Random Forest did provide a marginal improvement in
${\ R}^{2}$ (\<0.01 difference), it came with significant trade-offs. It
has much higher computational requirements, much longer training times,
and a complete loss of visual interpretability. This coincides with the
same reason that we preferred Decision Tree over Neural Networks. Given
that the single optimized Decision Tree achieved nearly identical
performance while remaining transparent and efficient, we confirmed that
it was the superior choice for our final product.

**c. Performance Evaluation and Real-Life Application**

The final optimized Decision Tree model, with a max_depth of 20 and
min_samples_leaf of 50, showed outstanding performance on the validation
data. The $R^{2}$ of 0.958 indicates that the model can explain about
96% of the variance in game pricing based on the provided features.

Real-Life Product Applications:

This predictive model can be useful in many scenarios in game
development settings:

-   **Dynamic Pricing Strategy:** Before launching games, developers can
    input their game\'s attributes to generate a hypothetical market
    price. This ensures the game price point aligns with current market
    standards and consumer expectations.

-   **Resource Allocation & ROI:** By analyzing the decision tree,
    project managers can identify which features act as the primary
    drivers for higher prices. For example, a company might want to
    prioritize extending Game Length over High Soundtrack Quality, since
    it will be more appreciated by its target customers, and generate
    more return on investment. This insight allows studios to allocate
    their budgets toward features that yield the highest return.

### **6. Potential Areas of Improvement and Further Discussions**

Although our Decision Tree model currently shows good performance on
video game pricing with high accuracy and interpretability, there is
still room for us to continue to expand and upgrade the model from a
static predictive tool to a business intelligence that supports dynamic
business decisions.

a.  **Integration of Natural Language Processing (NLP)**

Since numerical ratings are the one we focus for this model, we removed
the *User Review Text* column in our current preprocessing pipeline.
However, this might lead our model to not reflect the full
interpretation of the game's quality. A meaningful enhancement would be
to apply NLP techniques, such as sentiment analysis or keyword
extraction, or apply the Naive Bayes classifier that we covered
previously in determining fraud texts. By identifying qualitative
descriptors such as \"buggy,\" and "shallow", it would allow the model
to differentiate between a game with a low rating due to technical
performance versus one with a low rating due to poor story, which likely
impacts strategic decisions made by the management team of the game.

**b. Separate analysis for non-dominant columns**

![](media/image2.png){width="5.838542213473316in"
height="4.350836614173228in"}

According to our model results, there's really only 2 dominant factors
that capture all the variations in price: *User Rating* and *Game Length
(Hours)*, where their corresponding adjusted $R^{2}$is already 92.4% and
RMSE of 3.181 right after we implement the first tree with max length of
4. In contrast, other characteristics of the game, despite all the
transformation and dummy encoding during the preprocessing stage,
contribute very little in explaining the additional variance. The
columns were precise in definition but lacked the opportunity for
setting up the variations between different games, reducing their
overall predictive impact. One potential approach for that is to exclude
the dominant columns and rerun the regressions based on the remaining
feature, corresponding to the importance table that we have above and
understand better for the contributions of the rest of components in the
dataset. However, this approach is not perfectly suitable for the core
objective of our project outcome and should be treated more as the
supplement for additional exploration in later stages.

**c. Sensitivity Analysis Tool for calculation through budgets**

Beyond simple prediction, the model could be expanded into a \"What-If\"
simulation tool for project managers. By leveraging the decision tree\'s
logic structure, we could build an interface that performs sensitivity
analysis. This will allow the stakeholders to determine critical
development questions like the impact of additional revenue streaming by
raising the Graphics Quality from \'Medium\' to \'High\'. Ultimately,
this transformation would upgrade the model from a purely pricing
estimator into an active tool for Return on Investment (ROI) analysis
during game production and gaining maximum marginal returns.

**d. Validation Against Real Market Data**

A critical consideration for this project is within the nature of the
dataset, which appears to experience a pattern of synthetic or randomly
generated data. Although our model produced an exceptionally high
accuracy ($R^{2}$ = 0.96), this may be partially driven by the
relatively uniform or consistent generations, rather than complex
organic market forces. In the actual video-games market, the outcomes
are influenced by additional\"hidden\" variables that are not inside the
dataset, such as brand loyalty or marketing budgets. To ensure the model
can generalize to the actual video game market, the next phase of
development must involve validation against organic data scraped from
real-world platforms like Steam, the PlayStation Store, or Metacritic.

### 

### **7. Conclusion**

Referring back to the initial proposal of the project, our approach is
really just to deep dive into the video game market and determine an
optimized prediction strategy that the company is able to use for
acquiring the reasonable price for their game. By shifting away from
\"black box\" algorithms like Neural Networks and really just sticking
to the Decision Tree Regressor, we were able to observe the outcome that
really balances accurate predictive power with the transparency and
easy-interperability required for business decision-making.

Throughout the process for optimizing the model, we successfully
addressed the challenges of a diverse dataset, with large chunks of data
transformation involved in maximizing effects of different variables and
improvement discussions dedicated to suiting needs for stakeholders. The
final model confirms that game pricing in this environment is not
arbitrary and it is strongly correlated with tangible quality metrics
and user satisfaction.

Overall, this analysis successfully shows the idea behind our model.
With structured decision logic, we find out that game attributes can be
linked to a reasonable price range. Even though the model is optimized
for the provided database, the frameworks can be expanded into more
practical tools with real market inputs. In addition, studios can apply
this model as a benchmark to compare their games with industry trends,
understanding which feature can influence value the most as well as
optimizing the revenue strategies to improve the profit.

**8. Appendix**

Data Source: [[Video Game Reviews and
Ratings]{.underline}](https://www.kaggle.com/datasets/jahnavipaliwal/video-game-reviews-and-ratings)

Data Feature Overview

![](media/image5.png){width="5.222267060367454in"
height="4.326477471566054in"}
