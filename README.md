<a name="br1"></a>Blaise Hollowaybhollowa 1583872

CMPUT 466 Final Project – ML Algorithm Analysis on Wine Quality

Public GitHub Repo: [hꢀps://github.com/blaiseh33/CMPUT466-Final-Project](https://github.com/blaiseh33/CMPUT466-Final-Project)

**Introducꢀon and Formulaꢀon**:

` `We will aꢀempt to predict the quality of wines based on 11 aꢀributes: ﬁxed acidity,volaꢁle acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide,density, pH, sulphates, and alcohol. The output and ﬁnal aꢀribute is quality on a scale of 0-10.We have 2 datasets found [here](https://archive.ics.uci.edu/ml/datasets/wine+quality)[,](https://archive.ics.uci.edu/ml/datasets/wine+quality)[ ](https://archive.ics.uci.edu/ml/datasets/wine+quality)one with red wines and the other with white wines. Thesewines are variants of the Portuguese “Vinho Verde” wine, and the datasets are not ordered orbalances meaning that there are many more average scored wines compared to low or highscored ones. There are 1599 data entries for the red wine and 4898 entries for the white wine.We will use linear regression, polynomial regression, and neural net regression to test andanalyze the performance of each algorithm for this applicaꢁon. We will use a training-validaꢁon-test infrastructure with hyperparameter opꢁmizaꢁon/tuning. This experiment willalso be used to infer about what the most important features of the wine are in geꢂng a beꢀer-quality score. I am also interested to see how one can predict quality scores of a wine based onsimple chemical and physical data of the wine as this may lead to a conclusion that the qualityscore is arbitrary compared to this data, and it is all up to personal preference regarding whatmakes a wine higher quality than others.

**Approaches and Baselines**:

` `For linear regression using the sklearn.linear\_model we do not have anyhyperparameters to manually opꢁmize or tune as it is already done by the model. We will usethe default model parameters of {‘copy\_X’: True, ‘ﬁt\_intercept’: True, ‘n\_jobs’: None, ‘posiꢁve’:True}. For our polynomial regression, we are once again using the sklearn.linear\_model library,however this ꢁme we will be including a preprocessing step through the same library calledPolynomialFeatures. With this model, we can easily manipulate the degree of the polynomialbefore passing these new features to our linear model for training, validaꢁon, and tesꢁng. Forthe neural net regression model, we can manipulate the hyperparameters {alpha,hidden\_layer\_sizes}. We will test the accuracy, mean squared error, and mean absolute error forall of the algorithms and determine the best hyperparameters for each model. We will alsoconsider overﬁꢂng based on our training error and accuracy compared to our tesꢁng error andaccuracy. Because the regression models will result in real number predicꢁon values, we willsimply round these values to the nearest integer and ensure that these predicꢁons are withinthe quality range [0-10].




<a name="br2"></a>**Evaluaꢀon**:

` `To evaluate the performance of each algorithm we will use accuracy (percentage ofcorrect quality scores), mean squared error, and mean absolute error. Mean squared error andmean absolute error can be calculated through the sklearn metrics library, whereas I will use myaccuracy code from the previous coding assignments. We use the train-validate-testinfrastructure to idenꢁfy the best models for each algorithm through the mean squared errorover mulꢁple epochs. The main goal of this experiment is to determine how well theseregression algorithms can predict the quality scores of the wines, and these quality scores aregiven through professional taste tesꢁng. With that being said, the quality score may not have aperfect correlaꢁon between all or any of the features for the wine, and we can try to determinehow much the feature data inﬂuences the quality score. To do this, we will also test ourpredicꢁons for close accuracy where we allow our wine score predicꢁon to be within 1 to thecorrect quality score. I believe this is a reasonable staꢁsꢁc to observe due to the arbitrarynature of the quality score being given strictly by taste – we cannot know how much each of thefeatures have a direct eﬀect on the taste.

**Results**:

` `The results of the experiment are quite interesꢁng to me, and I believe that these resultsare due to the potenꢁal lack of correlaꢁon between the feature data and the somewhatarbitrary quality score. We can see in the 2 tables below for each wine type that the accuracy(exact correct quality score predicꢁons) are somewhat poor – all algorithms on both data setsare around 50%. Polynomial regression slightly outperforms the other 2 algorithms for accuracy.Close accuracy gives a much beꢀer predicꢁon on the quality as giving a buﬀer of plus or minus 1score leads to a greater than 90% accuracy for most algorithms on average. The neural networkperforms very well with the smaller data set of red wine for close accuracy, whereas thepolynomial regression doesn’t perform quite as well. Overall, I believe that all 3 algorithms aregood potenꢁal candidates for predicꢁng the quality of wines based on the 11 given features.With that said however, I do believe that the poor performance of the exact accuracy is likelydue to the wine quality score not having a direct correlaꢁon to all of the features and data given– the score is inﬂuenced by personal preference and taste. A rough esꢁmate of the score usingclose accuracy can however be used to interpret the quality of these wines. Based on thecoeﬃcients/weights of the ﬁnal model for each algorithm, we can also conclude that thefeatures with the greatest inﬂuence on the quality score are volaꢁle acidity, density, citric acid,pH, and alcohol.



<a name="br3"></a>White Wine:

` `Linear Reg. Polynomial Reg. NN Reg.Mean Squared Error  0.6347 0.6146 0.6971Mean Absolute Error  0.5256 0.4922 0.5545Accuracy  0.5267 0.5579 0.5133Close Accuracy (+-1)  0.9498 0.9565 0.9354

Red Wine:

` `Linear Reg. Polynomial Reg. NN Reg.Mean Squared Error 0.7257 0.9096 0.6086Mean Absolute Error  0.5652 0.5819 0.5083Accuracy  0.4949 0.5451 0.5384Close Accuracy (+-1)  0.9565 0.8996 0.9665

**References**:

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properꢁes. In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

[Scikit-learn:](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)[ ](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)[Machine](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)[ ](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)[Learning](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)[ ](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)[in](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)[ ](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)[Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)[,](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)[ ](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)Pedregosa *et al.*, JMLR 12, pp. 2825-2830, 2011.

hꢀps://datascience.stackexchange.com/quesꢁons/15135/train-test-validaꢁon-set-spliꢂng-in- sklearn
