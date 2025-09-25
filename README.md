# FPL Optimisation Project

Someone I know recently started getting into Fantasy Premier League (FPL). He explained to me that some players get more points than others, and that it involves trying to make the best team given only 100 million pounds.
My instant thought was: This sounds like a mathematical problem!
So, here we are! The quest to achieve the mathematically perfect FPL team!

# Side-Notes
- Originally, hyperparameter tuning was added to the Random Forest Regression model. However, after testing, it became apparent that, despite this improving the model's r2 score, it decreased the accuracy of the model in practice.
- I tried to implement XGBoost, however this seemed to make the model significantly less accurate.

# TODO

 - [X] Restructure project
 - - [X] Move all data to be stored in 1 JSON file, significantly simplifying project structure
 - [X] Potentially add XGBoost to Random Forest
 - [X] Create validation dataset
 - - [X] Have abstract `predict` method of some sort
 - [ ] Make HTML saving more maintainable
 - [ ] Potentially have RF Team Solver calculate different RFs for different positions
 - - [ ] Experiment more with different independent variables used for RF
 - [ ] Get opposition teams for old gameweeks
 - [ ] Implement Unit Tests
 - Improve accuracy (ongoing task)