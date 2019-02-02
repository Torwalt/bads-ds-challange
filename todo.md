# Todo list for the kaggle Challange

1. Split the known data into training and test data
2. Implement Cross-Validation, Boosting, Bootstrapping
    (maybe the same as other points) and Ensemble Learning
    - for smaller datasets, Leave One Out Cross Validation (LOOCV) is
        recommeded
3. Performing a sensitivity analysis of model performance in terms of error
    costs (for term paper)
4. How to include some regularization (do I need to include it?)
    - regularization penalizes overfitting, thus pernalizes model complexity
5. Think about how you could reduce the number of categories in the item sizes
    => some kind of business logic should be applied here e.g. map the numeric
    sizes to their character representations

Tips:

- Random Forest or XG Boost are very good

Left off:

- train xg_boost data (do I have to?)
- is error the same as the accuracy inverted (probably yes)