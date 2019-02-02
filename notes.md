# Notes on the kaggle Challange

- The data Ids are unique
- User Day of birth (user_dob) and delivery date have missing values
- correlation analysis not really possible, as they are only two relevant 
    numerical variables (Ids excluded)
- ~96% of the users are "Mrs"
- item_price has a lot of outliers, infact, 
    25% of the data are above the 75% quantile (which may be logical, but I
    dunno)
    - a standardization of that value may be a good idea (or scaling, or 
        normalization)
    - price is not normally distributed, but has a linksteile skewity(?)
- ~1% of delivery_dates are 1994 => error in data
- ~8.2% of delivery_dates is null
 