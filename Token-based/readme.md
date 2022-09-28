Here is our process:

1. We combine the sentence to a long string, and add two * symbols in the head and tail of the sentence separately.
For example: "I have a dog" would become "**I have a dog**"

2. We use the 3-grams method to split the string into many three-character pieces and set these pieces into a list.

3. We calculate the overlap coefficient, Jaccard coefficient, and Dice's coefficient.

4. We would compare the similarity of these lists using the overlap coefficient, Jaccard coefficient, and Dice's coefficient.

---------------------------------------------------------------------------------------------
How to choose the threshold

First, I draw the scatter plot for each coefficient: X-axis means the i-th coefficient; Y-axis means the value of the coefficient. Red color means the two sentences are matched; Blue means that the tow sentences not matched.
       The pictures show that the overlap coefficient is the best.

Second, I combine these three coefficients to a new coefficient, and the weights of the three coefficients are the same. (Like: coefficient_new=1/3*coefficient_overlap+1/3*coefficient_jaccard+1/3*coefficient_dice) Then I use the test set to acculate the accuracy based on such coeffficient.

Third, I try to use a classifier, which I use three coefficient as 3-dimensional input to predict the output label.
