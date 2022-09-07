# master_project_entity_matching

Questions：

1.For the large data set, it is hard for us to calculate a matrix(21000*21000) to achieve the pair match. We discussed about several solutions:

(1)Randomly shuffle the entities 10 times, each time choosing 2100 rows of them to calculate and compare these with the correct results.

(2)After calculating the whole matrix (21000*21000), we have to find the minimum distance in the matrix and reduce the size of matrix by merging this smallest distance.---still need a huge calculation at first.


2.We tried to combine several attributes into one string, but there are several missing items in different attributes. So we may need to use different weight for each attribute to get a combined distance. How can we choose these weight? In this case, for the two entity, if they are the same id, but one has 4 attributes and one has 1, how can we do?

Report link:
https://www.overleaf.com/project/630f46b9d6a9fff7b55d3dd2
