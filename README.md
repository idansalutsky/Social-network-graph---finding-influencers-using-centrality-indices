# Social-network-graph-finding-influencers-using-centrality-indices

The purpose of this task is for the highest level of exposure within a social network, in order to market artists' merchandise products. This will be done by choosing the 5 best influencers to collaborate with who will first receive the merchandise.

I'll start from the end - the initial thought was that we would like to find different groups from each other by choosing influencers with a "different audience", this assumption was right and wrong at the time.
Correct because for the "less connected" influencers the assumption is
was correct in that they were from different groups of people. The assumption is also incorrect because we received 3 influencers who simply manage to reach too many people who simply cannot be separated to a large extent from the other influencers.

I will now continue,
During the task we used the networkx package in order to represent the graph and perform calculations on it.

Assumption - when a user buys/receives merchandise of a certain artist, his friends on the network see this and also decide to buy it with probability p (a calculation that only takes into account neighbors with the product if the user has not listened in time, and another calculation that takes into account both playbacks and friends linked to the product for a user who has listened to the artist).

The test is performed for the largest customer base after 6 time steps of "sticking" to the users' purchases, that is, if one user bought a product at time t, the user who is linked to him will see this at the time step t+1.

The data are two tables-
1) A social network at two consecutive times t=0 and t=1-f, each point in time is described by a table in which two columns contain unique identifiers of people, each row describes a connection between the two people.
2) Data from a music application - described by 3 columns - user ID, Artist ID and number of playbacks the user listened to the artist.

solution-
1) Formation of edges - in order to refer to new bonds that are formed and the probability thereof, we performed a simple linear regression for the proportion of formation of edges - the edges that were formed and the total number of edges that had the potential to be formed (the members of the mutual members), where the variable is the number of the common members.
2) For finding the influencers, we found that in the graph there is a very large binding component, and the size of all the other components is very small, therefore we focused the finding of the influencers only on the large binding component.

For this component we calculated four centrality indices in the graphs - Degree, Closeness, Harmonic, Betweenness In addition to the centrality indices we calculated the "page rank" index for each node and the radius of the binding component, we found that the radius is 5 therefore we concluded that it is possible to reach every node in the binding component during the running of the program Because we run for 6 time periods.
We found that there are 8 central nodes in the graph, therefore we also searched for each centrality index the 8 nodes with the highest score, and left only the nodes that belonged to most of the indices we tested.

We accepted that the three most central nodes exist for each artist, therefore we searched for which two additional nodes for each artist would add the most adhesions, we used an algorithm
Hill Climbing on the three central nodes together with any additional permutation of two nodes out of all the other central nodes we found in the two central tests.
