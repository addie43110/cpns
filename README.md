# Polynomial Time Algorithm for Deciding Achievable Token Mass in Continuous Petri Nets
Addie Jordon, Juri Kolčák, Daniel Merkle

Universität Bielefeld, Bielefeld, Germany

2025

## Description

Determining the maximum yield of a product from a set of starting molecules is of interest in chemistry. Ordinary differential equations (ODEs) and flux balance analysis (FBA) are two popular methods for metabolism modeling, but ODEs require extensive knowledge about parameters (which may be unknown) and FBA provides no causal analysis for its solutions.

We thus propose continuous Petri nets (CPNs) as a new way of modeling chemical reaction networks and maximizing yield with causal realizability. A continuous Petri net is a directed graph whose nodes can be split into two types: places and transitions. In the context of chemical networks, places are used to represent molecules and transitions are used to represent the reactions between molecules. Places have non-negative real token mass which moves around the net when transitions are activated (fired). 

## `AtLeastReachable`

`AtLeastReachable` decides if at least x (x>0) token mass is reachable on a goal place given a set of initial molecule quantities. For example, in the pentose phosphate pathway, we can start with 1.0 token mass on ribose-6-phosphate and 1.0 token mass on H2O and then ask if at least 0.3 token mass can be moved to fructose-6-phosphate. Furthermore, our algorithm achieves this decision in polynomial running time.

Note: since `AtLeastReachable` only returns true or false, another algorithm must be added to find the maximum amount of token mass which can be placed on the goal compound. For this, we used binary search.

## `MILPMax`

The solutions to `AtLeastReachable` are, to keep it simple, large and messy. They often contain redundant firings of transitions that play no part in maximizing the yield. 

To mitigate this, we used mixed-integer linear programming (MILP) which allowed us to order solutions by their length, with minimal size solutions being most desirable. Since MILP is NP-hard, the running time is no longer polynomial. It should be noted, however, that in practice, `MILPMax` often runs faster than `AtLeastReachable` due to the large overhead required in `AtLeastReachable`.
