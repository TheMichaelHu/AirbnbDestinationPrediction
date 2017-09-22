# Project Proposal: Yelp Cuisine Classifier
By Michael Hu and David Zhao

## Introduction
In America, there are often assumptions and expectations that come bundled with food from different cultures, as if these cuisines have an identity of their own that span across restaurants. When we are looking for a fancy dinner or a quick meal at 2am, do specfic cuisines come to mind? We are curious if restaurants serving food from different cultures (Japanese, Thai, French, Mediterranean, etc.) can be modeled by looking at factors such as their price, hours of operation, names, etc., and, if so, which features contribute the most to how society views different cuisines. 

## Data
Our project will use the [Yelp Open Dataset](https://www.yelp.com/dataset), a source containing information on upwards of 150,000 businesses on Yelp, and nearly 5 million user reviews. We will be primarily interested in the features of restaurants themselves, but may also consider processing user reviews. However, due to the underlying inspirations of this project, we will avoid mining user reviews for words like "Italian", "braised pork", "halal", or any other tokens that may indicate the cuisine of a restaurant solely because of how strongly coupled they are with that cuisine.

## Justification
Having society fit a demographic's food and culture to a mold may serve as a barrier preventing restaurants that wish to break that mold from succeeding. If we are able to train a classifier that can accurately predict the cuisine of a restaurant based solely on its features, that may lead to research and awareness of pressures and expectation we as a society are imposing on restaurants serving the food of particular demographics
