# TinyImageNetPlayground
This is a tool which I created to test out different tiny-imagenet classification models for my computer vision module.

I reduced the number of classes from the usual 200 to 50. Since this is not an attempt at building a competative network, I extracted part of the training set to use as a test set. As a result, this code requires that the dataset is laid out in the following format:
```
+-- train
|   +-- n01443537
|   +-- n01629819
|   +-- etc

+-- val
|   +-- images
|   +-- |   +-- n01443537
|   +-- |   +-- n01629819
|   +-- |   +-- etc

+-- test
|   +-- n01443537
|   +-- n01629819
|   +-- etc
```
