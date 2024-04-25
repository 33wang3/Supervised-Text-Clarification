# Load required libraries
install.packages("caret")
library(caret)
library(tm)

## STEP 1: Import and clean the text data
# Load the dataset
health_data <- read.csv("HBM.csv") # Update the path to your actual file location

# Inspect the data to understand its structure
head(health_data) 

# The text column is named "Tweet" as per your document
health_text <- health_data$Tweet

# Remove non-ASCII characters (emojis etc.), if not relevant
health_text <- gsub("[^\x20-\x7e]", " ", health_text)

# Remove URLs, special characters, and numbers
health_text <- gsub("(@|http)[^[:blank:]]*|[[:punct:]]|[[:digit:]]", " ", health_text)
health_text <- gsub("\\s+", " ", health_text)

# Define and customize stopwords, possibly adding domain-specific terms
myStopwords = c(stopwords('english'), 'description', 'null', 'text', 'url', 'href', 'nofollow', 'false', 'true', 'rt', 'health')

## Create a document-term matrix for analysis
text_corpus <- VCorpus(VectorSource(health_text))
textDTM <- DocumentTermMatrix(text_corpus, list(tolower = TRUE, removePunctuation = TRUE, removeNumbers = TRUE, stopwords = myStopwords, stemming = TRUE))

# Partition the data into training and testing sets based on topics
set.seed(2024)
train_index <- createDataPartition(health_data$Topic, p = .6, list = FALSE)

# Create training and testing document-term matrices
trainingDTM <- textDTM[train_index, ]
testingDTM <- textDTM[-train_index, ]

# Convert DTM to matrices for analysis
training <- as.matrix(trainingDTM)
testing <- as.matrix(testingDTM)

# Append the labels to the training matrix
health_training <- health_data[train_index,]
training <- cbind(training, Label = health_training$Topic)

# Convert to data frame and factorize the labels
training <- as.data.frame(training)
training$Label <- as.factor(training$Label)

# Save the training data frame
write.csv(training, "health_training.csv")

### TRAINING with Supervised Machine Learning
# Example using Support Vector Machines
set.seed(2024)
svm_model <- train(Label ~ ., data = training, method = "svmLinear3")

## PREDICTION
test_pred <- predict(svm_model, newdata = testing)

## EVALUATION
# Evaluate model performance
health_testing <- health_data[-train_index, ]
test_truth <- health_testing$Topic

# Create confusion matrix and calculate performance metrics
test_pred <- factor(test_pred)
test_truth <- factor(test_truth)
confusionMatrix(test_pred, test_truth)

# Prepare a data frame with test data, ground truth, and predictions for analysis
pred_res <- as.data.frame(cbind(health_testing, Prediction = test_pred))
colnames(pred_res)[colnames(pred_res) == "Topic"] <- "Truth"

# Export the results
write.csv(pred_res, "prediction_results.csv", row.names = FALSE)

# Load the new tweet dataset
new_tweets <- read.csv("left_data.csv") # Update the path to your file

# Preprocess the text data
new_tweets_text <- new_tweets$Tweet
new_tweets_text <- gsub("[^\x20-\x7e]", " ", new_tweets_text)
new_tweets_text <- gsub("(@|http)[^[:blank:]]*|[[:punct:]]|[[:digit:]]", " ", new_tweets_text)
new_tweets_text <- gsub("\\s+", " ", new_tweets_text)

# Create a corpus from the cleaned text
new_corpus <- VCorpus(VectorSource(new_tweets_text))
new_corpus <- tm_map(new_corpus, content_transformer(tolower))
new_corpus <- tm_map(new_corpus, removePunctuation)
new_corpus <- tm_map(new_corpus, removeNumbers)
new_corpus <- tm_map(new_corpus, removeWords, myStopwords)
new_corpus <- tm_map(new_corpus, stemDocument)

# Convert the new corpus into a document-term matrix, matching the original training data
new_dtm <- DocumentTermMatrix(new_corpus, control = list(dictionary = Terms(textDTM)))

# Convert new DTM to matrix for prediction
new_matrix <- as.matrix(new_dtm)

# Predict the topics using the trained SVM model
new_predictions <- predict(svm_model, newdata = new_matrix)

# Add the predicted labels to the new tweets dataframe
new_tweets$Predicted_Topic <- new_predictions

# Export the new tweets with predictions
write.csv(new_tweets, "new_tweets_with_predictions.csv", row.names = FALSE)

