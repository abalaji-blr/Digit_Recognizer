# use random forest to classify the digit

library("randomForest")
library("data.table")
library("caret")

set.seed(123)

# read the dataset
train <- fread("../dataset/train.csv")
test <- fread("../dataset/test.csv")

# get the training examples and lables
samp <- sample(1:nrow(train), nrow(train))

train_sample <- train[samp, -1, with=FALSE]

train_label <- as.factor(train[samp,]$label)

# build model and predict
rfm <- randomForest(train_sample, train_label, xtest=test, ntree = 25)

result <- data.table(ImageId=1:nrow(test), Label=levels(train_label)[rfm$test$predicted])

write.csv(result, "rf_submit.csv", row.names = FALSE, quote = FALSE)
