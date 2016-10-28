## Digit recognizer using nnet

library(data.table)

library(nnet)
library(caret)  #varImp
library(Metrics)

train <- fread("../dataset/train.csv", header = TRUE)

# train using neural network
num_training_examples <- 40000

# input data
#train_data <- train[1:num_training_examples, -1, with=FALSE]



# build model with multinom
# multinom builds network with zero hidden layer
nn <- multinom(label ~ ., data = train[1:num_training_examples], MaxNWts = 12000, trace=T)
names(nn)

# use caret package & print top 5 features
#
model_feat <- varImp(nn)
model_feat$Features <- row.names(model_feat)
top_features <- model_feat[order(-model_feat$Overall),]
head(top_features, 5)


# perfrom cv validation
cv <- train[(num_training_examples+1):nrow(train),]

# do predict
pred <- predict(nn, newdata = cv[, -1, with=FALSE])
head(pred)

## get the accuracy
postResample(cv$label, pred)

# metric
met <- confusionMatrix( data = pred, reference = cv$label)
names(met)
met$table

################################################
### use nnet & set the hidden layer ############

# R format formula is not accepted by neural net.
# so, do the following

# except the target variable
feats <- names(train)[-1]

# concatenate the columns
f <- paste(feats, collapse = " + ")
f <- paste("label ~ ", f)

myFormula <- as.formula(f)


# class indicator
train_label <- class.ind(train$label[1:num_training_examples])
hidden_layer_size <- 15
mydata <- train[1:num_training_examples, -1, with=FALSE]

nn2 <- nnet(mydata, train_label, size = hidden_layer_size, softmax = TRUE, MaxNWts = 12000, trace=T)
names(nn2)
nn2$n

# do the prediction using new neural net nn2
# do predict
pred2 <- predict(nn2, newdata = cv[, -1, with=FALSE])
pred2_log <- predict(nn2, newdata = cv[, -1, with=FALSE], type = "class")
head(pred2)

## get the accuracy
postResample(cv$label, as.numeric(pred2_log))

met2 <- confusionMatrix( data = pred2_log, reference = cv$label)
names(met2)
met2$table
met2$overall

## test

test <- fread("../dataset/test.csv", header = TRUE)

test_pred <- predict(nn2, newdata = test, type = "class")
head(test_pred)

result_dt <- data.table(ImageId = 1:nrow(test), Label = as.numeric(test_pred))

# create submit file
write.csv(result_dt, file = "./submission.csv", row.names = FALSE, quote = FALSE)




