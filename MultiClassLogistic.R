MultiClassLogistic <- function(X, Y,learning_rate = 0.005, numiterations = 10000, Xtest) {
#### Applying Logistic regression first
#### Defining Sigmoid function
sigm <- function(z) {
    1/(1+ exp(-z))
}
h <- list()
m <- dim(X)[2]
if ((class(X))[[1]] != "matrix") {
  X <- as.matrix(X)
}
#### Converting data table Y to matrix
if ((class(Y))[[1]] != "matrix") {
  Y <- as.matrix(Y)
}
#### Converting data table Xtest to matrix
if ((class(Xtest))[[1]] != "matrix") {
  Xtest <- as.matrix(Xtest)
}

run_logistic <- function(Y) {
theta <- matrix(rep(0, dim(X)[1]),nrow = dim(X)[1], ncol = 1)
theta <- t(theta)
b <- 0
for (iter in 1:numiterations) {
  powerp <- (theta %*% X) + b
  #### Defining yhat which is sigmoid function
  yhat <- sigm(powerp)
  #### Defining Logistic loss
  L <- -Y %*% t(log(yhat)) - ((1-Y) %*% t(log(1-yhat)))
  #### Defininf the total cost 
  cost <- (1/(1*m))*(sum(L))
  if (iter %% 100 == 0) {
    print(cost)
  }
  grad_theta <- (1/(1*m))*((yhat - Y) %*% t(X))
  grad_b <- (1/(1*m))*sum((yhat - Y) %*% t(yhat-Y))
  theta <- theta - (learning_rate*grad_theta)
  b <- b - (learning_rate*grad_b)
}
#### Predicting the response as probabilities for Xtest
predictedprob <- sigm((theta %*% Xtest) + b)
return(list(theta,b, predictedprob))
}
g <- NULL
for (i in 1:dim(Y)[1]) {
  h[[i]] <- run_logistic(matrix(Y[i,], nrow = 1))
  print("Next classifier....")
  # print(head(h[[i]]))
  # print(h[[i]][3])
  g <- rbind(g, h[[i]][3][[1]])
}
g <- as.data.table(g)
predictedclass <- g[, lapply(.SD, which.max)]

return(list(h,g,predictedclass))
}
