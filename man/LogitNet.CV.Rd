\name{LogitNet.CV}
\alias{LogitNet.CV}
\title{Fit LogitNet models with cross validation.}
\description{Fit LogitNet models for a series of tuning parameters and return the cross valication error.}
   
\usage{
LogitNet.CV(X.m, weight.m, lambda.v, fold=5)
}

\arguments{
    \item{X.m}{numeric matrix (n by p). Columns are for variables and rows are for samples. 
                     Missing values are not allowed.}  
    \item{weight.m}{numeric matrix (p by p). This weight matrix allows the coefficients in the regression model to be penalized to varying degree,
                     such that the spatial correlation along the genome profiles is taken into account. It can use the output from LogitNet.weight function.}  
    \item{lambda.v}{numeric vector. It is a sequence of \eqn{l_1} norm penalty parameters.} 
    \item{fold}{numeric scaler. It specifies the fold number in cross validation. The default is 5.}
          }        
\details{
     LogitNet.CV helps to select the tuning parameter through cross validation. LogitNet is developed for infering interaction network of binary variables. The method 
     is based on penalized logistic regression with an extension to account for spatial correlation in the genomic instability data. (Wang, Chao and Hsu, 2009).}
\value{
   A list with four components
    \item{beta_reg}{A numeric array with dimention (P, P, Fold, lambda.n), which records the 
                     estimated coefficient matrix at each lambda from the penalized model. Here \code{lambda.n} is the lenght of \code{lambda.v}.}
    \item{beta_unbias}{A numeric array with dimention (P, P, Fold, lambda.n), which records the estimated coefficient matrix 
                            at each lambda from the un-penalized model 
                           (refit LogitNet only using the selected variables).}
    \item{likelihood.test}{A numeric matrix (Fold by lambda.n), which records the likelihood of each logistic regression on the testing data for each cross validation fold.}
    \item{likelihood.train}{A numeric matrix (Fold by lambda.n), which records the likelihood of each logistic regression on the training data for each cross validation fold.}
    }

\references{
 Pei Wang, Dennis Chao, Li Hsu, "Learning oncogenic pathways from binary genomic instability data", Biometrics, (submitted 2009, July)
}

\author{Pei Wang, Dennis Chao, Li Hsu}

\keyword{methods}

\examples{

######################## get data example

data(LogitNet.data)
data.m=LogitNet.data$data.m
chromosome=LogitNet.data$chromosome
p=ncol(data.m)

######################## specify the penalty parameter
lambda.n=5
lambda.v=exp(seq(log(13), log(30), length=lambda.n))

######################## calculate the weight matrix
w.m=LogitNet.weight(data.m, chr=chromosome) 

######################## perform cross validation to select lambda
if(0) ### this part will take 10 minutes.
{
try.CV=LogitNet.CV(data.m, w.m, lambda.v, fold=5) 
temp=apply(try.CV[[3]], 2, sum) 
index=which.max(temp) 
}
index=2
######################## estimate the model at selected lambda

result=LogitNet(data.m, w.m, lambda.v[index]) ###20-30 seconds

######################## illustrate the result similar to Figure 3 of Wang et al. (2009)).

temp=result
diag(temp)=0

par(cex=1.8)
image(1:p, 1:p, temp!=0, col=c("white", "red"), axes=FALSE, xlab="Marker Loci", ylab="Marker Loci")
abline(h=(0:5)*p/6+p/6/2, col=4, lty=3, lwd=0.8)
abline(v=(0:5)*p/6+p/6/2, col=4, lty=3, lwd=0.8)
axis(1, at=c(1,1:6*100), labels=c(1,1:6*100))
axis(2, at=c(1,1:6*100), labels=c(1,1:6*100))
axis(3, at=(0:5)*p/6+p/6/2, labels=c("A", "B", "C", "D", "E", "F"), col.axis=4, tick=FALSE)
axis(4, at=(0:5)*p/6+p/6/2, labels=c("A", "B", "C", "D", "E", "F"), col.axis=4, tick=FALSE)

lab.v=c("A", "B", "C", "D", "E", "F")

cut=30
for(i in 0:4)
{
   cur=i*p/6+p/6/2
   cur2=(i+1)*p/6+p/6/2

   x.cur=c(cur-cut, cur, cur+cut, cur)
   y.cur=c(cur2, cur2-cut, cur2, cur2+cut)
   polygon(x.cur, y.cur, border=grey(0.5))
   polygon(y.cur, x.cur, border=grey(0.5))
}

}
