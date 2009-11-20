\name{LogitNet.weight}
\alias{LogitNet.weight}
\title{Derive the weight matrix for fitting the LogitNet model.}
\description{Derive the weight matrix for fitting the LogitNet model.}
   
\usage{
LogitNet.weight(X.m, chr)
}

\arguments{
    \item{X.m}{numeric matrix (n by p). Columns are for genes/loci and rows are for samples. 
                     Missing values are not allowed. The genes/loci are ordered according to their positions on the genome. }  
    \item{chr}{numeric vector of length p. This vector gives the chromosome information of each gene/locus.}  
          }        
\details{
     This function returns a weight matrix charactering the spatial correlations along the genome. This matrix provides the value for one input parameter of function LogitNet().
     LogitNet is developed for infering interaction network of binary variables. The method is based on penalized
logistic regression with an extension to account for spatial correlation in the genomic instability data. (Wang, Chao and Hsu, 2009).}

\value{
    \item{w.s}{numeric matrix (p by p), which characterizes the spatial correlation along the genome for each gene/locus.}
    }

\references{
 Pei Wang, Dennis Chao, Li Hsu, "Learning oncogenic pathways from binary genomic instability data", Biometrics, (submitted 2009, July)
}

\author{Pei Wang, Dennis Chao, Li Hsu}

\keyword{methods}

\examples{

######################## obtain a data example

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
######################## estimate the model at selected lambda values

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
