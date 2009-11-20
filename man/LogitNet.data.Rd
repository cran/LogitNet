\name{LogitNet.data}
\alias{LogitNet.data}
\title{Example Data for LogitNet package}
\description{A list containing an example data for package \code{LogitNet}}

\value{
LogitNet.data is a list of two components: 
\item{data.m}{a numeric matrix consisting of 200 rows (samples) and 600 columns (genes).}
\item{chromosome}{a numeric vector of length 600.} 
}

\details{
\code{data.m} is an simulated example based on the chain pathway described in Section 3 of Wang et al. 2009. 
}

\references{
 Pei Wang, Dennis Chao, Li Hsu, "Learning oncogenic pathways from binary genomic instability data", Biometrics, (submitted 2009, July)
}

\usage{data(LogitNet.data)}

\keyword{datasets}

