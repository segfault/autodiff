
t <- read.table("bfgs_test1.table", header=FALSE)

f <- function(x1, x2) 0.26*(x1^2 + x2^2) - 0.48*x1*x2

x <- seq(-3, 3, length= 100)
y <- x
z <- outer(x, y, f)

image(x, y, z, col=terrain.colors(100))
points(t, type="b")
