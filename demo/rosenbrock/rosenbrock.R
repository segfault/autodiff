t.bfgs  <- read.table("rosenbrock.bfgs.table", header=FALSE)
t.rprop <- read.table("rosenbrock.rprop.table", header=FALSE)

a <- 1
b <- 100
f <- function(x1, x2) (a - x1)^2 + b*(x2 - x1^2)^2

x <- seq(-1.5, 2, length=200)
y <- seq(-0.5, 3, length=200)
z <- outer(x, y, f)

png("rosenbrock.png", width=900)
par(mfrow=c(1,2))
image(x, y, z, col=heat.colors(200)[40:200], main="BFGS")
contour(x, y, z, add=T, col="white", nlevels=20)
points(t.bfgs,  type="b")

image(x, y, z, col=heat.colors(200)[40:200], main="Rprop")
contour(x, y, z, add=T, col="white", nlevels=20)
points(t.rprop,  type="b")
dev.off()
